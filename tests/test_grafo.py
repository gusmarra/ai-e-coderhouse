from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from app.agents.supervisor import aplicar_guardas
from app.base_hotel import BuscadorHotel, cargar_chunks
from app.config import MAX_PASOS
from app.graph import construir_grafo, enrutar, estado_inicial
from tests.fakes import ModeloGuionado, decision, llamada
from tests.trazas import ejecutar

PREGUNTA = "4 noches a USD 150, temporada alta, cancela 3 dias antes: ¿penalidad?"
LIMITE: RunnableConfig = {"recursion_limit": 25}


@pytest.fixture(scope="module")
def buscador() -> BuscadorHotel:
    return BuscadorHotel(cargar_chunks())


def _especialistas(*extra: AIMessage) -> ModeloGuionado:
    """Guion base: investigador (busca + responde), analista (calcula + responde)."""
    return ModeloGuionado(
        guion=[
            llamada("buscar_en_documentos_hotel", "i1", consulta="temporada alta cancelacion penalidad"),
            AIMessage("Penalidad: 50% de la estadía reservada (fuente: politica-reservas-cancelaciones.md)."),
            llamada("calcular", "a1", expresion="0.5 * 4 * 150"),
            AIMessage("Penalidad = 0.5 * 4 * 150 = USD 300."),
            *extra,
        ]
    )


def _supervisor(*decisiones: object) -> ModeloGuionado:
    return ModeloGuionado(
        guion=[AIMessage("La penalidad es de USD 300 (politica-reservas-cancelaciones.md).")],
        decisiones=list(decisiones),  # type: ignore[arg-type]
    )


async def test_flujo_completo_investigador_analista_sintesis(buscador: BuscadorHotel) -> None:
    sup = _supervisor(
        decision("investigador", "Buscar la penalidad de temporada alta"),
        decision("analista", "Calcular la penalidad"),
        decision("FINISH"),
    )
    grafo = construir_grafo(sup, llm_especialistas=_especialistas(), buscador=buscador)

    traza = await ejecutar(grafo, PREGUNTA)

    assert traza.ruta == ["supervisor", "investigador", "supervisor", "analista", "supervisor", "sintesis"]
    assert [c["agente"] for c in traza.contribuciones] == ["investigador", "analista"]
    assert traza.contribuciones[0]["herramientas"] == ["buscar_en_documentos_hotel"]
    assert traza.contribuciones[1]["herramientas"] == ["calcular"]
    assert "USD 300" in traza.respuesta
    assert traza.pasos == 2


async def test_estado_final_conserva_el_aporte_de_cada_agente(buscador: BuscadorHotel) -> None:
    sup = _supervisor(decision("investigador", "x"), decision("analista", "y"), decision("FINISH"))
    grafo = construir_grafo(sup, llm_especialistas=_especialistas(), buscador=buscador)

    final = await grafo.ainvoke(estado_inicial(PREGUNTA), LIMITE)

    assert set(final["hallazgos"]) == {"investigador", "analista"}  # el reducer fusiona, no pisa
    assert "politica-reservas-cancelaciones.md" in final["hallazgos"]["investigador"]["fuentes"]
    assert final["task_completed"] is True
    assert [m.name for m in final["messages"][1:]] == ["investigador", "analista", "supervisor"]


async def test_el_analista_recibe_la_consigna_y_los_datos_no_el_historial(buscador: BuscadorHotel) -> None:
    esp = _especialistas()
    sup = _supervisor(decision("investigador", "buscar"), decision("analista", "calcular"), decision("FINISH"))
    await construir_grafo(sup, llm_especialistas=esp, buscador=buscador).ainvoke(estado_inicial(PREGUNTA), LIMITE)

    primera_del_analista = esp.recibidos[2]  # 0-1: investigador, 2: analista
    assert [m.type for m in primera_del_analista] == ["system", "human"]  # sin ToolMessages ni charla ajena
    texto = str(primera_del_analista[1].content)
    assert "Tu consigna ahora: calcular" in texto
    assert "50% de la estadía" in texto  # el hallazgo del investigador si le llega


async def test_el_supervisor_ve_hallazgos_con_fuentes_pero_no_trazas_de_herramientas(
    buscador: BuscadorHotel,
) -> None:
    sup = _supervisor(decision("investigador", "x"), decision("analista", "y"), decision("FINISH"))
    await construir_grafo(sup, llm_especialistas=_especialistas(), buscador=buscador).ainvoke(
        estado_inicial(PREGUNTA), LIMITE
    )
    tercera_decision = str(sup.recibidos[2][1].content)
    assert "[investigador] version 1 · fuentes: politica-reservas-cancelaciones.md" in tercera_decision
    assert "[analista] version 1" in tercera_decision
    assert "(fuente: politica-reservas-cancelaciones.md, seccion" not in tercera_decision  # output crudo de la tool


async def test_refinamiento_el_supervisor_devuelve_el_trabajo_al_analista(buscador: BuscadorHotel) -> None:
    sup = _supervisor(
        decision("investigador", "buscar"),
        decision("analista", "calcular"),
        decision("analista", "Falta indicar el plazo sin cargo", "rubrica 3"),
        decision("FINISH"),
    )
    esp = _especialistas(AIMessage("USD 300; plazo sin cargo: hasta 7 días antes."))
    grafo = construir_grafo(sup, llm_especialistas=esp, buscador=buscador)

    final = await grafo.ainvoke(estado_inicial(PREGUNTA), LIMITE)

    assert [c["agente"] for c in final["contribuciones"]] == ["investigador", "analista", "analista"]
    assert final["hallazgos"]["analista"]["version"] == 2
    assert "7 días" in final["hallazgos"]["analista"]["contenido"]


async def test_supervisor_infinito_el_tope_de_refinamientos_fuerza_el_cierre(buscador: BuscadorHotel) -> None:
    """Un supervisor que siempre pide 'una correccion mas' no puede colgar el sistema."""
    sup = _supervisor(decision("investigador", "buscar"), decision("analista", "otra vez"))  # repite el ultimo
    grafo = construir_grafo(sup, llm_especialistas=_especialistas(), buscador=buscador)

    traza = await ejecutar(grafo, PREGUNTA)

    assert traza.ruta[-1] == "sintesis"
    assert traza.ruta.count("analista") == 2  # 1 pasada + 1 refinamiento, no mas
    assert traza.pasos <= MAX_PASOS


async def test_supervisor_que_solo_pide_investigar_tambien_termina(buscador: BuscadorHotel) -> None:
    sup = _supervisor(decision("investigador", "de nuevo"))
    traza = await ejecutar(construir_grafo(sup, llm_especialistas=_especialistas(), buscador=buscador), PREGUNTA)
    assert traza.ruta.count("investigador") == 2
    assert traza.ruta[-1] == "sintesis"


def test_guarda_el_analista_no_arranca_sin_datos() -> None:
    corregida = aplicar_guardas(decision("analista", "calcular"), estado_inicial(PREGUNTA))  # type: ignore[arg-type]
    assert corregida.siguiente == "investigador"


def test_guarda_no_se_cierra_sin_haber_investigado() -> None:
    corregida = aplicar_guardas(decision("FINISH"), estado_inicial(PREGUNTA))  # type: ignore[arg-type]
    assert corregida.siguiente == "investigador"


def test_guarda_tope_de_pasos_fuerza_finish() -> None:
    estado = estado_inicial(PREGUNTA) | {"pasos": MAX_PASOS}
    assert aplicar_guardas(decision("investigador", "x"), estado).siguiente == "FINISH"  # type: ignore[arg-type]


def test_guarda_completa_consigna_vacia_con_el_pedido_original() -> None:
    hallazgo = {"contenido": "x", "fuentes": [], "version": 1}
    estado = estado_inicial(PREGUNTA) | {"hallazgos": {"investigador": hallazgo}}
    assert aplicar_guardas(decision("analista", ""), estado).instruccion == PREGUNTA  # type: ignore[arg-type]


def test_enrutar_mapea_finish_al_nodo_sintesis() -> None:
    assert enrutar({"next_agent": "FINISH"}) == "sintesis"  # type: ignore[typeddict-item]
    assert enrutar({"next_agent": "analista"}) == "analista"  # type: ignore[typeddict-item]


def test_el_diagrama_refleja_la_topologia() -> None:
    mermaid = construir_grafo(_supervisor(decision("FINISH"))).get_graph().draw_mermaid()
    for nodo in ("supervisor", "investigador", "analista", "sintesis"):
        assert nodo in mermaid


def test_toda_accion_del_ejecutor_pasa_por_la_aprobacion_humana() -> None:
    assert enrutar({"next_agent": "ejecutor"}) == "aprobacion"  # type: ignore[typeddict-item]


def test_el_diagrama_incluye_el_nodo_hitl_y_el_ejecutor() -> None:
    mermaid = construir_grafo(_supervisor(decision("FINISH"))).get_graph().draw_mermaid()
    assert "aprobacion" in mermaid
    assert "ejecutor" in mermaid


def test_guarda_el_ejecutor_no_corre_dos_veces() -> None:
    hallazgo = {"contenido": "x", "fuentes": [], "version": 1}
    estado = estado_inicial(PREGUNTA) | {
        "hallazgos": {"investigador": hallazgo},
        "contribuciones": [{"paso": 2, "agente": "ejecutor", "instruccion": "", "resumen": "", "herramientas": []}],
    }
    assert aplicar_guardas(decision("ejecutor", "otra vez"), estado).siguiente == "FINISH"  # type: ignore[arg-type]
