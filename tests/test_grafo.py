from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from grafo import construir_grafo, recortar_contexto
from tests.fakes import ModeloGuionado, llamada
from trazas import Turno, ejecutar_turno


def _nombres_de_acciones(turno: Turno) -> list[str]:
    return [p.herramienta or "" for p in turno.pasos if p.tipo == "accion"]


async def test_razonamiento_multipaso_encadena_dos_herramientas() -> None:
    llm = ModeloGuionado(
        guion=[
            llamada("buscar_cliente", "c1", nombre="Laura Gómez"),
            llamada("buscar_pedidos", "c2", cliente_id=102),
            AIMessage("Laura Gómez tuvo 3 pedidos por un total de $14.500."),
        ]
    )
    grafo = construir_grafo(llm, InMemorySaver())

    turno = await ejecutar_turno(grafo, "¿Cuántos pedidos tuvo Laura Gómez?", "t1")

    assert _nombres_de_acciones(turno) == ["buscar_cliente", "buscar_pedidos"]
    assert turno.llamadas_herramienta == 2
    assert [p.tipo for p in turno.pasos] == ["accion", "observacion", "accion", "observacion", "respuesta"]
    assert "14.500" in turno.respuesta


async def test_sin_tool_calls_el_grafo_termina_sin_pasar_por_herramientas() -> None:
    grafo = construir_grafo(ModeloGuionado(guion=[AIMessage("Hola!")]), InMemorySaver())
    turno = await ejecutar_turno(grafo, "hola", "t1")
    assert [p.tipo for p in turno.pasos] == ["respuesta"]


async def test_ciclo_de_retorno_el_error_vuelve_al_modelo_que_reintenta() -> None:
    llm = ModeloGuionado(
        guion=[
            llamada("buscar_pedidos", "c1", cliente_id=999),  # id inexistente -> error
            llamada("buscar_cliente", "c2", nombre="Laura Gómez"),  # 2do intento, otra estrategia
            llamada("buscar_pedidos", "c3", cliente_id=102),
            AIMessage("Ese id no existe; con el nombre encontré 3 pedidos."),
        ]
    )
    grafo = construir_grafo(llm, InMemorySaver())

    turno = await ejecutar_turno(grafo, "pedidos del cliente 999", "t1")

    observaciones = [p for p in turno.pasos if p.tipo == "observacion"]
    assert observaciones[0].error and not observaciones[1].error
    # El modelo vio el ToolMessage con el error antes de decidir su 2do intento.
    vistos_en_2da_llamada = llm.recibidos[1]
    assert isinstance(vistos_en_2da_llamada[-1], ToolMessage)
    assert "No existe un cliente" in vistos_en_2da_llamada[-1].text


async def test_una_excepcion_en_la_herramienta_no_rompe_el_grafo() -> None:
    # cliente_id no numerico: la validacion del schema falla y ToolNode lo devuelve como error.
    guion = [llamada("buscar_pedidos", "c1", cliente_id="abc"), AIMessage("Necesito un id.")]
    llm = ModeloGuionado(guion=guion)
    turno = await ejecutar_turno(construir_grafo(llm, InMemorySaver()), "pedidos", "t1")
    assert turno.pasos[1].tipo == "observacion" and turno.pasos[1].error
    assert turno.respuesta == "Necesito un id."


async def test_recursion_limit_corta_un_loop_infinito() -> None:
    llm = ModeloGuionado(guion=[llamada("buscar_pedidos", "c1", cliente_id=102)])  # repite siempre
    turno = await ejecutar_turno(construir_grafo(llm, InMemorySaver()), "loop", "t1")
    assert turno.pasos[-1].tipo == "limite"
    assert "No pude resolverlo" in turno.respuesta
    assert turno.llamadas_herramienta <= 5


async def test_mismo_thread_id_recuerda_y_otro_thread_no() -> None:
    llm = ModeloGuionado(guion=[AIMessage("uno"), AIMessage("dos"), AIMessage("tres")])
    grafo = construir_grafo(llm, InMemorySaver())

    await ejecutar_turno(grafo, "Me llamo Ana", "hilo-a")
    await ejecutar_turno(grafo, "¿Cómo me llamo?", "hilo-a")
    await ejecutar_turno(grafo, "¿Cómo me llamo?", "hilo-b")

    def humanos(i: int) -> list[str]:
        return [m.text for m in llm.recibidos[i] if isinstance(m, HumanMessage)]

    assert humanos(1) == ["Me llamo Ana", "¿Cómo me llamo?"]
    assert humanos(2) == ["¿Cómo me llamo?"]


async def test_la_memoria_sobrevive_a_reabrir_el_archivo_sqlite(tmp_path: Path) -> None:
    db = str(tmp_path / "cp.sqlite")
    llm = ModeloGuionado(guion=[AIMessage("Hola Ana"), AIMessage("Te llamás Ana")])

    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        await ejecutar_turno(construir_grafo(llm, saver), "Me llamo Ana", "hilo")
    async with AsyncSqliteSaver.from_conn_string(db) as saver:  # "proceso nuevo"
        await ejecutar_turno(construir_grafo(llm, saver), "¿Cómo me llamo?", "hilo")

    previos = [m.text for m in llm.recibidos[1] if isinstance(m, HumanMessage | AIMessage)]
    assert previos == ["Me llamo Ana", "Hola Ana", "¿Cómo me llamo?"]


def test_recortar_contexto_no_deja_tool_messages_huerfanos() -> None:
    mensajes = [
        HumanMessage("viejo"),
        llamada("buscar_pedidos", "x", cliente_id=1),
        ToolMessage("{}", tool_call_id="x"),
        AIMessage("ok"),
        HumanMessage("nuevo"),
        AIMessage("resp"),
    ]
    recortados = recortar_contexto(mensajes, maximo=4)
    assert isinstance(recortados[0], HumanMessage) and recortados[0].text == "nuevo"
