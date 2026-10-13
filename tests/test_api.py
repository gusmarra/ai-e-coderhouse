"""API + worker + HITL, offline: LLM guionado, Redis falso (fakeredis) y checkpointer en memoria."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fakeredis import FakeAsyncRedis
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.executor_agent import REGISTRO_ACCIONES
from app.base_hotel import BuscadorHotel, cargar_chunks
from app.graph import construir_grafo
from app.main import crear_app
from tests.fakes import ModeloGuionado, decision, llamada

PEDIDO = "Cancela la reserva BS-10482 (4 noches a USD 150, temporada alta, 3 dias antes) y cobra la penalidad."


def _grafo(*, decisiones: list[Any] | None = None) -> Any:
    sup = ModeloGuionado(
        guion=[AIMessage("Reserva BS-10482 cancelada con penalidad de USD 300.")],
        decisiones=decisiones
        if decisiones is not None
        else [
            decision("investigador", "buscar la penalidad"),
            decision("analista", "calcular"),
            decision("ejecutor", "Cancelar BS-10482 con penalidad USD 300"),
            decision("FINISH"),
        ],
    )
    esp = ModeloGuionado(
        guion=[
            llamada("buscar_en_documentos_hotel", "i1", consulta="temporada alta cancelacion penalidad"),
            AIMessage("Penalidad: 50% de la estadia (fuente: politica-reservas-cancelaciones.md)."),
            llamada("calcular", "a1", expresion="0.5 * 4 * 150"),
            AIMessage("Penalidad = 0.5 * 4 * 150 = USD 300."),
            llamada(
                "cancelar_reserva", "e1", codigo_reserva="BS-10482", penalidad_usd=300, motivo="pedido del huesped"
            ),
            AIMessage("Cancelada BS-10482, penalidad USD 300, operacion registrada."),
        ]
    )
    return construir_grafo(
        sup, llm_especialistas=esp, buscador=BuscadorHotel(cargar_chunks()), checkpointer=InMemorySaver()
    )


async def _esperar(cliente: httpx.AsyncClient, job_id: str, estados: set[str]) -> dict[str, Any]:
    for _ in range(200):
        job = (await cliente.get(f"/tasks/{job_id}")).json()
        if job["status"] in estados:
            return job  # type: ignore[no-any-return]
        await asyncio.sleep(0.02)
    raise AssertionError(f"el job no llego a {estados}: {job}")


@pytest.fixture
async def cliente() -> AsyncIterator[tuple[httpx.AsyncClient, Any]]:
    REGISTRO_ACCIONES.clear()
    api = crear_app(grafo=_grafo(), redis=FakeAsyncRedis(decode_responses=True))
    async with api.router.lifespan_context(api):
        transporte = httpx.ASGITransport(app=api)
        async with httpx.AsyncClient(transport=transporte, base_url="http://test") as c:
            yield c, api


async def test_post_encola_y_devuelve_job_id_sin_esperar_al_agente(cliente: Any) -> None:
    http, _ = cliente
    resp = await http.post("/tasks", json={"pregunta": PEDIDO})
    assert resp.status_code == 202
    cuerpo = resp.json()
    assert cuerpo["status"] == "PENDING"
    assert (await http.get(f"/tasks/{cuerpo['job_id']}")).json()["pregunta"] == PEDIDO


async def test_accion_critica_pausa_hasta_que_un_humano_aprueba(cliente: Any) -> None:
    http, _ = cliente
    job_id = (await http.post("/tasks", json={"pregunta": PEDIDO})).json()["job_id"]

    pausado = await _esperar(http, job_id, {"AWAITING_APPROVAL", "FAILED", "DONE"})
    assert pausado["status"] == "AWAITING_APPROVAL", pausado
    assert pausado["pendiente"]["tipo"] == "aprobacion_requerida"
    assert "BS-10482" in pausado["pendiente"]["accion"]
    assert REGISTRO_ACCIONES == []  # nada se ejecuto todavia

    assert (await http.post(f"/tasks/{job_id}/approve", json={"aprobado": True})).status_code == 202
    final = await _esperar(http, job_id, {"DONE", "FAILED"})

    assert final["status"] == "DONE", final
    assert final["aprobado"] == "True"
    assert "USD 300" in final["respuesta"]
    assert [a["reserva"] for a in REGISTRO_ACCIONES] == ["BS-10482"]  # se ejecuto exactamente una vez


async def test_rechazo_humano_cierra_sin_ejecutar_nada(cliente: Any) -> None:
    http, _ = cliente
    job_id = (await http.post("/tasks", json={"pregunta": PEDIDO})).json()["job_id"]
    await _esperar(http, job_id, {"AWAITING_APPROVAL"})

    await http.post(f"/tasks/{job_id}/approve", json={"aprobado": False, "comentario": "falta autorizacion"})
    final = await _esperar(http, job_id, {"DONE", "FAILED"})

    assert final["status"] == "DONE"
    assert final["aprobado"] == "False"
    assert REGISTRO_ACCIONES == []


async def test_no_se_puede_aprobar_dos_veces_ni_un_job_que_no_espera(cliente: Any) -> None:
    http, _ = cliente
    job_id = (await http.post("/tasks", json={"pregunta": PEDIDO})).json()["job_id"]
    await _esperar(http, job_id, {"AWAITING_APPROVAL"})

    primera = await http.post(f"/tasks/{job_id}/approve", json={"aprobado": True})
    segunda = await http.post(f"/tasks/{job_id}/approve", json={"aprobado": True})
    assert (primera.status_code, segunda.status_code) == (202, 409)
    await _esperar(http, job_id, {"DONE", "FAILED"})
    assert len(REGISTRO_ACCIONES) == 1
    assert (await http.post("/tasks/no-existe/approve", json={"aprobado": True})).status_code == 404


async def test_si_el_agente_falla_el_job_queda_en_failed_con_el_error() -> None:
    api = crear_app(grafo=_grafo(decisiones=[]), redis=FakeAsyncRedis(decode_responses=True))  # supervisor roto
    async with api.router.lifespan_context(api):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api), base_url="http://test") as http:
            job_id = (await http.post("/tasks", json={"pregunta": "consulta cualquiera"})).json()["job_id"]
            final = await _esperar(http, job_id, {"FAILED", "DONE"})
    assert final["status"] == "FAILED"
    assert final["error"].startswith("IndexError")


async def test_consulta_sin_accion_critica_no_pide_aprobacion() -> None:
    api = crear_app(
        grafo=_grafo(decisiones=[decision("investigador", "buscar"), decision("FINISH")]),
        redis=FakeAsyncRedis(decode_responses=True),
    )
    async with api.router.lifespan_context(api):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api), base_url="http://test") as http:
            job_id = (await http.post("/tasks", json={"pregunta": "¿Cual es la politica de mascotas?"})).json()[
                "job_id"
            ]
            final = await _esperar(http, job_id, {"FAILED", "DONE", "AWAITING_APPROVAL"})
    assert final["status"] == "DONE"
    assert final["aprobado"] == ""


async def test_get_de_job_inexistente_es_404_y_pregunta_vacia_es_422(cliente: Any) -> None:
    http, _ = cliente
    assert (await http.get("/tasks/xyz")).status_code == 404
    assert (await http.post("/tasks", json={"pregunta": ""})).status_code == 422
