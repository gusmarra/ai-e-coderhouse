"""El grafo del orquestador: topologia jerarquica Supervisor -> especialistas -> Supervisor.

    START -> supervisor --(enrutar)--> investigador --+
                 ^   |                                |
                 |   +-------------> analista --------+
                 |   |                                |
                 |   +--> aprobacion --(si)--> ejecutor
                 |   |        |                       |
                 +---+<-------+-----------------------+
                     |   (no) v
                     +-------------> sintesis -> END

`aprobacion` es el nodo Human-in-the-loop (app/hitl.py): pausa el grafo con `interrupt()` hasta que
una persona aprueba o rechaza la accion critica. Si rechaza, se salta el ejecutor y se cierra.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents import crear_analista, crear_ejecutor, crear_investigador, crear_sintesis, crear_supervisor
from app.base_hotel import BuscadorHotel
from app.hitl import nodo_aprobacion
from app.state import EstadoOrquestador

__all__ = ["checkpointer_redis", "construir_grafo", "enrutar", "estado_inicial"]


def enrutar(estado: EstadoOrquestador) -> Literal["investigador", "analista", "aprobacion", "sintesis"]:
    """Arista condicional: traduce la decision del supervisor al nombre del nodo siguiente.

    El `Literal` del retorno le dice a LangGraph (y al diagrama Mermaid) a donde puede ir.
    """
    destino = estado["next_agent"]
    if destino == "FINISH":
        return "sintesis"
    return "aprobacion" if destino == "ejecutor" else destino  # toda accion con efectos pasa por un humano


def estado_inicial(pregunta: str) -> dict[str, Any]:
    """Entrada del grafo: el pedido y los contadores en cero."""
    return {
        "messages": [HumanMessage(pregunta)],
        "pregunta": pregunta,
        "pasos": 0,
        "hallazgos": {},
        "contribuciones": [],
        "task_completed": False,
        "aprobado": None,
    }


def construir_grafo(
    llm: BaseChatModel,
    *,
    buscador: BuscadorHotel | None = None,
    llm_especialistas: BaseChatModel | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Arma y compila el grafo. `llm_especialistas` permite un modelo distinto (p. ej. mas barato)."""
    esp = llm_especialistas or llm
    grafo = StateGraph(EstadoOrquestador)
    grafo.add_node("supervisor", crear_supervisor(llm))  # type: ignore[call-overload]
    grafo.add_node("investigador", crear_investigador(esp, buscador))  # type: ignore[call-overload]
    grafo.add_node("analista", crear_analista(esp))  # type: ignore[call-overload]
    grafo.add_node("ejecutor", crear_ejecutor(esp))  # type: ignore[call-overload]
    grafo.add_node("aprobacion", nodo_aprobacion)  # type: ignore[call-overload]
    grafo.add_node("sintesis", crear_sintesis(llm))  # type: ignore[call-overload]

    grafo.add_edge(START, "supervisor")
    grafo.add_conditional_edges(
        "supervisor",
        enrutar,
        {"investigador": "investigador", "analista": "analista", "aprobacion": "aprobacion", "sintesis": "sintesis"},
    )
    grafo.add_edge("investigador", "supervisor")
    grafo.add_edge("analista", "supervisor")
    # `aprobacion` decide con Command(goto=...): "ejecutor" si aprueban, "sintesis" si rechazan.
    grafo.add_edge("ejecutor", "supervisor")
    grafo.add_edge("sintesis", END)
    return grafo.compile(checkpointer=checkpointer)


@asynccontextmanager
async def checkpointer_redis(url: str) -> AsyncIterator[BaseCheckpointSaver]:
    """Checkpointer de LangGraph sobre Redis (requiere Redis Stack: RediSearch + RedisJSON).

    Guarda el estado de cada super-paso por `thread_id` (= job_id): sobrevive a reinicios de la
    API y es lo que permite pausar en `interrupt()` y reanudar desde otro request.
    """
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver

    async with AsyncRedisSaver.from_conn_string(url) as saver:
        await saver.asetup()
        yield saver
