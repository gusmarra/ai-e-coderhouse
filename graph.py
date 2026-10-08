"""El grafo del orquestador: topologia jerarquica Supervisor -> especialistas -> Supervisor.

    START -> supervisor --(enrutar)--> investigador --+
                 ^   |                                |
                 |   +-------------> analista --------+
                 |   |                                |
                 +---+<-------------------------------+
                     +-------------> sintesis -> END

Los especialistas NUNCA se hablan entre si: todo pasa por el supervisor, que es el unico que
ve el panorama completo. Cada especialista devuelve el control al supervisor al terminar, y el
supervisor decide si hace falta otra vuelta (refinamiento) o si cierra con `sintesis`.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents import crear_analista, crear_investigador, crear_sintesis, crear_supervisor
from base_hotel import BuscadorHotel
from state import EstadoOrquestador

__all__ = ["construir_grafo", "enrutar", "estado_inicial"]


def enrutar(estado: EstadoOrquestador) -> Literal["investigador", "analista", "sintesis"]:
    """Arista condicional: traduce la decision del supervisor al nombre del nodo siguiente.

    El `Literal` del retorno le dice a LangGraph (y al diagrama Mermaid) a donde puede ir.
    """
    destino = estado["next_agent"]
    return "sintesis" if destino == "FINISH" else destino


def estado_inicial(pregunta: str) -> dict[str, Any]:
    """Entrada del grafo: el pedido y los contadores en cero."""
    return {
        "messages": [HumanMessage(pregunta)],
        "pregunta": pregunta,
        "pasos": 0,
        "hallazgos": {},
        "contribuciones": [],
        "task_completed": False,
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
    grafo.add_node("sintesis", crear_sintesis(llm))  # type: ignore[call-overload]

    grafo.add_edge(START, "supervisor")
    grafo.add_conditional_edges(
        "supervisor",
        enrutar,
        {"investigador": "investigador", "analista": "analista", "sintesis": "sintesis"},
    )
    grafo.add_edge("investigador", "supervisor")
    grafo.add_edge("analista", "supervisor")
    grafo.add_edge("sintesis", END)
    return grafo.compile(checkpointer=checkpointer)
