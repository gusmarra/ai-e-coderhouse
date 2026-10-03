"""Fases 2 y 3 - Estado, grafo ReAct y persistencia.

El ciclo ReAct (Reason + Act) es un grafo con un loop:

            +------------------------------+
            v                              |
    START -> modelo --(tools_condition)--> tools
                |
                +--(sin tool_calls)--> END

* `modelo` razona: el LLM lee la conversacion y decide si responde o pide herramientas.
* `tools_condition` mira el ultimo mensaje: si trae `tool_calls` va a `tools`, si no a END.
  Esa es toda la "autonomia": no hay ningun if/else del programador decidiendo la ruta.
* `tools` ejecuta las llamadas y agrega los resultados como `ToolMessage`; la arista
  `tools -> modelo` cierra el ciclo para que el LLM observe el resultado y re-razone.
  Si el resultado es un error, el modelo lo ve y puede reintentar o pedir aclaraciones.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, trim_messages
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config import MAX_MENSAJES_CONTEXTO
from herramientas import HERRAMIENTAS

__all__ = ["PROMPT_SISTEMA", "EstadoAgente", "construir_grafo", "recortar_contexto"]

PROMPT_SISTEMA = """\
Sos un asistente de soporte que responde consultas sobre clientes y sus pedidos.

Reglas:
- Los datos de clientes y pedidos SOLO se obtienen con las herramientas. Nunca inventes
  ids, cantidades ni montos: si no los tenes, consultalos.
- Si el usuario nombra a un cliente sin dar su id, buscalo primero por nombre.
- Si una herramienta devuelve un error, leelo: corregi el parametro y reintenta con otra
  estrategia (por ejemplo, buscar por nombre en vez de por id). Si no hay forma de
  resolverlo, deci claramente que paso y pedile al usuario el dato que falta.
- Si hay varias coincidencias posibles, no adivines: pedile al usuario que aclare cual es.
- Si la respuesta ya esta en la conversacion, no vuelvas a consultar.
- Responde en espanol rioplatense, breve, con los montos como $14.500.
"""


class EstadoAgente(MessagesState):
    """Estado del grafo: hereda `messages: Annotated[list[AnyMessage], add_messages]`.

    El reducer `add_messages` es lo que hace que cada nodo *agregue* mensajes en vez de
    pisar la lista (equivale a `operator.add`, pero ademas actualiza por id y entiende
    mensajes sueltos). Un nodo devuelve solo lo nuevo, `{"messages": [respuesta]}`, y
    LangGraph lo fusiona con el estado anterior.
    """


def recortar_contexto(
    mensajes: Sequence[BaseMessage], maximo: int = MAX_MENSAJES_CONTEXTO
) -> list[BaseMessage]:
    """Antidoto contra el "estado sucio": limita lo que ve el modelo, sin tocar lo guardado.

    Se queda con los ultimos `maximo` mensajes empezando siempre en un mensaje humano, asi
    nunca queda un ToolMessage huerfano (sin el AIMessage que lo pidio), que las APIs rechazan.
    """
    recortados = trim_messages(
        list(mensajes),
        max_tokens=maximo,
        token_counter=len,  # contamos mensajes, no tokens: simple y predecible
        strategy="last",
        start_on="human",
        allow_partial=False,
    )
    return recortados or list(mensajes)


def construir_grafo(
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver | None = None,
    herramientas: Sequence[BaseTool] = HERRAMIENTAS,
) -> CompiledStateGraph:
    """Arma y compila el grafo ReAct. Sin `checkpointer` el agente es efimero."""
    llm_con_herramientas = llm.bind_tools(list(herramientas))

    async def modelo(estado: EstadoAgente) -> dict[str, list[BaseMessage]]:
        contexto = recortar_contexto(estado["messages"])
        respuesta = await llm_con_herramientas.ainvoke([SystemMessage(PROMPT_SISTEMA), *contexto])
        return {"messages": [respuesta]}

    grafo = StateGraph(EstadoAgente)
    grafo.add_node("modelo", modelo)  # type: ignore[call-overload]  # mypy no infiere NodeInputT del TypedDict
    # handle_tool_errors: si una herramienta lanza una excepcion, el error vuelve al
    # modelo como ToolMessage (en vez de romper el grafo) y puede reintentar.
    grafo.add_node("tools", ToolNode(list(herramientas), handle_tool_errors=True))
    grafo.add_edge(START, "modelo")
    grafo.add_conditional_edges("modelo", tools_condition, {"tools": "tools", END: END})
    grafo.add_edge("tools", "modelo")
    return grafo.compile(checkpointer=checkpointer)
