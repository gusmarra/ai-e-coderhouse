"""Ejecucion de un turno del agente y registro de la traza ReAct.

`ejecutar_turno` corre el grafo con `stream_mode="updates"`: cada vez que un nodo
termina, emite lo que escribio al estado. Eso permite reconstruir, en orden, el ciclo
Pensamiento -> Accion -> Observacion -> Respuesta, e imprimirlo en vivo.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from config import RECURSION_LIMIT

__all__ = ["Paso", "Turno", "ejecutar_turno", "formatear_paso", "pasos_desde_mensajes"]

TipoPaso = Literal["pensamiento", "accion", "observacion", "respuesta", "limite"]


@dataclass
class Paso:
    tipo: TipoPaso
    contenido: str
    herramienta: str | None = None
    argumentos: dict[str, Any] | None = None
    error: bool = False


@dataclass
class Turno:
    usuario: str
    thread_id: str
    pasos: list[Paso] = field(default_factory=list)
    respuesta: str = ""

    @property
    def llamadas_herramienta(self) -> int:
        return sum(1 for p in self.pasos if p.tipo == "accion")

    def a_dict(self) -> dict[str, Any]:
        return {
            "usuario": self.usuario,
            "thread_id": self.thread_id,
            "llamadas_herramienta": self.llamadas_herramienta,
            "pasos": [{k: v for k, v in asdict(p).items() if v not in (None, False)} for p in self.pasos],
            "respuesta": self.respuesta,
        }


def _texto(mensaje: BaseMessage) -> str:
    """Texto plano de un mensaje (Gemini a veces devuelve una lista de bloques)."""
    return mensaje.text.strip()


def _es_error(contenido: str) -> bool:
    try:
        datos = json.loads(contenido)
    except ValueError:
        return contenido.lower().startswith("error")
    return isinstance(datos, dict) and "error" in datos


def pasos_desde_mensajes(mensajes: list[BaseMessage]) -> list[Paso]:
    """Traduce los mensajes nuevos que escribio un nodo a pasos de la traza ReAct."""
    pasos: list[Paso] = []
    for m in mensajes:
        if isinstance(m, AIMessage):
            if m.tool_calls:
                if texto := _texto(m):
                    pasos.append(Paso("pensamiento", texto))
                pasos.extend(
                    Paso("accion", f"{c['name']}({c['args']})", c["name"], dict(c["args"]))
                    for c in m.tool_calls
                )
            else:
                pasos.append(Paso("respuesta", _texto(m)))
        elif isinstance(m, ToolMessage):
            contenido = _texto(m)
            error = _es_error(contenido) or m.status == "error"
            pasos.append(Paso("observacion", contenido, m.name, error=error))
    return pasos


def formatear_paso(paso: Paso) -> str:
    etiqueta = {
        "pensamiento": "PENSAMIENTO ",
        "accion": "ACCION      ",
        "observacion": "OBSERVACION ",
        "respuesta": "RESPUESTA   ",
        "limite": "LIMITE      ",
    }[paso.tipo]
    marca = " [ERROR]" if paso.error else ""
    return f"  {etiqueta}{paso.contenido}{marca}"


async def ejecutar_turno(
    grafo: CompiledStateGraph,
    pregunta: str,
    thread_id: str,
    *,
    on_paso: Callable[[Paso], None] | None = None,
) -> Turno:
    """Envia una pregunta al agente en el `thread_id` dado y devuelve el turno con su traza.

    El `thread_id` es la clave bajo la que el checkpointer guarda el estado: mismo id,
    misma conversacion. `recursion_limit` corta el ciclo si el modelo no converge.
    """
    turno = Turno(usuario=pregunta, thread_id=thread_id)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}

    def registrar(paso: Paso) -> None:
        turno.pasos.append(paso)
        if on_paso:
            on_paso(paso)

    try:
        async for update in grafo.astream(
            {"messages": [HumanMessage(pregunta)]}, config, stream_mode="updates"
        ):
            for salida in update.values():
                for paso in pasos_desde_mensajes(salida["messages"]):
                    registrar(paso)
                    if paso.tipo == "respuesta":
                        turno.respuesta = paso.contenido
    except GraphRecursionError:
        turno.respuesta = (
            f"No pude resolverlo en {RECURSION_LIMIT} pasos. Probá reformular la pregunta con más datos."
        )
        registrar(Paso("limite", f"recursion_limit={RECURSION_LIMIT} alcanzado; se corta el ciclo"))
    return turno
