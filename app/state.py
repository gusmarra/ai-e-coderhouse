"""Estado compartido del orquestador: la unica "memoria" que circula entre los nodos.

Hereda de `MessagesState` (campo `messages` con el reducer `add_messages`) y le suma campos
para que el supervisor decida con datos y no solo leyendo la charla:

* quien manda ahora (`next_agent`) y con que consigna (`instruccion`);
* que produjo cada especialista (`hallazgos`) y quien aporto que y cuando (`contribuciones`);
* cuando parar (`pasos`, `task_completed`).

Regla de oro del multi-agente: los campos que escriben varios nodos llevan un reducer explicito
que dice como se fusionan las escrituras. Sin reducer, la ultima escritura pisa a las anteriores
y se pierde contexto.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import MessagesState

__all__ = [
    "AGENTES",
    "FINISH",
    "Agente",
    "Contribucion",
    "EstadoOrquestador",
    "Hallazgo",
    "Siguiente",
    "fusionar_hallazgos",
]

#: Nombres de los especialistas = nombres de los nodos del grafo.
Agente = Literal["investigador", "analista", "ejecutor"]
AGENTES: tuple[Agente, ...] = ("investigador", "analista", "ejecutor")

#: Lo que puede decidir el supervisor: un especialista o cerrar (ir a sintesis).
FINISH: Literal["FINISH"] = "FINISH"
Siguiente = Literal["investigador", "analista", "ejecutor", "FINISH"]


class Hallazgo(TypedDict):
    """Ultimo output de un especialista: lo que se le pasa a los demas (no toda su charla)."""

    contenido: str
    fuentes: list[str]  # archivos del hotel consultados, si aplica
    version: int  # cuantas veces corrio: 1 = primera pasada, 2 = refinado


class Contribucion(TypedDict):
    """Registro de auditoria: una linea por turno de especialista."""

    paso: int
    agente: Agente
    instruccion: str  # lo que el supervisor le pidio
    resumen: str  # inicio de lo que devolvio
    herramientas: list[str]  # que herramientas llego a usar


def fusionar_hallazgos(actual: dict[str, Hallazgo], nuevo: dict[str, Hallazgo]) -> dict[str, Hallazgo]:
    """Reducer de `hallazgos`: fusiona por agente en vez de pisar el dict entero.

    Si el analista escribe {"analista": ...}, el hallazgo del investigador sigue ahi.
    """
    return {**actual, **nuevo}


class EstadoOrquestador(MessagesState):
    """Estado del grafo. `messages` (heredado) es el hilo visible: pedido, turnos y respuesta."""

    #: Pedido original del usuario, intacto. Los especialistas lo reciben como contexto,
    #: no el historial completo.
    pregunta: str
    #: A quien delega el supervisor en este turno (o FINISH).
    next_agent: Siguiente
    #: Consigna concreta para ese especialista: lo unico de la charla que recibe, junto con
    #: los hallazgos que necesita (evita la contaminacion de contexto).
    instruccion: str
    #: Por que el supervisor tomo esa decision: queda en la traza.
    razon: str
    #: Ultimo output de cada especialista, fusionado por agente.
    hallazgos: Annotated[dict[str, Hallazgo], fusionar_hallazgos]
    #: Bitacora de quien aporto que. `operator.add` acumula: nunca se pierde un turno.
    contribuciones: Annotated[list[Contribucion], operator.add]
    #: Turnos de especialistas consumidos. Alimenta el freno anti-"Supervisor Infinito".
    pasos: int
    #: True cuando el supervisor valido el resultado contra su rubrica y puede cerrar.
    task_completed: bool
    #: Resultado de la aprobacion humana de la accion critica: True/False (None si no hubo pedido).
    aprobado: bool | None
    #: Respuesta final redactada por el nodo de sintesis.
    respuesta_final: str
