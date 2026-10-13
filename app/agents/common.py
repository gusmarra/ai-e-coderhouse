"""Piezas compartidas por los especialistas: como se ejecuta un turno y como se reporta al estado.

Cada especialista es un agente ReAct (`create_react_agent`) con SU propio historial, que vive
solo durante el turno. El nodo que lo envuelve cumple tres roles:

1. **Aislar el contexto**: arma una entrada minima (consigna + datos que necesita) en vez de
   pasarle todo el historial del grafo. Menos tokens, menos distraccion y ningun especialista
   ve las herramientas ni el razonamiento interno de otro.
2. **Ejecutar** el agente ReAct con un tope de pasos propio.
3. **Reportar** al estado compartido solo el resultado: un `Hallazgo` (por agente), una
   `Contribucion` (bitacora) y un mensaje corto en el hilo visible.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from app.state import Agente, Contribucion, EstadoOrquestador, Hallazgo

__all__ = ["NodoEspecialista", "crear_nodo_especialista"]

#: Tope de super-pasos del agente ReAct interno (cada llamada a herramienta gasta 2).
RECURSION_LIMIT_ESPECIALISTA = 10

_FUENTE = re.compile(r"fuente:\s*([\w\-]+\.md)")

NodoEspecialista = Callable[[EstadoOrquestador], Awaitable[dict[str, Any]]]


def _herramientas_usadas(mensajes: Sequence[BaseMessage]) -> list[str]:
    return [tc["name"] for m in mensajes if isinstance(m, AIMessage) for tc in m.tool_calls]


def _fuentes(mensajes: Sequence[BaseMessage]) -> list[str]:
    """Archivos del hotel que aparecen en las observaciones (ToolMessage) del turno."""
    vistas: list[str] = []
    for m in mensajes:
        if m.type == "tool":
            vistas.extend(f for f in _FUENTE.findall(str(m.content)) if f not in vistas)
    return vistas


def _resumir(texto: str, limite: int = 160) -> str:
    plano = " ".join(texto.split())
    return plano if len(plano) <= limite else plano[: limite - 1] + "…"


def crear_nodo_especialista(
    nombre: Agente,
    llm: BaseChatModel,
    herramientas: Sequence[BaseTool],
    prompt_sistema: str,
    armar_entrada: Callable[[EstadoOrquestador], str],
) -> NodoEspecialista:
    """Nodo del grafo que corre un agente ReAct aislado y vuelca su resultado al estado."""
    with warnings.catch_warnings():  # LangGraph 1.x lo mueve a langchain.agents; sigue funcionando en 1.x
        warnings.filterwarnings("ignore", message=".*create_react_agent.*")
        agente = create_react_agent(llm, list(herramientas), prompt=prompt_sistema, name=nombre)

    async def nodo(estado: EstadoOrquestador) -> dict[str, Any]:
        entrada = armar_entrada(estado)
        resultado = await agente.ainvoke(
            {"messages": [HumanMessage(entrada)]},
            {"recursion_limit": RECURSION_LIMIT_ESPECIALISTA},
        )
        mensajes = resultado["messages"]
        salida = mensajes[-1].text.strip() or "(el especialista no devolvio texto)"
        previo = estado.get("hallazgos", {}).get(nombre)
        paso = estado.get("pasos", 0) + 1
        hallazgo: Hallazgo = {
            "contenido": salida,
            "fuentes": _fuentes(mensajes),
            "version": (previo["version"] + 1) if previo else 1,
        }
        contribucion: Contribucion = {
            "paso": paso,
            "agente": nombre,
            "instruccion": estado.get("instruccion", ""),
            "resumen": _resumir(salida),
            "herramientas": _herramientas_usadas(mensajes),
        }
        return {
            "messages": [AIMessage(content=salida, name=nombre)],
            "hallazgos": {nombre: hallazgo},
            "contribuciones": [contribucion],
            "pasos": paso,
        }

    nodo.__name__ = f"nodo_{nombre}"
    return nodo
