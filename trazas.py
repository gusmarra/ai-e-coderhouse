"""Corre el grafo y reconstruye la traza de delegacion: quien actuo, con que consigna y que devolvio."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from config import RECURSION_LIMIT
from graph import estado_inicial

__all__ = ["Evento", "Traza", "ejecutar", "formatear_evento", "guardar"]


@dataclass
class Evento:
    nodo: str  # supervisor | investigador | analista | sintesis
    detalle: str  # decision, resumen del output o respuesta final
    consigna: str = ""  # lo que el supervisor le pidio al especialista
    herramientas: list[str] = field(default_factory=list)


@dataclass
class Traza:
    pregunta: str
    eventos: list[Evento] = field(default_factory=list)
    respuesta: str = ""
    pasos: int = 0
    contribuciones: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ruta(self) -> list[str]:
        return [e.nodo for e in self.eventos]


def _a_evento(nodo: str, update: dict[str, Any]) -> Evento:
    if nodo == "supervisor":
        destino = "FINISH" if update["next_agent"] == "FINISH" else update["next_agent"]
        consigna = f" · consigna: {update['instruccion']}" if update.get("instruccion") else ""
        return Evento(nodo, f"-> {destino} · {update['razon']}{consigna}")
    if nodo == "sintesis":
        return Evento(nodo, update["respuesta_final"])
    contrib = update["contribuciones"][0]
    return Evento(nodo, update["hallazgos"][nodo]["contenido"], contrib["instruccion"], contrib["herramientas"])


def formatear_evento(evento: Evento) -> str:
    cabecera = f"[{evento.nodo.upper()}]"
    if evento.herramientas:
        cabecera += f" herramientas: {', '.join(evento.herramientas)}"
    cuerpo = []
    if evento.consigna:
        cuerpo.append(f"  consigna: {evento.consigna}")
    cuerpo.append("  " + evento.detalle.replace("\n", "\n  "))
    return cabecera + "\n" + "\n".join(cuerpo)


async def ejecutar(
    grafo: CompiledStateGraph,
    pregunta: str,
    *,
    config: dict[str, Any] | None = None,
    on_evento: Callable[[Evento], None] | None = None,
) -> Traza:
    """Ejecuta una consulta con `recursion_limit` y devuelve la traza completa."""
    traza = Traza(pregunta=pregunta)
    cfg: RunnableConfig = {"recursion_limit": RECURSION_LIMIT, **(config or {})}  # type: ignore[typeddict-item]
    async for chunk in grafo.astream(estado_inicial(pregunta), cfg, stream_mode="updates"):
        for nodo, update in chunk.items():
            evento = _a_evento(nodo, update)
            traza.eventos.append(evento)
            if nodo in ("investigador", "analista"):
                traza.contribuciones.extend(update["contribuciones"])
                traza.pasos = update["pasos"]
            if nodo == "sintesis":
                traza.respuesta = update["respuesta_final"]
            if on_evento:
                on_evento(evento)
    return traza


def guardar(traza: Traza, destino_json: Path, destino_log: Path, *, modelo: str = "") -> None:
    """Escribe la traza como JSON (maquina) y como texto legible (humano)."""
    destino_json.parent.mkdir(parents=True, exist_ok=True)
    datos = asdict(traza) | {"ruta": traza.ruta, "modelo": modelo}
    destino_json.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    cabecera = f"Pregunta: {traza.pregunta}\nRuta: {' -> '.join(traza.ruta)}\n{'=' * 70}\n"
    destino_log.write_text(cabecera + "\n\n".join(formatear_evento(e) for e in traza.eventos) + "\n", encoding="utf-8")
