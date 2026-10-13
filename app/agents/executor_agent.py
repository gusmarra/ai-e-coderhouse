"""Agente Ejecutor: el unico con herramientas que CAMBIAN el mundo (efectos secundarios).

Por eso nunca corre solo: el grafo lo pone detras del nodo de aprobacion humana (`app/hitl.py`).
La herramienta simula el sistema de reservas con un registro en memoria; en produccion seria una
llamada al PMS del hotel (la parte critica de verdad: cancelar reservas y devolver plata).
"""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool

from app.agents.common import NodoEspecialista, crear_nodo_especialista
from app.state import EstadoOrquestador

__all__ = ["PROMPT_EJECUTOR", "REGISTRO_ACCIONES", "crear_ejecutor", "crear_herramientas_ejecutor"]

#: Bitacora de acciones realmente ejecutadas (simula la escritura en el sistema de reservas).
REGISTRO_ACCIONES: list[dict[str, Any]] = []
_lock = threading.Lock()

PROMPT_EJECUTOR = """\
Sos el EJECUTOR del Hotel Bahia Serena. Un humano YA aprobo la accion que te piden: ejecutala.

Reglas:
- Usa `cancelar_reserva` una sola vez, con el codigo de reserva y el monto de penalidad que figuran
  en la consigna y en los datos recibidos. No inventes codigos ni montos: si falta alguno, deci
  que falta en lugar de ejecutar.
- Devolve una linea con lo que hiciste (codigo, penalidad, numero de operacion).
"""


def crear_herramientas_ejecutor() -> list[BaseTool]:
    @tool(parse_docstring=True)
    def cancelar_reserva(codigo_reserva: str, penalidad_usd: float, motivo: str) -> dict[str, Any]:
        """Cancela una reserva en el sistema del hotel y cobra la penalidad indicada (EFECTO REAL).

        Accion irreversible: solo se llama despues de la aprobacion humana. Devuelve el numero de
        operacion registrado.

        Args:
            codigo_reserva: Codigo de la reserva a cancelar, por ejemplo "BS-10482".
            penalidad_usd: Penalidad a cobrar en USD (0 si la cancelacion es sin cargo).
            motivo: Motivo de la cancelacion, en una frase.
        """
        with _lock:
            operacion = f"OP-{len(REGISTRO_ACCIONES) + 1:05d}"
            REGISTRO_ACCIONES.append(
                {"operacion": operacion, "reserva": codigo_reserva, "penalidad_usd": penalidad_usd, "motivo": motivo}
            )
        return {
            "operacion": operacion,
            "reserva": codigo_reserva,
            "estado": "CANCELADA",
            "penalidad_usd": penalidad_usd,
        }

    return [cancelar_reserva]


def _entrada(estado: EstadoOrquestador) -> str:
    partes = [
        f"Pedido original del usuario: {estado['pregunta']}",
        f"Accion aprobada por un humano, a ejecutar: {estado['instruccion']}",
    ]
    for agente, h in estado.get("hallazgos", {}).items():
        if agente != "ejecutor":
            partes.append(f"Datos de {agente}:\n{h['contenido']}")
    return "\n\n".join(partes)


def crear_ejecutor(llm: BaseChatModel) -> NodoEspecialista:
    return crear_nodo_especialista("ejecutor", llm, crear_herramientas_ejecutor(), PROMPT_EJECUTOR, _entrada)
