"""Supervisor: el router inteligente y controlador de flujo del grafo.

Hace dos cosas muy distintas, y por eso estan separadas:

1. **Decide con un LLM** (salida estructurada `Decision`): quien interviene ahora o si ya se puede
   cerrar. Su prompt incluye una RUBRICA de suficiencia: es la validacion de los outputs de los
   especialistas antes de dar el END.
2. **Aplica guardas deterministicas** (`aplicar_guardas`) que el LLM no puede saltearse: tope de
   pasos, tope de refinamientos por especialista, no analizar sin datos, no cerrar sin haber
   investigado. Son el freno contra el "Supervisor Infinito": un modelo caprichoso puede pedir
   "una correccion mas" para siempre; el codigo no.

El nodo `sintesis` redacta la respuesta final a partir de los hallazgos.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import MAX_PASOS, MAX_REFINAMIENTOS
from app.state import AGENTES, FINISH, EstadoOrquestador, Siguiente

__all__ = [
    "PROMPT_SINTESIS",
    "PROMPT_SUPERVISOR",
    "Decision",
    "aplicar_guardas",
    "crear_sintesis",
    "crear_supervisor",
]

log = logging.getLogger("orquestador.supervisor")


class Decision(BaseModel):
    """Salida estructurada del supervisor: se mapea 1 a 1 con los nodos del grafo."""

    siguiente: Siguiente = Field(
        description="'investigador', 'analista', 'ejecutor' o 'FINISH' si la rubrica se cumple y se puede cerrar."
    )
    instruccion: str = Field(
        default="",
        description=(
            "Consigna concreta y autocontenida para el especialista elegido (vacia si FINISH). "
            "Si es un refinamiento, indica exactamente que falta o que esta mal."
        ),
    )
    razon: str = Field(description="Una oracion: que punto de la rubrica motiva esta decision.")


PROMPT_SUPERVISOR = """\
Sos el SUPERVISOR de un equipo de tres especialistas que responde consultas sobre el Hotel Bahia Serena.
Dada la conversacion actual, decidis QUIEN debe intervenir ahora o si es momento de FINALIZAR.

Equipo:
- investigador: busca en los documentos del hotel (politicas, reglamento, servicios, seguridad).
  Es el unico que accede a los datos. No calcula ni valora.
- analista: procesa datos que ya tiene el investigador: calculos exactos, extraccion de cifras,
  tono/sentimiento. No accede a los documentos.
- ejecutor: realiza acciones con EFECTOS REALES (hoy: cancelar una reserva y cobrar la penalidad).
  Solo se lo llama si el usuario PIDE EXPLICITAMENTE ejecutar la accion y ya hay un codigo de
  reserva. Antes de que actue, un humano tiene que aprobarla: el sistema lo gestiona solo.

Flujo normal: investigador -> analista (solo si el pedido requiere calculo o valoracion) -> FINISH.
- Pedido solo de consulta (sin calculo ni valoracion): investigador -> FINISH.
- Pedido de ejecutar una cancelacion: investigador (politica) -> analista (penalidad) -> ejecutor -> FINISH.
  La consigna al ejecutor incluye codigo de reserva y penalidad exacta. Nunca ejecutes por tu cuenta
  algo que el usuario solo consulto ("¿cuanto pagaria si cancelo?" NO es un pedido de cancelar).
- Nunca llames al analista ni al ejecutor si todavia no hay datos del investigador.
- Cada consigna debe ser concreta y autocontenida: el especialista NO ve la charla completa,
  solo tu consigna y los datos que necesita.

RUBRICA DE SUFICIENCIA (para elegir FINISH tienen que cumplirse TODOS los puntos):
1. FUENTES: cada dato del hotel usado viene del investigador con su archivo de origen. Si el
   investigador dice que no encontro algo, eso se informa tal cual: no se rellena ni se reintenta
   en vano.
2. COMPLETITUD: cada parte del pedido original tiene respuesta (nada quedo sin cubrir).
3. COMPUTO: si el pedido incluye cuentas, totales, porcentajes o valoraciones, el analista
   corrio y muestra la formula con numeros y el resultado.
4. ACCION: si el pedido incluia ejecutar algo, el ejecutor corro (o un humano la rechazo).
5. CONSISTENCIA: las cifras del analista coinciden con las del investigador (mismos montos,
   porcentajes y plazos); no hay supuestos que el investigador no respalde.

Si falla un punto, delega al responsable con una consigna de REFINAMIENTO que diga exactamente
que corregir. No pidas refinamientos por detalles menores: cada vuelta cuesta, y hay un maximo
de turnos. Si la rubrica se cumple, elegi FINISH.
"""


def _formatear_estado(estado: EstadoOrquestador) -> str:
    """Lo que ve el supervisor: pedido, presupuesto restante y hallazgos. Sin trazas de herramientas."""
    hallazgos = estado.get("hallazgos", {})
    lineas = [
        f"PEDIDO ORIGINAL: {estado['pregunta']}",
        f"TURNOS USADOS: {estado.get('pasos', 0)} de {MAX_PASOS}.",
    ]
    for agente in AGENTES:
        h = hallazgos.get(agente)
        if h is None:
            lineas.append(f"\n[{agente}] todavia no intervino.")
            continue
        fuentes = ", ".join(h["fuentes"]) or "sin fuentes declaradas"
        lineas.append(f"\n[{agente}] version {h['version']} · fuentes: {fuentes}\n{h['contenido']}")
    return "\n".join(lineas)


def _veces(estado: EstadoOrquestador, agente: str) -> int:
    return sum(1 for c in estado.get("contribuciones", []) if c["agente"] == agente)


def aplicar_guardas(decision: Decision, estado: EstadoOrquestador) -> Decision:
    """Corrige las decisiones que violan los limites duros. Devuelve la decision final.

    El orden importa: primero lo que fuerza el cierre (tope de pasos), despues lo que redirige.
    """
    hallazgos = estado.get("hallazgos", {})
    pasos = estado.get("pasos", 0)
    siguiente = decision.siguiente

    if siguiente != FINISH and pasos >= MAX_PASOS:
        return Decision(siguiente=FINISH, razon=f"[guarda] tope de {MAX_PASOS} turnos: cierro con lo que hay.")
    if siguiente in ("analista", "ejecutor") and "investigador" not in hallazgos:
        return Decision(
            siguiente="investigador",
            instruccion=estado["pregunta"],
            razon="[guarda] el analista y el ejecutor no pueden trabajar sin datos: primero investigar.",
        )
    if siguiente == FINISH and "investigador" not in hallazgos:
        return Decision(
            siguiente="investigador",
            instruccion=estado["pregunta"],
            razon="[guarda] no se cierra sin haber consultado los documentos del hotel.",
        )
    if siguiente == "ejecutor" and _veces(estado, "ejecutor") >= 1:  # un efecto secundario nunca se repite
        return Decision(siguiente=FINISH, razon="[guarda] la accion critica ya se ejecuto: no se repite.")
    if siguiente in AGENTES and _veces(estado, siguiente) >= 1 + MAX_REFINAMIENTOS:
        return Decision(
            siguiente=FINISH,
            razon=f"[guarda] el {siguiente} ya agoto sus {MAX_REFINAMIENTOS} refinamiento(s): cierro.",
        )
    if siguiente in AGENTES and not decision.instruccion.strip():
        return decision.model_copy(update={"instruccion": estado["pregunta"]})
    return decision


NodoSupervisor = Callable[[EstadoOrquestador], Awaitable[dict[str, Any]]]


def crear_supervisor(llm: BaseChatModel) -> NodoSupervisor:
    decisor = llm.with_structured_output(Decision)

    async def supervisor(estado: EstadoOrquestador) -> dict[str, Any]:
        propuesta = await decisor.ainvoke([SystemMessage(PROMPT_SUPERVISOR), HumanMessage(_formatear_estado(estado))])
        if not isinstance(propuesta, Decision):  # algunos proveedores devuelven dict
            propuesta = Decision.model_validate(propuesta)
        final = aplicar_guardas(propuesta, estado)
        if final is not propuesta:
            log.info("guarda: %s -> %s (%s)", propuesta.siguiente, final.siguiente, final.razon)
        return {
            "next_agent": final.siguiente,
            "instruccion": final.instruccion,
            "razon": final.razon,
            "task_completed": final.siguiente == FINISH,
        }

    return supervisor


# ----------------------------------------------------------------------
# Sintesis final
# ----------------------------------------------------------------------
PROMPT_SINTESIS = """\
Redactas la RESPUESTA FINAL para el usuario del Hotel Bahia Serena, a partir de los hallazgos del
equipo. Reglas:
- Usa solo lo que dicen los hallazgos; no agregues datos nuevos ni cambies cifras.
- Responde primero lo que se pidio (el resultado o la conclusion) y despues el detalle breve.
- Si hay calculo, muestra la cuenta con sus numeros.
- Cita las fuentes (archivos) entre parentesis.
- Si algo del pedido quedo sin respaldo en los hallazgos, decilo claramente en vez de completarlo.
- Espanol rioplatense, claro y breve. Sin mencionar a los agentes ni al proceso interno.
"""


def crear_sintesis(llm: BaseChatModel) -> NodoSupervisor:
    async def sintesis(estado: EstadoOrquestador) -> dict[str, Any]:
        respuesta = await llm.ainvoke([SystemMessage(PROMPT_SINTESIS), HumanMessage(_formatear_estado(estado))])
        texto = respuesta.text.strip()
        return {"respuesta_final": texto, "messages": [AIMessage(content=texto, name="supervisor")]}

    return sintesis
