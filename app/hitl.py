"""Human-in-the-loop: pausa obligatoria antes de cualquier accion con efectos secundarios.

Dos piezas:

* `nodo_aprobacion`: nodo del grafo. Llama a `interrupt()`, que congela la ejecucion y guarda el
  estado en el checkpointer (Redis). El grafo no avanza hasta que alguien reanuda ese mismo
  `thread_id` con `Command(resume=...)`.
* `router`: `POST /tasks/{id}/approve`, el input externo que reanuda (o rechaza) la tarea.

Como `interrupt()` reejecuta el nodo desde el principio al reanudar, el nodo no hace nada con
efectos antes de pausar: solo arma el pedido y decide el destino.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from app.jobs import EstadoJob
from app.state import EstadoOrquestador, Hallazgo

__all__ = ["Aprobacion", "nodo_aprobacion", "router"]


class Aprobacion(BaseModel):
    """Cuerpo de `POST /tasks/{id}/approve`."""

    aprobado: bool = Field(description="True ejecuta la accion critica; False la cancela y cierra la tarea.")
    comentario: str = Field(default="", description="Motivo o nota del revisor (queda en la traza).")


def nodo_aprobacion(estado: EstadoOrquestador) -> Command[Literal["ejecutor", "sintesis"]]:
    """Pausa hasta recibir la decision humana y enruta: aprobada -> ejecutor, rechazada -> sintesis."""
    decision: dict[str, Any] = interrupt(
        {
            "tipo": "aprobacion_requerida",
            "accion": estado["instruccion"],
            "motivo": estado.get("razon", ""),
            "pregunta": estado["pregunta"],
        }
    )
    if decision.get("aprobado"):
        return Command(goto="ejecutor", update={"aprobado": True})
    nota = decision.get("comentario") or "sin comentario"
    rechazo: Hallazgo = {
        "contenido": f"ACCION RECHAZADA por el revisor humano ({nota}). No se ejecuto nada.",
        "fuentes": [],
        "version": 1,
    }
    return Command(
        goto="sintesis",
        update={
            "aprobado": False,
            "hallazgos": {"ejecutor": rechazo},
            "messages": [AIMessage(content=rechazo["contenido"], name="aprobacion")],
        },
    )


router = APIRouter(tags=["hitl"])


@router.post("/tasks/{job_id}/approve", status_code=202)
async def aprobar(job_id: str, cuerpo: Aprobacion, request: Request) -> dict[str, str]:
    """Reanuda una tarea pausada en `AWAITING_APPROVAL`. No bloquea: la reanudacion se encola."""
    worker = request.app.state.worker
    store = request.app.state.store
    job = await store.obtener(job_id)
    if job is None:
        raise HTTPException(404, "job inexistente")
    # Transicion atomica: si dos revisores aprueban a la vez, solo uno gana (el otro recibe 409).
    if not await store.transicion(job_id, EstadoJob.AWAITING_APPROVAL, EstadoJob.PENDING):
        raise HTTPException(409, f"el job esta en {job['status']}: solo se aprueba en AWAITING_APPROVAL")
    await worker.encolar_reanudacion(job_id, cuerpo.aprobado, cuerpo.comentario)
    return {"job_id": job_id, "status": EstadoJob.PENDING.value}
