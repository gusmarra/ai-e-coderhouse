"""Observabilidad: una traza por ejecucion del grafo, con un span por nodo / llamada al LLM / tool.

`OBSERVABILITY` elige el backend:

* `phoenix` (por defecto): Arize Phoenix via OpenInference. `register(auto_instrument=True)`
  engancha el instrumentador de LangChain, que cubre LangGraph: cada nodo, cada llamada al modelo
  (con tokens de entrada/salida, de donde Phoenix calcula el costo) y cada herramienta son spans.
* `langsmith`: solo hace falta `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY`; LangChain/LangGraph
  envian las trazas solos. Esta funcion valida que esten las variables.
* `none`: sin trazas (tests).

Cada job lleva `session_id = job_id` y `job_id` en la metadata del `RunnableConfig` (ver
`config_traza`), asi se filtra en el dashboard todo lo que hizo una tarea, incluso si se pausa
por aprobacion y se reanuda en otra traza.
"""

from __future__ import annotations

import logging
import os
from typing import Any

__all__ = ["config_traza", "init_observabilidad"]

log = logging.getLogger("orquestador.observabilidad")

PROYECTO = os.getenv("OBSERVABILITY_PROJECT", "orquestador-hotel")


def init_observabilidad() -> str:
    """Inicializa el backend elegido. Devuelve su nombre ('none' si quedo desactivado)."""
    backend = os.getenv("OBSERVABILITY", "phoenix").lower()
    if backend == "none":
        return "none"
    if backend == "langsmith":
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", PROYECTO)
        if not os.getenv("LANGSMITH_API_KEY"):
            log.warning("OBSERVABILITY=langsmith pero falta LANGSMITH_API_KEY: no se enviaran trazas")
            return "none"
        return "langsmith"
    if backend != "phoenix":
        raise ValueError(f"OBSERVABILITY desconocido: {backend!r} (phoenix | langsmith | none)")
    try:
        from phoenix.otel import register

        # Lee PHOENIX_COLLECTOR_ENDPOINT (por defecto http://localhost:6006). batch=True: no frena al agente.
        register(project_name=PROYECTO, auto_instrument=True, batch=True)
    except Exception:  # un colector caido no debe tumbar la API
        log.exception("no se pudo inicializar Phoenix: la API sigue sin trazas")
        return "none"
    return "phoenix"


def config_traza(job_id: str, **extra: Any) -> dict[str, Any]:
    """Pieza del RunnableConfig que etiqueta la traza con el job (y lo agrupa como sesion)."""
    return {
        "metadata": {"job_id": job_id, "session_id": job_id, "thread_id": job_id, **extra},
        "tags": ["pre-entrega-7", f"job:{job_id}"],
        "run_name": "orquestador-hotel",
    }
