"""Configuracion central: variables de entorno y limites del orquestador."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

__all__ = [
    "DATA_DIR",
    "EMBEDDING_DIMENSION",
    "JOB_TIMEOUT_S",
    "MAX_PASOS",
    "MAX_REFINAMIENTOS",
    "RECURSION_LIMIT",
    "REDIS_URL",
    "WORKER_CONCURRENCIA",
    "nombre_indice",
    "nombre_namespace",
]

PACKAGE_DIR = Path(__file__).parent
ROOT = PACKAGE_DIR.parent
DATA_DIR = PACKAGE_DIR / "data"

load_dotenv(ROOT / ".env")

#: Antidoto contra el "Supervisor Infinito" (1/3): maximo de turnos de especialistas por tarea.
#: Al llegar, el supervisor deja de delegar y manda lo que hay a sintesis.
MAX_PASOS = 6

#: Antidoto (2/3): cuantas veces se puede volver a llamar al MISMO especialista tras su
#: primera pasada. Con 1, cada uno tiene una oportunidad de refinar y nada mas.
MAX_REFINAMIENTOS = 1

#: Antidoto (3/3), la red de seguridad de LangGraph: tope duro de super-pasos por invocacion.
#: Cada turno de especialista gasta 2 (supervisor + especialista); sobra margen para MAX_PASOS
#: mas la sintesis. Si algo se escapa de las reglas anteriores, LangGraph corta con un error.
RECURSION_LIMIT = 2 * MAX_PASOS + 6

#: Dimension del indice de Pinecone de la pre-entrega 4 (tiene que coincidir con la de la ingesta).
EMBEDDING_DIMENSION = 1536

_INDICE_POR_DEFECTO = "hotel-bahia-serena"


def nombre_indice() -> str:
    return os.getenv("INDEX_NAME") or _INDICE_POR_DEFECTO


def nombre_namespace() -> str:
    return os.getenv("PINECONE_NAMESPACE") or _INDICE_POR_DEFECTO

#: Infraestructura de la API (pre-entrega 7).
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
#: Tareas en paralelo que atiende el worker; el resto espera en la cola. Con el free tier de un LLM, bajo.
WORKER_CONCURRENCIA = int(os.getenv("WORKER_CONCURRENCIA", "5"))
#: Tope de tiempo por ejecucion: si el LLM se cuelga, el job pasa a FAILED en vez de quedar RUNNING para siempre.
JOB_TIMEOUT_S = float(os.getenv("JOB_TIMEOUT_S", "300"))
