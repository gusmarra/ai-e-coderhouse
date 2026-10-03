"""Configuracion central: variables de entorno y limites del agente."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

__all__ = [
    "MAX_MENSAJES_CONTEXTO",
    "RECURSION_LIMIT",
    "TRAZAS_DIR",
    "ruta_checkpoints",
]

ROOT = Path(__file__).parent
TRAZAS_DIR = ROOT / "trazas"

load_dotenv(ROOT / ".env")

#: Techo de "super-pasos" del grafo por invocacion. Cada vuelta modelo -> herramienta
#: -> modelo gasta 2 pasos, asi que 10 alcanza para ~4 llamadas a herramientas.
#: Sin este tope, un modelo que se queda llamando herramientas en loop quema la API.
RECURSION_LIMIT = 10

#: Cuantos mensajes (como maximo) le mostramos al modelo en cada llamada. El estado
#: guardado en SQLite conserva TODO el historial; solo recortamos lo que se envia.
MAX_MENSAJES_CONTEXTO = 30


def ruta_checkpoints() -> Path:
    """Archivo SQLite donde el checkpointer guarda el estado de cada thread."""
    return Path(os.getenv("CHECKPOINT_DB") or ROOT / "checkpoints.sqlite")
