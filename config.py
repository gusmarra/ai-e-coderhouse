"""Configuracion central: variables de entorno y constantes del indice.

Todo lo que puede variar entre maquinas (keys, nombre del indice, namespace)
viene del `.env`; lo que define la *forma* del indice (dimension, metrica)
vive aca como constante porque tiene que coincidir entre la ingesta, la
consulta y la creacion del indice en Pinecone.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

__all__ = [
    "DATA_DIR",
    "EMBEDDING_DIMENSION",
    "METRICA",
    "nombre_indice",
    "nombre_namespace",
    "pinecone_client",
    "pinecone_region",
]

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

load_dotenv(ROOT / ".env")

#: Dimension del indice. 1536 es la nativa de OpenAI `text-embedding-3-small`;
#: con Gemini se fuerza el mismo tamano via `output_dimensionality` (ver llm.py)
#: para que el mismo indice sirva con cualquiera de los dos proveedores.
#: Si el indice se creo con otra dimension, Pinecone rechaza el upsert:
#: el clasico "mismatch de dimensiones".
EMBEDDING_DIMENSION = 1536

#: Coseno: lo habitual para embeddings de texto (ignora la magnitud del vector).
METRICA = "cosine"

_INDICE_POR_DEFECTO = "hotel-bahia-serena"


def nombre_indice() -> str:
    return os.getenv("INDEX_NAME") or _INDICE_POR_DEFECTO


def nombre_namespace() -> str:
    """Namespace donde viven los vectores de este corpus.

    Un namespace es una particion del indice: la consulta solo mira los
    vectores de ese namespace, asi que separar corpus (o clientes) evita
    resultados ruidosos y hace la busqueda mas barata.
    """
    return os.getenv("PINECONE_NAMESPACE") or _INDICE_POR_DEFECTO


def pinecone_region() -> tuple[str, str]:
    """(cloud, region) del indice Serverless. El plan gratuito solo admite aws/us-east-1."""
    return os.getenv("PINECONE_CLOUD") or "aws", os.getenv("PINECONE_REGION") or "us-east-1"


def pinecone_client():
    """Cliente del SDK nativo. Falla con un mensaje claro si falta la key."""
    from pinecone import Pinecone

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("falta PINECONE_API_KEY: definila en .env (ver .env.example)")
    return Pinecone(api_key=api_key)
