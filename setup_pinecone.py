"""Inicializacion del indice de Pinecone (modo Serverless).

Idempotente: si el indice ya existe lo reutiliza (verificando que su
dimension coincida con la de los embeddings); si no existe, lo crea.

    python setup_pinecone.py
"""

from __future__ import annotations

import logging
from typing import Any

from config import (
    EMBEDDING_DIMENSION,
    METRICA,
    nombre_indice,
    pinecone_client,
    pinecone_region,
)

__all__ = ["asegurar_indice", "obtener_indice"]

log = logging.getLogger("pre_entrega_4.setup")


def asegurar_indice(
    pc: Any | None = None,
    *,
    nombre: str | None = None,
    dimension: int = EMBEDDING_DIMENSION,
    metrica: str = METRICA,
) -> bool:
    """Crea el indice si no existe. Devuelve True si lo creo, False si ya estaba.

    Si ya existe con otra dimension lanza `ValueError`: subir vectores de 1536
    a un indice de 768 fallaria recien en el upsert y con un error poco claro.
    """
    pc = pc if pc is not None else pinecone_client()
    nombre = nombre or nombre_indice()

    if pc.has_index(nombre):
        existente = pc.describe_index(nombre).dimension
        if existente != dimension:
            raise ValueError(
                f"el indice '{nombre}' existe con dimension {existente}, pero los embeddings "
                f"tienen {dimension}. Borralo o usa otro INDEX_NAME."
            )
        log.info("el indice '%s' ya existe (dimension %d): no se crea", nombre, dimension)
        return False

    from pinecone import ServerlessSpec

    cloud, region = pinecone_region()
    log.info("creando indice serverless '%s' (%d dims, %s, %s/%s)", nombre, dimension, metrica, cloud, region)
    pc.create_index(
        name=nombre,
        dimension=dimension,
        metric=metrica,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )  # sin `timeout` espera a que el indice quede listo
    return True


def obtener_indice(pc: Any | None = None, *, nombre: str | None = None) -> Any:
    """Asegura que el indice exista y devuelve el handle `Index` para upsert/query."""
    pc = pc if pc is not None else pinecone_client()
    nombre = nombre or nombre_indice()
    asegurar_indice(pc, nombre=nombre)
    return pc.Index(nombre)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")
    creado = asegurar_indice()
    print(f"Indice '{nombre_indice()}': {'creado' if creado else 'ya existia'}.")
