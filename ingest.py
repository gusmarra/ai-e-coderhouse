"""Pipeline de ingesta: `data/*.md` -> chunks -> embeddings -> Pinecone.

Cada vector que sube a Pinecone lleva en su metadata **el texto original**
(`text`) ademas de su procedencia, asi la consulta devuelve el contenido sin
ir a buscarlo a otra base de datos:

| campo         | ejemplo                                  | para que sirve                     |
| ------------- | ---------------------------------------- | ---------------------------------- |
| `text`        | "Late check-out: hasta las 13:00 hs..."  | contenido del chunk                |
| `source`      | `politica-reservas-cancelaciones.md`     | cita / archivo de origen           |
| `doc_id`      | `politica-reservas-cancelaciones`        | id estable del documento (eval)    |
| `categoria`   | `reservas`                               | filtro por metadata en la consulta |
| `etiquetas`   | `["check-in", "cancelacion"]`            | filtro por metadata                |
| `page`        | `1`                                      | pagina (los .md no tienen: siempre 1) |
| `seccion`     | `3. Check-in y check-out`                | ubicacion dentro del documento     |
| `chunk_index` | `2`                                      | orden del chunk en el documento    |
| `chunk_id`    | `politica-reservas-cancelaciones#002`    | id del vector (upsert idempotente) |

    python ingest.py            # sube los chunks (idempotente: mismos ids, se pisan)
    python ingest.py --reset    # vacia el namespace antes de subir
"""

from __future__ import annotations

import bisect
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DATA_DIR, nombre_namespace

__all__ = [
    "CATALOGO",
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "a_registros",
    "cargar_documentos",
    "fragmentar",
    "ingerir",
]

log = logging.getLogger("pre_entrega_4.ingest")

#: Consigna: chunks de ~500-800 tokens. `RecursiveCharacterTextSplitter` mide
#: en caracteres; se usa la heuristica de ~4 caracteres por token (evita bajar
#: el encoding de tiktoken de internet). Chicos pierden contexto semantico,
#: grandes diluyen el embedding: 500 tokens con 50 de overlap es el punto medio.
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
_CHARS_POR_TOKEN = 4
CHUNK_SIZE = CHUNK_SIZE_TOKENS * _CHARS_POR_TOKEN
CHUNK_OVERLAP = CHUNK_OVERLAP_TOKENS * _CHARS_POR_TOKEN

#: Pinecone recomienda upserts de a lotes chicos (limite de 2 MB por request).
TAMANO_LOTE = 100

#: Categoria y etiquetas de cada documento. Los `.md` no traen front matter,
#: asi que esta es la unica fuente de metadata "de negocio".
CATALOGO: dict[str, dict[str, Any]] = {
    "politica-reservas-cancelaciones.md": {
        "categoria": "reservas",
        "etiquetas": ["reservas", "check-in", "check-out", "cancelacion", "reembolsos", "pagos"],
    },
    "reglamento-interno-huespedes.md": {
        "categoria": "reglamento",
        "etiquetas": ["normas", "mascotas", "fumadores", "visitas", "danos"],
    },
    "servicios-comodidades.md": {
        "categoria": "servicios",
        "etiquetas": ["habitaciones", "restaurante", "spa", "piscina", "wifi", "estacionamiento"],
    },
    "protocolo-seguridad-emergencias.md": {
        "categoria": "seguridad",
        "etiquetas": ["evacuacion", "incendios", "emergencias", "camaras", "acceso"],
    },
}


def cargar_documentos(data_dir: Path = DATA_DIR) -> list[Document]:
    """Lee cada `.md`/`.txt` de `data_dir` como un `Document` con su metadata de negocio."""
    archivos = sorted([*data_dir.glob("*.md"), *data_dir.glob("*.txt")])
    if not archivos:
        raise FileNotFoundError(f"no hay archivos .md/.txt en {data_dir}")
    documentos = []
    for archivo in archivos:
        extra = CATALOGO.get(archivo.name, {"categoria": "general", "etiquetas": []})
        documentos.append(
            Document(
                page_content=archivo.read_text(encoding="utf-8"),
                metadata={
                    "source": archivo.name,
                    "doc_id": archivo.stem,
                    "page": 1,
                    **extra,
                },
            )
        )
    return documentos


_ENCABEZADO = re.compile(r"^#{1,3} (.+)$", re.MULTILINE)


def _seccion_en(texto: str, posicion: int) -> str:
    """Encabezado Markdown vigente en `posicion` del documento original."""
    encabezados = [(m.start(), m.group(1).strip()) for m in _ENCABEZADO.finditer(texto)]
    if not encabezados:
        return ""
    i = bisect.bisect_right([pos for pos, _ in encabezados], posicion) - 1
    return encabezados[max(i, 0)][1]


def fragmentar(documentos: list[Document]) -> list[Document]:
    """Chunking recursivo + metadata por chunk (`chunk_id`, `chunk_index`, `seccion`).

    Corta primero por encabezado, despues por parrafo y oracion, y recien al
    final en medio de una palabra. El `chunk_id` es deterministico (documento +
    orden), asi que re-ingerir el mismo corpus pisa los vectores en vez de
    duplicarlos, y el recuperador BM25 local arma exactamente los mismos ids.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks: list[Document] = []
    for documento in documentos:
        partes = splitter.split_documents([documento])
        for i, parte in enumerate(partes):
            parte.metadata["chunk_index"] = i
            parte.metadata["chunk_id"] = f"{documento.metadata['doc_id']}#{i:03d}"
            parte.metadata["seccion"] = _seccion_en(documento.page_content, parte.metadata["start_index"])
            chunks.append(parte)
    return chunks


def a_registros(chunks: list[Document], embeddings: Embeddings) -> list[dict[str, Any]]:
    """Convierte chunks en registros `{id, values, metadata}` listos para `index.upsert`.

    Los embeddings de todos los chunks se piden en una sola llamada.
    """
    vectores = embeddings.embed_documents([c.page_content for c in chunks])
    registros = []
    for chunk, vector in zip(chunks, vectores, strict=True):
        metadata = {k: v for k, v in chunk.metadata.items() if k != "start_index"}
        metadata["text"] = chunk.page_content
        registros.append({"id": chunk.metadata["chunk_id"], "values": vector, "metadata": metadata})
    return registros


def _contar_vectores(index: Any, namespace: str) -> int:
    namespaces = index.describe_index_stats().namespaces or {}
    resumen = namespaces.get(namespace)
    return int(getattr(resumen, "vector_count", 0) or 0)


def _esperar_consistencia(index: Any, namespace: str, esperados: int, timeout: float = 30.0) -> None:
    """Pinecone es eventualmente consistente: un query inmediato al upsert puede no ver los vectores."""
    limite = time.monotonic() + timeout
    while time.monotonic() < limite:
        if _contar_vectores(index, namespace) >= esperados:
            return
        time.sleep(1.0)
    log.warning("pasaron %.0fs y el namespace '%s' aun no muestra %d vectores", timeout, namespace, esperados)


def ingerir(
    *,
    reset: bool = False,
    index: Any | None = None,
    embeddings: Embeddings | None = None,
    namespace: str | None = None,
    data_dir: Path = DATA_DIR,
    esperar: bool = True,
) -> int:
    """Carga, fragmenta, embebe y sube el corpus. Devuelve la cantidad de chunks subidos."""
    if index is None:
        from setup_pinecone import obtener_indice

        index = obtener_indice()
    if embeddings is None:
        from llm import embeddings_por_defecto

        embeddings = embeddings_por_defecto()
    namespace = namespace or nombre_namespace()

    documentos = cargar_documentos(data_dir)
    chunks = fragmentar(documentos)
    log.info(
        "%d documentos -> %d chunks (~%d tokens, overlap ~%d)",
        len(documentos),
        len(chunks),
        CHUNK_SIZE_TOKENS,
        CHUNK_OVERLAP_TOKENS,
    )

    if reset:
        from pinecone.exceptions import NotFoundException

        try:
            index.delete(delete_all=True, namespace=namespace)
            log.info("namespace '%s' vaciado", namespace)
        except NotFoundException:  # el namespace todavia no existe
            pass

    registros = a_registros(chunks, embeddings)
    for inicio in range(0, len(registros), TAMANO_LOTE):
        index.upsert(vectors=registros[inicio : inicio + TAMANO_LOTE], namespace=namespace)
    log.info("%d vectores subidos al namespace '%s'", len(registros), namespace)

    if esperar:
        _esperar_consistencia(index, namespace, len(registros))
    return len(registros)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")
    total = ingerir(reset="--reset" in sys.argv[1:])
    print(f"Ingesta completa: {total} chunks.")
