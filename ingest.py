"""Modulo de ingesta: lee los `.txt`/`.md` de `data/`, los fragmenta y los
persiste en una coleccion de ChromaDB.

Se corre una sola vez (o cada vez que cambian los documentos fuente):
`ingerir()` verifica si la coleccion ya tiene chunks antes de volver a
embeberlos, para no gastar cuota de la API de embeddings en cada corrida de
`main.py` (ver nota de cuota en el README: la key de Gemini del entorno del
curso es de free tier).
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm import embeddings_por_defecto

__all__ = ["ingerir"]

log = logging.getLogger("clase_3.ingest")

DATA_DIR = Path(__file__).parent / "data"
PERSIST_DIR = Path(__file__).parent / "vectorstore"
COLLECTION_NAME = "hotel_bahia_serena"

#: Tamano de chunk pedido por la consigna: "minimo 500 tokens con 50 de
#: overlap". `RecursiveCharacterTextSplitter` mide en caracteres, no en
#: tokens; en vez de `from_tiktoken_encoder` (que en la primera corrida
#: descarga el archivo de encoding de tiktoken desde internet) se usa la
#: heuristica estandar de ~4 caracteres por token en prosa en espanol/ingles.
#: Es una aproximacion documentada, no una medicion exacta, pero evita una
#: dependencia de red para algo que deberia poder correr offline.
CHUNK_SIZE_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
_CHARS_POR_TOKEN = 4
CHUNK_SIZE = CHUNK_SIZE_TOKENS * _CHARS_POR_TOKEN
CHUNK_OVERLAP = CHUNK_OVERLAP_TOKENS * _CHARS_POR_TOKEN


def _cargar_documentos(data_dir: Path = DATA_DIR) -> list[Document]:
    """Lee cada `.md`/`.txt` de `data_dir` como un `Document` con su nombre de archivo en metadata."""
    archivos = sorted(list(data_dir.glob("*.md")) + list(data_dir.glob("*.txt")))
    if not archivos:
        raise FileNotFoundError(f"no hay archivos .md/.txt en {data_dir}")
    return [
        Document(page_content=archivo.read_text(encoding="utf-8"), metadata={"source": archivo.name})
        for archivo in archivos
    ]


def _fragmentar(documentos: list[Document]) -> list[Document]:
    """Chunking recursivo: intenta cortar por parrafo, despues por oracion,
    antes de cortar a lo bruto en medio de una palabra. El overlap evita que
    una oracion clave quede partida justo en el limite entre dos fragmentos.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documentos)


def _abrir_coleccion() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings_por_defecto(),
        persist_directory=str(PERSIST_DIR),
    )


def ingerir(*, forzar: bool = False, data_dir: Path = DATA_DIR) -> Chroma:
    """Puebla (o reabre) la coleccion de ChromaDB en `PERSIST_DIR`.

    Si la coleccion ya tiene documentos y `forzar=False`, no vuelve a leer ni
    a embeber nada: solo abre la coleccion existente y la devuelve. Con
    `forzar=True` la borra y la reconstruye desde cero (usar cuando cambian
    los archivos de `data/`).
    """
    vectorstore = _abrir_coleccion()
    ya_indexada = vectorstore._collection.count() > 0  # noqa: SLF001 - sin API publica para el count

    if ya_indexada and not forzar:
        log.info(
            "coleccion '%s' ya tiene %d chunks: no se reindexa (usar forzar=True para reconstruir)",
            COLLECTION_NAME,
            vectorstore._collection.count(),  # noqa: SLF001
        )
        return vectorstore

    if ya_indexada and forzar:
        log.info("forzando reindexado: se borra la coleccion existente")
        vectorstore.delete_collection()
        vectorstore = _abrir_coleccion()

    documentos = _cargar_documentos(data_dir)
    chunks = _fragmentar(documentos)
    log.info(
        "indexando %d chunks de %d documentos (chunk_size~%d tokens, overlap~%d tokens)",
        len(chunks),
        len(documentos),
        CHUNK_SIZE_TOKENS,
        CHUNK_OVERLAP_TOKENS,
    )
    vectorstore.add_documents(chunks)
    log.info("ingesta completa: %d chunks persistidos en %s", len(chunks), PERSIST_DIR)
    return vectorstore


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")
    ingerir(forzar="--forzar" in sys.argv)
