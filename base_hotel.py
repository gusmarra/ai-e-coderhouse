"""Base de conocimiento del Hotel Bahia Serena (la de las pre-entregas 3 y 4).

Reusa lo construido antes, en version compacta:

* mismo corpus (`data/*.md`) y mismo chunking (500 tokens ~ 2000 caracteres, 50 de overlap)
  con `chunk_id`, `categoria` y `seccion` en la metadata;
* recuperador hibrido = BM25 (terminos exactos: "$", "DEA", "late check-out") + vectorial de
  Pinecone (parafrasis), fusionados con Reciprocal Rank Fusion.

Degradacion elegante: si no hay `PINECONE_API_KEY` (o Pinecone/embeddings fallan en runtime),
el buscador sigue con BM25 local. El orquestador nunca se cae porque falte un servicio externo,
y los tests corren sin red.
"""

from __future__ import annotations

import bisect
import logging
import os
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import ConfigDict

with warnings.catch_warnings():  # langchain-community avisa que esta en "sunset"
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from config import DATA_DIR, nombre_indice, nombre_namespace

__all__ = [
    "BuscadorHotel",
    "PineconeVectorRetriever",
    "buscador_por_defecto",
    "cargar_chunks",
    "tokenizar",
]

log = logging.getLogger("orquestador.base_hotel")

CHUNK_SIZE = 500 * 4  # ~500 tokens (heuristica de 4 caracteres por token, como en la pre-entrega 4)
CHUNK_OVERLAP = 50 * 4
TOP_K = 5
K_CANDIDATOS = 10

_CATEGORIAS = {
    "politica-reservas-cancelaciones": "reservas",
    "reglamento-interno-huespedes": "reglamento",
    "servicios-comodidades": "servicios",
    "protocolo-seguridad-emergencias": "seguridad",
}
_STOPWORDS = frozenset(
    "a al con de del el en es la las lo los me mi o para por que se si su sus un una y u "
    "cual cuales cuanto cuanta cuantos cuantas como donde cuando hay puedo tiene ser son".split()
)
_ENCABEZADO = re.compile(r"^#{1,3} (.+)$", re.MULTILINE)


def tokenizar(texto: str) -> list[str]:
    """Minusculas, sin acentos ni puntuacion, sin stopwords: "¿Cuánto" matchea "cuanto"."""
    sin_acentos = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    return [t for t in re.findall(r"[a-z0-9]+", sin_acentos) if t not in _STOPWORDS]


def _seccion_en(texto: str, posicion: int) -> str:
    encabezados = [(m.start(), m.group(1).strip()) for m in _ENCABEZADO.finditer(texto)]
    if not encabezados:
        return ""
    i = bisect.bisect_right([pos for pos, _ in encabezados], posicion) - 1
    return encabezados[max(i, 0)][1]


def cargar_chunks(data_dir: Path = DATA_DIR) -> list[Document]:
    """`data/*.md` -> chunks con `source`, `doc_id`, `categoria`, `seccion` y `chunk_id` estable."""
    archivos = sorted(data_dir.glob("*.md"))
    if not archivos:
        raise FileNotFoundError(f"no hay archivos .md en {data_dir}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks: list[Document] = []
    for archivo in archivos:
        texto = archivo.read_text(encoding="utf-8")
        base = Document(
            page_content=texto,
            metadata={
                "source": archivo.name,
                "doc_id": archivo.stem,
                "categoria": _CATEGORIAS.get(archivo.stem, "general"),
            },
        )
        for i, parte in enumerate(splitter.split_documents([base])):
            parte.metadata["chunk_index"] = i
            parte.metadata["chunk_id"] = f"{archivo.stem}#{i:03d}"
            parte.metadata["seccion"] = _seccion_en(texto, parte.metadata["start_index"])
            chunks.append(parte)
    return chunks


class PineconeVectorRetriever(BaseRetriever):
    """Recuperador vectorial sobre el SDK nativo de Pinecone (el texto viaja en la metadata)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: Any
    embeddings: Embeddings
    namespace: str
    k: int = K_CANDIDATOS

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        respuesta = self.index.query(
            vector=self.embeddings.embed_query(query),
            top_k=self.k,
            namespace=self.namespace,
            include_metadata=True,
        )
        documentos = []
        for match in respuesta.matches:
            metadata = dict(match.metadata or {})
            texto = metadata.pop("text", "")
            metadata["score"] = match.score
            documentos.append(Document(page_content=texto, metadata=metadata))
        return documentos


class BuscadorHotel:
    """`buscar(consulta)` -> top-k chunks. Hibrido si hay vectorial, BM25 puro si no."""

    def __init__(
        self,
        chunks: list[Document],
        vectorial: BaseRetriever | None = None,
        *,
        top_k: int = TOP_K,
    ) -> None:
        if not chunks:
            raise ValueError("BuscadorHotel necesita al menos un chunk")
        self.top_k = top_k
        self._bm25 = BM25Retriever.from_documents(chunks, k=K_CANDIDATOS, preprocess_func=tokenizar)
        self._vectorial = vectorial
        self._hibrido = (
            EnsembleRetriever(retrievers=[self._bm25, vectorial], weights=[0.5, 0.5], id_key="chunk_id")
            if vectorial is not None
            else None
        )

    @property
    def modo(self) -> str:
        return "hibrido (BM25 + Pinecone)" if self._hibrido else "BM25 local"

    def buscar(self, consulta: str) -> list[Document]:
        if not consulta or not consulta.strip():
            raise ValueError("la consulta esta vacia")
        consulta = consulta.strip()
        if self._hibrido is not None:
            try:
                return self._hibrido.invoke(consulta)[: self.top_k]
            except Exception as exc:  # red caida, cuota de embeddings, indice borrado...
                log.warning("busqueda vectorial fallo (%s): sigo con BM25 local", type(exc).__name__)
        return self._bm25.invoke(consulta)[: self.top_k]


def _vectorial_desde_entorno() -> BaseRetriever | None:
    """Recuperador de Pinecone si hay key y el indice existe; `None` en cualquier otro caso."""
    if not os.getenv("PINECONE_API_KEY"):
        return None
    try:
        from pinecone import Pinecone

        from llm import crear_embeddings

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        if not pc.has_index(nombre_indice()):
            log.warning("el indice %r no existe en Pinecone: solo BM25", nombre_indice())
            return None
        return PineconeVectorRetriever(
            index=pc.Index(nombre_indice()), embeddings=crear_embeddings(), namespace=nombre_namespace()
        )
    except Exception as exc:
        log.warning("no pude conectar con Pinecone (%s): solo BM25", type(exc).__name__)
        return None


_buscador: BuscadorHotel | None = None


def buscador_por_defecto() -> BuscadorHotel:
    """Instancia unica del proceso (el BM25 se construye una vez)."""
    global _buscador
    if _buscador is None:
        _buscador = BuscadorHotel(cargar_chunks(), _vectorial_desde_entorno())
        log.info("buscador listo: %s", _buscador.modo)
    return _buscador
