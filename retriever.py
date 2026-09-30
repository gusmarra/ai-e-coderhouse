"""Recuperador hibrido: busqueda vectorial (Pinecone) + lexica (BM25).

* **Vectorial**: encuentra chunks que *significan* lo mismo que la pregunta,
  aunque no compartan palabras ("¿puedo traer a mi perro?" -> politica de
  mascotas). Falla con terminos exactos: siglas, nombres propios, montos.
* **BM25**: puntua por coincidencia de palabras ponderada por rareza. Acierta
  con "DEA" o "$3.000" y falla con parafrasis.

`EnsembleRetriever` corre ambos y fusiona los rankings con Reciprocal Rank
Fusion (RRF): un chunk suma `peso / (c + posicion)` por cada lista en la que
aparece, asi que los que salen bien rankeados en las dos suben al tope.

`RAGSystem` encapsula todo: recibe una consulta y devuelve los top-5 chunks.

Nota de escala: el BM25 de LangChain vive en memoria y se construye al
arrancar desde los mismos chunks que la ingesta (mismo `chunk_id`). Sirve para
un corpus chico/mediano; para millones de chunks lo escalable es mover la
parte lexica a Pinecone con vectores sparse.
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

with warnings.catch_warnings():  # langchain-community avisa que esta en "sunset"
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

__all__ = [
    "TOP_K",
    "PineconeVectorRetriever",
    "RAGSystem",
    "tokenizar",
]

#: Top-5 que pide la consigna.
TOP_K = 5

#: Cuantos candidatos aporta cada recuperador antes de fusionar. Mas que TOP_K
#: para que la fusion tenga de donde elegir.
K_CANDIDATOS = 10

#: (BM25, vectorial). Parejos: el corpus tiene tanto terminos exactos como parafrasis.
PESOS_POR_DEFECTO = (0.5, 0.5)

_STOPWORDS = frozenset(
    "a al con de del el en es la las lo los me mi o para por que se si su sus un una y u "
    "cual cuales cuanto cuanta cuantos cuantas como donde cuando hay puedo tiene ser son".split()
)


def tokenizar(texto: str) -> list[str]:
    """Minusculas, sin acentos ni puntuacion, sin stopwords en espanol.

    El preprocesado por defecto de BM25Retriever es `str.split`: "check-in."
    y "Check-in" serian terminos distintos y "¿Cuánto" no matchearia "cuanto".
    """
    sin_acentos = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    return [t for t in re.findall(r"[a-z0-9]+", sin_acentos) if t not in _STOPWORDS]


class PineconeVectorRetriever(BaseRetriever):
    """Retriever vectorial sobre el SDK nativo de Pinecone.

    Embebe la consulta, hace `index.query` en un namespace (con filtro de
    metadata opcional) y arma los `Document` con el texto que viaja en la
    metadata de cada vector. Un `Document` de salida conserva todos los
    metadatos (`source`, `doc_id`, `categoria`, ...) mas el `score` de Pinecone.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: Any
    embeddings: Embeddings
    namespace: str
    k: int = K_CANDIDATOS
    filter: dict[str, Any] | None = None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        respuesta = self.index.query(
            vector=self.embeddings.embed_query(query),
            top_k=self.k,
            namespace=self.namespace,
            include_metadata=True,
            filter=self.filter,
        )
        documentos = []
        for match in respuesta.matches:
            metadata = dict(match.metadata or {})
            texto = metadata.pop("text", "")
            metadata["score"] = match.score
            documentos.append(Document(page_content=texto, metadata=metadata))
        return documentos


class RAGSystem:
    """Recuperador hibrido listo para usar: `RAGSystem.desde_entorno().retrieve("...")`.

    `documentos` son los chunks locales (base del BM25); `fabrica_vectorial`
    recibe un filtro de metadata (o `None`) y devuelve el recuperador vectorial.
    Inyectar ambos permite probarlo sin red.
    """

    def __init__(
        self,
        documentos: list[Document],
        fabrica_vectorial: Callable[[dict[str, Any] | None], BaseRetriever],
        *,
        top_k: int = TOP_K,
        k_candidatos: int = K_CANDIDATOS,
        pesos: tuple[float, float] = PESOS_POR_DEFECTO,
    ) -> None:
        if not documentos:
            raise ValueError("RAGSystem necesita al menos un documento para el BM25")
        self.documentos = documentos
        self.top_k = top_k
        self.k_candidatos = k_candidatos
        self.pesos = pesos
        self._fabrica_vectorial = fabrica_vectorial
        self._cache: dict[str | None, tuple[BM25Retriever, BaseRetriever, EnsembleRetriever]] = {}

    @classmethod
    def desde_entorno(cls, **kwargs: Any) -> RAGSystem:
        """Corpus local de `data/` + indice/namespace de Pinecone segun el `.env`."""
        from config import nombre_namespace
        from ingest import cargar_documentos, fragmentar
        from llm import embeddings_por_defecto
        from setup_pinecone import obtener_indice

        index = obtener_indice()
        embeddings = embeddings_por_defecto()
        k = kwargs.get("k_candidatos", K_CANDIDATOS)

        def fabrica(filtro: dict[str, Any] | None) -> BaseRetriever:
            return PineconeVectorRetriever(
                index=index, embeddings=embeddings, namespace=nombre_namespace(), k=k, filter=filtro
            )

        return cls(fragmentar(cargar_documentos()), fabrica, **kwargs)

    def _componentes(self, categoria: str | None) -> tuple[BM25Retriever, BaseRetriever, EnsembleRetriever]:
        if categoria not in self._cache:
            docs = [d for d in self.documentos if categoria in (None, d.metadata.get("categoria"))]
            if not docs:
                raise ValueError(f"no hay documentos de la categoria {categoria!r}")
            bm25 = BM25Retriever.from_documents(docs, k=self.k_candidatos, preprocess_func=tokenizar)
            vectorial = self._fabrica_vectorial({"categoria": {"$eq": categoria}} if categoria else None)
            ensemble = EnsembleRetriever(
                retrievers=[bm25, vectorial], weights=list(self.pesos), id_key="chunk_id"
            )
            self._cache[categoria] = (bm25, vectorial, ensemble)
        return self._cache[categoria]

    def bm25(self, categoria: str | None = None) -> BM25Retriever:
        return self._componentes(categoria)[0]

    def vectorial(self, categoria: str | None = None) -> BaseRetriever:
        return self._componentes(categoria)[1]

    def retrieve(self, query: str, *, categoria: str | None = None) -> list[Document]:
        """Top-k chunks fusionando BM25 y vectorial. `categoria` filtra por metadata en ambos."""
        if not query or not query.strip():
            raise ValueError("la consulta esta vacia")
        return self._componentes(categoria)[2].invoke(query.strip())[: self.top_k]

    async def aretrieve(self, query: str, *, categoria: str | None = None) -> list[Document]:
        if not query or not query.strip():
            raise ValueError("la consulta esta vacia")
        return (await self._componentes(categoria)[2].ainvoke(query.strip()))[: self.top_k]

    # Misma interfaz que un Runnable: permite pasar un RAGSystem donde `rag.py` espera un retriever.
    def invoke(self, query: str) -> list[Document]:
        return self.retrieve(query)

    async def ainvoke(self, query: str) -> list[Document]:
        return await self.aretrieve(query)
