"""Recuperador hibrido (retriever.py): BM25 + vectorial fusionados con EnsembleRetriever."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from ingest import ingerir
from retriever import TOP_K, PineconeVectorRetriever, RAGSystem, tokenizar

NAMESPACE = "test-ns"


def _rag_con_vectorial_ciego(chunks, top_k=TOP_K) -> RAGSystem:
    """Vectorial que siempre devuelve los mismos chunks de otro documento: el BM25 tiene que rescatar."""
    irrelevantes = [c for c in chunks if c.metadata["doc_id"] == "servicios-comodidades"][:3]
    return RAGSystem(chunks, lambda _filtro: RunnableLambda(lambda _q: irrelevantes), top_k=top_k)


# --- tokenizador ----------------------------------------------------------
def test_tokenizar_normaliza_acentos_mayusculas_y_puntuacion():
    assert tokenizar("¿Cuánto cuesta el Check-in?") == ["cuesta", "check", "in"]


def test_tokenizar_descarta_stopwords():
    assert "de" not in tokenizar("tarjeta de credito")


# --- recuperador vectorial sobre Pinecone --------------------------------
def test_el_retriever_vectorial_devuelve_documentos_con_texto_y_metadata(chunks, embeddings, index):
    ingerir(index=index, embeddings=embeddings, namespace=NAMESPACE, esperar=False)
    retriever = PineconeVectorRetriever(index=index, embeddings=embeddings, namespace=NAMESPACE, k=3)
    docs = retriever.invoke("valet parking por dia")
    assert len(docs) == 3
    assert all(d.page_content for d in docs)
    assert "text" not in docs[0].metadata
    assert {"source", "doc_id", "categoria", "chunk_id", "score"} <= set(docs[0].metadata)


def test_el_retriever_vectorial_respeta_el_namespace(chunks, embeddings, index):
    ingerir(index=index, embeddings=embeddings, namespace=NAMESPACE, esperar=False)
    otro = PineconeVectorRetriever(index=index, embeddings=embeddings, namespace="otro", k=3)
    assert otro.invoke("valet parking") == []


def test_el_retriever_vectorial_aplica_el_filtro_de_metadata(chunks, embeddings, index):
    ingerir(index=index, embeddings=embeddings, namespace=NAMESPACE, esperar=False)
    retriever = PineconeVectorRetriever(
        index=index, embeddings=embeddings, namespace=NAMESPACE, k=10, filter={"categoria": {"$eq": "seguridad"}}
    )
    assert {d.metadata["categoria"] for d in retriever.invoke("tarjeta")} == {"seguridad"}


# --- hibrido ---------------------------------------------------------------
def test_devuelve_como_maximo_top_k_documentos(chunks):
    rag = _rag_con_vectorial_ciego(chunks, top_k=5)
    assert len(rag.retrieve("tarjeta magnetica")) <= 5


def test_bm25_rescata_un_termino_exacto_que_el_vectorial_no_ve(chunks):
    rag = _rag_con_vectorial_ciego(chunks)
    solo_vectorial = rag.vectorial().invoke("desfibrilador DEA")
    assert not any("desfibrilador" in d.page_content.lower() for d in solo_vectorial)

    hibrido = rag.retrieve("desfibrilador DEA")
    # RRF suma contribuciones: un chunk presente en ambas listas puede superar al rank 1 de una
    # sola, asi que no se exige el primer puesto, solo que BM25 lo meta en el top-5.
    assert any("desfibrilador" in d.page_content.lower() for d in hibrido)


def test_la_fusion_no_duplica_chunks(chunks):
    # Mismo chunk por ambas ramas: debe aparecer una sola vez (dedup por chunk_id).
    objetivo = next(c for c in chunks if "valet" in c.page_content.lower())
    rag = RAGSystem(chunks, lambda _f: RunnableLambda(lambda _q: [objetivo]))
    ids = [d.metadata["chunk_id"] for d in rag.retrieve("valet parking")]
    assert len(ids) == len(set(ids))
    assert ids[0] == objetivo.metadata["chunk_id"]  # aparece en ambas listas: sube al tope


def test_el_filtro_de_categoria_llega_a_las_dos_ramas(chunks):
    recibido = {}

    def fabrica(filtro):
        recibido["filtro"] = filtro
        return RunnableLambda(lambda _q: [])

    rag = RAGSystem(chunks, fabrica)
    docs = rag.retrieve("tarjeta", categoria="seguridad")
    assert recibido["filtro"] == {"categoria": {"$eq": "seguridad"}}
    assert docs
    assert {d.metadata["categoria"] for d in docs} == {"seguridad"}


def test_categoria_inexistente_es_un_error(chunks):
    with pytest.raises(ValueError, match="categoria"):
        _rag_con_vectorial_ciego(chunks).retrieve("hola", categoria="no-existe")


def test_consulta_vacia_es_un_error(chunks):
    with pytest.raises(ValueError):
        _rag_con_vectorial_ciego(chunks).retrieve("   ")


def test_sin_documentos_locales_es_un_error():
    with pytest.raises(ValueError):
        RAGSystem([], lambda _f: RunnableLambda(lambda _q: []))


def test_version_asincrona_y_runnable_equivalentes(chunks):
    rag = _rag_con_vectorial_ciego(chunks)
    sync = [d.metadata["chunk_id"] for d in rag.retrieve("valet parking")]
    assert [d.metadata["chunk_id"] for d in asyncio.run(rag.aretrieve("valet parking"))] == sync
    assert [d.metadata["chunk_id"] for d in rag.invoke("valet parking")] == sync


def test_extremo_a_extremo_con_index_falso(chunks, embeddings, index):
    ingerir(index=index, embeddings=embeddings, namespace=NAMESPACE, esperar=False)

    def fabrica(filtro):
        return PineconeVectorRetriever(
            index=index, embeddings=embeddings, namespace=NAMESPACE, k=10, filter=filtro
        )

    rag = RAGSystem(chunks, fabrica)
    docs: list[Document] = rag.retrieve("cuanto cuesta el valet parking por dia")
    assert len(docs) == 5
    assert "servicios-comodidades" in {d.metadata["doc_id"] for d in docs}
