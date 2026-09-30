"""Pipeline de ingesta a Pinecone (ingest.py), contra un Index falso."""

from __future__ import annotations

import pytest

from config import EMBEDDING_DIMENSION
from ingest import a_registros, ingerir

NAMESPACE = "test-ns"
_TIPOS_PINECONE = (str, int, float, bool)


def test_los_registros_tienen_id_vector_y_metadata(chunks, embeddings):
    registros = a_registros(chunks, embeddings)
    assert len(registros) == len(chunks)
    for r in registros:
        assert set(r) == {"id", "values", "metadata"}
        assert len(r["values"]) == EMBEDDING_DIMENSION


def test_el_texto_original_viaja_en_la_metadata(chunks, embeddings):
    for chunk, r in zip(chunks, a_registros(chunks, embeddings), strict=True):
        assert r["metadata"]["text"] == chunk.page_content
        assert r["metadata"]["source"] == chunk.metadata["source"]


def test_la_metadata_es_compatible_con_pinecone(chunks, embeddings):
    # Pinecone solo admite str, numeros, bool y listas de str; y hasta 40 KB por vector.
    for r in a_registros(chunks, embeddings):
        for campo, valor in r["metadata"].items():
            if isinstance(valor, list):
                assert all(isinstance(v, str) for v in valor), campo
            else:
                assert isinstance(valor, _TIPOS_PINECONE), campo
        assert len(str(r["metadata"]).encode()) < 40_000


def test_ingerir_sube_todos_los_chunks_al_namespace(chunks, embeddings, index):
    total = ingerir(index=index, embeddings=embeddings, namespace=NAMESPACE, esperar=False)
    assert total == len(chunks)
    assert len(index.namespaces[NAMESPACE]) == len(chunks)


def test_ingerir_es_idempotente(chunks, embeddings, index):
    ingerir(index=index, embeddings=embeddings, namespace=NAMESPACE, esperar=False)
    ingerir(index=index, embeddings=embeddings, namespace=NAMESPACE, esperar=False)
    assert len(index.namespaces[NAMESPACE]) == len(chunks)


def test_reset_vacia_el_namespace_antes_de_subir(chunks, embeddings, index):
    index.upsert([{"id": "viejo#000", "values": [0.0] * EMBEDDING_DIMENSION, "metadata": {}}], namespace=NAMESPACE)
    ingerir(index=index, embeddings=embeddings, namespace=NAMESPACE, reset=True, esperar=False)
    assert "viejo#000" not in index.namespaces[NAMESPACE]


def test_los_namespaces_no_se_mezclan(chunks, embeddings, index):
    ingerir(index=index, embeddings=embeddings, namespace="a", esperar=False)
    assert "b" not in index.namespaces


def test_la_ingesta_espera_a_que_pinecone_vea_los_vectores(chunks, embeddings, index):
    # Con el Index falso el conteo es inmediato: no debe colgarse ni loguear timeout.
    assert ingerir(index=index, embeddings=embeddings, namespace=NAMESPACE) == len(chunks)


@pytest.mark.parametrize("lote", [1, 7])
def test_subida_por_lotes(chunks, embeddings, index, monkeypatch, lote):
    monkeypatch.setattr("ingest.TAMANO_LOTE", lote)
    ingerir(index=index, embeddings=embeddings, namespace=NAMESPACE, esperar=False)
    assert all(n <= lote for _, n in index.upserts)
    assert sum(n for _, n in index.upserts) == len(chunks)
