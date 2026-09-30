"""Factory de embeddings (llm.py): dimension del indice y cache de consultas."""

from __future__ import annotations

import pytest

from config import EMBEDDING_DIMENSION
from llm import EmbeddingsConCache, crear_embeddings
from tests.fakes import EmbeddingsFalsos


def test_el_cache_no_repite_la_llamada_por_la_misma_pregunta():
    base = EmbeddingsFalsos()
    cache = EmbeddingsConCache(base)
    assert cache.embed_query("hola") == cache.embed_query("hola")
    cache.embed_query("chau")
    assert base.llamadas_query == 2


@pytest.mark.parametrize(
    ("proveedor", "variable", "atributo"),
    [("gemini", "GEMINI_API_KEY", "output_dimensionality"), ("openai", "OPENAI_API_KEY", "dimensions")],
)
def test_los_embeddings_salen_con_la_dimension_del_indice(monkeypatch, proveedor, variable, atributo):
    monkeypatch.setenv(variable, "clave-falsa")
    modelo = crear_embeddings(proveedor)
    assert getattr(modelo, atributo) == EMBEDDING_DIMENSION
