"""Inicializacion del indice (setup_pinecone.py): crear, reutilizar o rechazar."""

from __future__ import annotations

import pytest

from config import EMBEDDING_DIMENSION, METRICA
from setup_pinecone import asegurar_indice, obtener_indice
from tests.fakes import ClientePineconeFalso, IndexFalso


def test_crea_el_indice_serverless_si_no_existe():
    pc = ClientePineconeFalso()
    assert asegurar_indice(pc, nombre="nuevo") is True
    creado = pc.creados[0]
    assert creado["name"] == "nuevo"
    assert creado["dimension"] == EMBEDDING_DIMENSION
    assert creado["metric"] == METRICA
    assert type(creado["spec"]).__name__ == "ServerlessSpec"


def test_no_recrea_un_indice_existente_con_la_dimension_correcta():
    pc = ClientePineconeFalso({"existente": EMBEDDING_DIMENSION})
    assert asegurar_indice(pc, nombre="existente") is False
    assert pc.creados == []


def test_rechaza_un_indice_con_otra_dimension():
    pc = ClientePineconeFalso({"viejo": 768})
    with pytest.raises(ValueError, match="dimension 768"):
        asegurar_indice(pc, nombre="viejo")


def test_obtener_indice_crea_y_devuelve_el_handle():
    pc = ClientePineconeFalso()
    assert isinstance(obtener_indice(pc, nombre="nuevo"), IndexFalso)
    assert "nuevo" in pc.indices


def test_el_nombre_del_indice_sale_de_index_name(monkeypatch):
    from config import nombre_indice

    monkeypatch.setenv("INDEX_NAME", "mi-indice")
    assert nombre_indice() == "mi-indice"
    monkeypatch.delenv("INDEX_NAME")
    assert nombre_indice() == "hotel-bahia-serena"


def test_sin_api_key_el_error_es_claro(monkeypatch):
    from config import pinecone_client

    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="PINECONE_API_KEY"):
        pinecone_client()
