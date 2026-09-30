"""Prompt, contexto/referencias, cadena LCEL y `get_rag_response` (rag.py), sin red."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from rag import (
    HUMANO,
    PROMPT,
    SISTEMA,
    _armar_referencias,
    _formatear_contexto,
    construir_generacion,
    get_rag_response,
)
from schemas import MENSAJE_SIN_CONTEXTO, RespuestaModelo, RespuestaRAG


@pytest.fixture
def documentos_falsos() -> list[Document]:
    return [
        Document(
            page_content="El check-in es a partir de las 15:00 hs.",
            metadata={"source": "politica-reservas-cancelaciones.md", "seccion": "3. Check-in y check-out"},
        ),
        Document(
            page_content="El check-out es hasta las 11:00 hs.",
            metadata={"source": "politica-reservas-cancelaciones.md"},
        ),
    ]


# --- prompt ---------------------------------------------------------------
def test_el_prompt_de_sistema_instruye_a_usar_solo_el_contexto():
    assert "CONTEXTO" in SISTEMA
    assert MENSAJE_SIN_CONTEXTO in SISTEMA


def test_el_prompt_humano_expone_contexto_y_pregunta():
    assert "{contexto}" in HUMANO
    assert "{pregunta}" in HUMANO


def test_las_instrucciones_de_formato_del_parser_llegan_al_prompt():
    sistema = PROMPT.format_messages(contexto="CTX", pregunta="P")[0].content
    assert "respuesta" in sistema.lower()
    assert "encontrado_en_contexto" in sistema


# --- contexto y referencias ----------------------------------------------
def test_el_contexto_etiqueta_cada_chunk_con_su_fuente():
    docs = [
        Document(page_content="a", metadata={"source": "uno.md"}),
        Document(page_content="b", metadata={"source": "dos.md"}),
    ]
    contexto = _formatear_contexto(docs)
    assert "[Fuente: uno.md]" in contexto
    assert "[Fuente: dos.md]" in contexto


def test_las_referencias_salen_de_los_metadatos_del_retriever(documentos_falsos):
    refs = _armar_referencias(documentos_falsos)
    assert len(refs) == len(documentos_falsos)
    assert refs[0].fuente == "politica-reservas-cancelaciones.md"
    assert refs[0].seccion == "3. Check-in y check-out"
    assert refs[1].seccion is None


def test_un_chunk_largo_se_recorta_en_la_referencia():
    doc = Document(page_content="x" * 400, metadata={"source": "a.md"})
    assert len(_armar_referencias([doc])[0].fragmento) < 400


def test_sin_documentos_no_rompe_ni_hay_referencias():
    assert _formatear_contexto([]) != ""
    assert _armar_referencias([]) == []


# --- cadena LCEL ----------------------------------------------------------
def test_la_cadena_lcel_parsea_el_json_del_modelo():
    modelo = FakeListChatModel(
        responses=['{"respuesta": "El check-in es a partir de las 15:00 hs.", "encontrado_en_contexto": true}']
    )
    resultado = asyncio.run(
        construir_generacion(modelo).ainvoke({"contexto": "...", "pregunta": "¿A que hora es el check-in?"})
    )
    assert isinstance(resultado, RespuestaModelo)
    assert resultado.encontrado_en_contexto is True


# --- get_rag_response -----------------------------------------------------
def _generacion(respuesta: str, encontrado: bool):
    return RunnableLambda(lambda _: RespuestaModelo(respuesta=respuesta, encontrado_en_contexto=encontrado))


def test_get_rag_response_orquesta_recuperacion_y_generacion(documentos_falsos):
    resultado = asyncio.run(
        get_rag_response(
            "¿A que hora es el check-in?",
            retriever=RunnableLambda(lambda _: documentos_falsos),
            generacion=_generacion("El check-in es a las 15:00 hs.", True),
        )
    )
    assert isinstance(resultado, RespuestaRAG)
    assert resultado.encontrado_en_contexto is True
    assert len(resultado.referencias) == 2


def test_una_pregunta_trampa_devuelve_el_mensaje_fijo(documentos_falsos):
    resultado = asyncio.run(
        get_rag_response(
            "¿El hotel ofrece guarderia o kids club?",
            retriever=RunnableLambda(lambda _: documentos_falsos),
            generacion=_generacion(MENSAJE_SIN_CONTEXTO, False),
        )
    )
    assert resultado.respuesta == MENSAJE_SIN_CONTEXTO


def test_get_rag_response_rechaza_consulta_vacia(documentos_falsos):
    with pytest.raises(ValueError):
        asyncio.run(
            get_rag_response(
                "   ",
                retriever=RunnableLambda(lambda _: documentos_falsos),
                generacion=_generacion("ok", True),
            )
        )
