"""Esquemas Pydantic (schemas.py)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas import MENSAJE_SIN_CONTEXTO, Referencia, RespuestaModelo, RespuestaRAG


def test_respuesta_modelo_valida():
    r = RespuestaModelo(respuesta="El check-in es a partir de las 15:00 hs.", encontrado_en_contexto=True)
    assert r.encontrado_en_contexto is True


def test_respuesta_modelo_acepta_el_mensaje_fijo_de_no_lo_se():
    r = RespuestaModelo(respuesta=MENSAJE_SIN_CONTEXTO, encontrado_en_contexto=False)
    assert r.respuesta == MENSAJE_SIN_CONTEXTO


def test_respuesta_modelo_rechaza_respuesta_vacia():
    with pytest.raises(ValidationError):
        RespuestaModelo(respuesta="", encontrado_en_contexto=True)


def test_respuesta_modelo_rechaza_campos_extra():
    with pytest.raises(ValidationError):
        RespuestaModelo(respuesta="ok", encontrado_en_contexto=True, campo_extra="no deberia existir")


def test_respuesta_rag_agrupa_respuesta_y_referencias():
    ref = Referencia(fuente="politica-reservas-cancelaciones.md", fragmento="Check-in: a partir de las 15:00 hs.")
    rag = RespuestaRAG(respuesta="...", encontrado_en_contexto=True, referencias=[ref])
    assert len(rag.referencias) == 1
    assert rag.referencias[0].seccion is None


def test_respuesta_rag_admite_referencias_vacias():
    r = RespuestaRAG(respuesta=MENSAJE_SIN_CONTEXTO, encontrado_en_contexto=False)
    assert r.referencias == []
