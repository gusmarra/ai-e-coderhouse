"""Metricas de evaluate.py y validez del golden set."""

from __future__ import annotations

import json

import pytest
from langchain_core.documents import Document

from evaluate import (
    GOLDEN_SET_POR_DEFECTO,
    cargar_golden_set,
    evaluar,
    main,
    precision_at_k,
    precision_maxima,
    recall_at_k,
)


def _docs(*doc_ids: str) -> list[Document]:
    return [Document(page_content="x", metadata={"doc_id": d}) for d in doc_ids]


# --- metricas -------------------------------------------------------------
def test_precision_at_k_cuenta_relevantes_sobre_k():
    assert precision_at_k(["a", "b", "a", "c", "a"], {"a"}, 5) == pytest.approx(0.6)


def test_precision_divide_por_k_aunque_haya_menos_resultados():
    assert precision_at_k(["a"], {"a"}, 5) == pytest.approx(0.2)


def test_precision_solo_mira_los_primeros_k():
    assert precision_at_k(["b", "b", "a"], {"a"}, 2) == 0.0


def test_recall_at_k_es_uno_si_el_documento_esta_en_el_top_k():
    assert recall_at_k(["b", "c", "a"], {"a"}, 5) == 1.0


def test_recall_at_k_es_cero_si_el_documento_no_esta():
    assert recall_at_k(["b", "c"], {"a"}, 5) == 0.0


def test_recall_fuera_del_top_k_no_cuenta():
    assert recall_at_k(["b", "c", "a"], {"a"}, 2) == 0.0


def test_recall_con_varios_relevantes_es_fraccionario():
    assert recall_at_k(["a", "x"], {"a", "b"}, 5) == pytest.approx(0.5)


def test_recall_sin_relevantes_no_esta_definido():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 5)


def test_precision_maxima_es_el_techo_por_cantidad_de_chunks():
    assert precision_maxima(3, 5) == pytest.approx(0.6)
    assert precision_maxima(9, 5) == 1.0


# --- evaluar --------------------------------------------------------------
def test_evaluar_promedia_sobre_las_preguntas():
    casos = [
        {"pregunta": "p1", "documento_id_esperado": "a"},
        {"pregunta": "p2", "documento_id_esperado": "b"},
    ]
    buscar = {"p1": _docs("a", "a", "x", "x", "x"), "p2": _docs("x", "x", "x", "x", "x")}.__getitem__
    r = evaluar(casos, buscar, 5, chunks_por_documento={"a": 3, "b": 2})
    assert r["recall"] == pytest.approx(0.5)
    assert r["precision"] == pytest.approx((0.4 + 0.0) / 2)
    assert r["precision_maxima"] == pytest.approx((0.6 + 0.4) / 2)
    assert r["filas"][1]["recall"] == 0.0


def test_evaluar_acepta_varios_documentos_esperados():
    casos = [{"pregunta": "p", "documento_id_esperado": ["a", "b"]}]
    r = evaluar(casos, lambda _: _docs("a", "x"), 5)
    assert r["recall"] == pytest.approx(0.5)
    assert r["precision_maxima"] is None


# --- golden set -----------------------------------------------------------
def test_el_golden_set_tiene_al_menos_cinco_preguntas():
    assert len(cargar_golden_set()) >= 5


def test_el_golden_set_referencia_documentos_que_existen(documentos):
    existentes = {d.metadata["doc_id"] for d in documentos}
    for caso in cargar_golden_set():
        assert caso["documento_id_esperado"] in existentes, caso["pregunta"]


def test_el_golden_set_cubre_todos_los_documentos(documentos):
    esperados = {c["documento_id_esperado"] for c in cargar_golden_set()}
    assert esperados == {d.metadata["doc_id"] for d in documentos}


def test_un_golden_set_mal_formado_se_rechaza(tmp_path):
    ruta = tmp_path / "g.json"
    ruta.write_text(json.dumps([{"pregunta": "sin respuesta esperada"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="documento_id_esperado"):
        cargar_golden_set(ruta)


def test_el_golden_set_por_defecto_existe():
    assert GOLDEN_SET_POR_DEFECTO.exists()


def test_main_informa_un_reporte_y_respeta_min_recall(chunks, monkeypatch, capsys):
    from langchain_core.runnables import RunnableLambda

    from retriever import RAGSystem

    def desde_entorno(**kwargs):
        ciego = [c for c in chunks if c.metadata["doc_id"] == "servicios-comodidades"][:3]
        return RAGSystem(chunks, lambda _f: RunnableLambda(lambda _q: ciego), **kwargs)

    monkeypatch.setattr(RAGSystem, "desde_entorno", staticmethod(desde_entorno))

    assert main(["--min-recall", "0"]) == 0
    salida = capsys.readouterr().out
    assert "Precision@5" in salida
    assert "Recall@5" in salida
    assert "hibrido" in salida

    assert main(["--min-recall", "1.01"]) == 1
