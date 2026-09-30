"""Chunking de data/*.md y metadata de cada chunk (ingest.py)."""

from __future__ import annotations

from ingest import CATALOGO, CHUNK_OVERLAP, CHUNK_SIZE, cargar_documentos, fragmentar


def test_hay_al_menos_tres_documentos(documentos):
    assert len(documentos) >= 3


def test_cada_documento_tiene_source_y_doc_id_distintos(documentos):
    assert len({d.metadata["source"] for d in documentos}) == len(documentos)
    assert len({d.metadata["doc_id"] for d in documentos}) == len(documentos)


def test_todos_los_documentos_estan_en_el_catalogo(documentos):
    assert {d.metadata["source"] for d in documentos} <= set(CATALOGO)


def test_la_fragmentacion_genera_mas_de_un_chunk_por_documento(documentos, chunks):
    assert len(chunks) > len(documentos)


def test_ningun_chunk_supera_1_5x_el_chunk_size(chunks):
    # Margen: el splitter recursivo puede pasarse un poco si un separador (p.ej. una
    # fila de tabla) no admite un corte mas fino; no debe ser varias veces el pedido.
    assert not [c for c in chunks if len(c.page_content) > CHUNK_SIZE * 1.5]


def test_overlap_mayor_a_cero():
    assert CHUNK_OVERLAP > 0


def test_no_se_pierde_ningun_documento(documentos, chunks):
    assert {c.metadata["doc_id"] for c in chunks} == {d.metadata["doc_id"] for d in documentos}


def test_cada_chunk_conserva_la_metadata_de_negocio(chunks):
    for c in chunks:
        m = c.metadata
        assert m["source"].endswith(".md")
        assert m["categoria"] in {"reservas", "reglamento", "servicios", "seguridad"}
        assert m["etiquetas"]
        assert all(isinstance(e, str) for e in m["etiquetas"])
        assert m["page"] == 1
        assert m["seccion"]


def test_chunk_id_es_unico_y_deterministico(chunks):
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids == [c.metadata["chunk_id"] for c in fragmentar(cargar_documentos())]


def test_chunk_index_arranca_en_cero_en_cada_documento(chunks):
    primeros = {c.metadata["doc_id"] for c in chunks if c.metadata["chunk_index"] == 0}
    assert primeros == {c.metadata["doc_id"] for c in chunks}


def test_la_seccion_corresponde_al_contenido(chunks):
    # El chunk que habla del late check-out tiene que caer bajo la seccion de check-in/check-out.
    chunk = next(c for c in chunks if "Late check-out" in c.page_content)
    assert "Check-in y check-out" in chunk.metadata["seccion"]
