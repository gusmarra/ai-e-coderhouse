from __future__ import annotations

import pytest

from app.agents.analyst_agent import analizar_sentimiento, calcular, evaluar_expresion, extraer_cifras
from app.agents.research_agent import crear_herramientas_investigador
from app.base_hotel import BuscadorHotel, cargar_chunks, tokenizar


@pytest.fixture(scope="module")
def buscador() -> BuscadorHotel:
    return BuscadorHotel(cargar_chunks())  # solo BM25: sin red


def test_calcular_penalidad_de_temporada_alta() -> None:
    assert calcular.invoke({"expresion": "0.5 * 4 * 150"})["resultado"] == 300


def test_calcular_acepta_coma_decimal_y_potencia() -> None:
    assert evaluar_expresion("1,5 * 2") == 3
    assert evaluar_expresion("2^3") == 8


@pytest.mark.parametrize("peligrosa", ["__import__('os').system('x')", "open('f')", "a + 1", "2 ** 9999", ""])
def test_calcular_rechaza_todo_lo_que_no_sea_aritmetica(peligrosa: str) -> None:
    assert "error" in calcular.invoke({"expresion": peligrosa})


def test_calcular_division_por_cero_es_error_accionable() -> None:
    assert "division por cero" in calcular.invoke({"expresion": "1/0"})["error"]


def test_extraer_cifras() -> None:
    texto = "Cargo del 50% de la estadía. Tarifa USD 150. Check-in 15:00 hs. Hasta 7 días antes."
    cifras = extraer_cifras.invoke({"texto": texto})
    assert cifras["porcentajes"] == ["50%"]
    assert cifras["montos"] == ["USD 150"]
    assert "15:00 hs" in cifras["horarios"]
    assert "7 dias" in cifras["plazos"]


def test_sentimiento_distingue_restrictivo_de_favorable() -> None:
    negativo = analizar_sentimiento.invoke({"texto": "Prohibido fumar. Penalidad y multa por daños."})
    positivo = analizar_sentimiento.invoke({"texto": "Desayuno incluido, late check-out gratis, sin cargo."})
    assert negativo["etiqueta"] == "negativo"
    assert positivo["etiqueta"] == "positivo"
    assert "penalidad" in negativo["negativas"]


def test_sentimiento_sin_palabras_clave_es_neutro() -> None:
    assert analizar_sentimiento.invoke({"texto": "El hotel tiene ocho pisos."})["etiqueta"] == "neutro"


def test_tokenizar_ignora_tildes_y_stopwords() -> None:
    assert tokenizar("¿Cuánto cuesta el Late Check-out?") == ["cuesta", "late", "check", "out"]


def test_buscar_devuelve_fragmentos_con_fuente(buscador: BuscadorHotel) -> None:
    buscar = crear_herramientas_investigador(buscador)[0]
    salida = buscar.invoke({"consulta": "tarifa temporada alta cancelacion penalidad"})
    assert "fuente: politica-reservas-cancelaciones.md" in salida
    assert "50%" in salida


def test_buscar_con_consulta_vacia_devuelve_error_en_vez_de_romper(buscador: BuscadorHotel) -> None:
    buscar = crear_herramientas_investigador(buscador)[0]
    assert buscar.invoke({"consulta": "  "}).startswith("Error")


def test_buscador_cae_a_bm25_si_la_busqueda_vectorial_falla() -> None:
    from langchain_core.retrievers import BaseRetriever

    class Roto(BaseRetriever):
        def _get_relevant_documents(self, query: str, **kwargs: object) -> list:  # type: ignore[override]
            raise ConnectionError("sin red")

    b = BuscadorHotel(cargar_chunks(), Roto())
    assert b.modo.startswith("hibrido")
    assert b.buscar("late check-out")  # no lanza: degrada a BM25
