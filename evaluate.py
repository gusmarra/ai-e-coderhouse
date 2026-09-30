"""Evaluacion del recuperador con un "golden set": Precision@k y Recall@k.

`golden_set.json` es una lista de `{"pregunta": ..., "documento_id_esperado": ...}`
donde se sabe de antemano de que documento sale la respuesta (el id es el
`doc_id` de la metadata, o sea el nombre del archivo sin extension). Para cada
pregunta se recuperan los top-k chunks y se mide, a nivel documento:

* **Recall@k**: de los documentos esperados, que fraccion aparece entre los k
  recuperados. Con un unico documento esperado es 1 (acierto) o 0 (fallo).
* **Precision@k**: de los k chunks recuperados, que fraccion pertenece a un
  documento esperado. Ojo con el techo: si el documento tiene 3 chunks, como
  mucho 3 de los 5 recuperados pueden ser suyos (Precision@5 <= 0.6). Por eso
  el reporte muestra tambien la precision *maxima posible*.

Compara tres modos sobre las mismas preguntas: solo vectorial, solo BM25 e
hibrido, que es donde se ve (o no) lo que aporta combinarlos.

    python evaluate.py                 # k=5, golden_set.json
    python evaluate.py --k 3 --min-recall 0.8
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

__all__ = [
    "cargar_golden_set",
    "evaluar",
    "precision_at_k",
    "precision_maxima",
    "recall_at_k",
]

GOLDEN_SET_POR_DEFECTO = Path(__file__).parent / "golden_set.json"


def precision_at_k(recuperados: list[str], relevantes: Iterable[str], k: int) -> float:
    """Fraccion de los primeros `k` recuperados que es relevante (siempre se divide por `k`)."""
    relevantes = set(relevantes)
    return sum(1 for r in recuperados[:k] if r in relevantes) / k


def recall_at_k(recuperados: list[str], relevantes: Iterable[str], k: int) -> float:
    """Fraccion de los relevantes que aparece entre los primeros `k` recuperados."""
    relevantes = set(relevantes)
    if not relevantes:
        raise ValueError("no hay documentos relevantes: el recall no esta definido")
    return len(relevantes & set(recuperados[:k])) / len(relevantes)


def precision_maxima(chunks_relevantes: int, k: int) -> float:
    """Techo de Precision@k cuando solo existen `chunks_relevantes` chunks utiles."""
    return min(chunks_relevantes, k) / k


def cargar_golden_set(ruta: Path = GOLDEN_SET_POR_DEFECTO) -> list[dict[str, Any]]:
    casos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    if not casos:
        raise ValueError(f"el golden set {ruta} esta vacio")
    for i, caso in enumerate(casos):
        if not caso.get("pregunta") or not caso.get("documento_id_esperado"):
            raise ValueError(f"caso {i} sin 'pregunta' o 'documento_id_esperado': {caso}")
    return casos


def _esperados(caso: dict[str, Any]) -> set[str]:
    esperado = caso["documento_id_esperado"]
    return {esperado} if isinstance(esperado, str) else set(esperado)


def evaluar(
    casos: list[dict[str, Any]],
    buscar: Callable[[str], list[Document]],
    k: int,
    *,
    chunks_por_documento: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Corre `buscar` sobre cada pregunta y promedia las metricas.

    `chunks_por_documento` (doc_id -> cantidad de chunks) habilita el calculo de
    la precision maxima posible.
    """
    filas = []
    for caso in casos:
        relevantes = _esperados(caso)
        recuperados = [d.metadata["doc_id"] for d in buscar(caso["pregunta"])[:k]]
        fila = {
            "pregunta": caso["pregunta"],
            "esperados": sorted(relevantes),
            "recuperados": recuperados,
            "precision": precision_at_k(recuperados, relevantes, k),
            "recall": recall_at_k(recuperados, relevantes, k),
        }
        if chunks_por_documento:
            fila["precision_maxima"] = precision_maxima(
                sum(chunks_por_documento.get(r, 0) for r in relevantes), k
            )
        filas.append(fila)

    def promedio(campo: str) -> float | None:
        valores = [f[campo] for f in filas if campo in f]
        return sum(valores) / len(valores) if valores else None

    return {
        "k": k,
        "filas": filas,
        "precision": promedio("precision"),
        "recall": promedio("recall"),
        "precision_maxima": promedio("precision_maxima"),
    }


# ----------------------------------------------------------------------
# Reporte en consola
# ----------------------------------------------------------------------
def _imprimir_detalle(resultado: dict[str, Any]) -> None:
    k = resultado["k"]
    print(f"\nDetalle del modo hibrido (k={k})")
    print(f"  {'':2} {'P@' + str(k):>5} {'R@' + str(k):>5}  pregunta")
    for fila in resultado["filas"]:
        marca = "ok" if fila["recall"] == 1 else "--"
        print(f"  {marca:2} {fila['precision']:>5.2f} {fila['recall']:>5.2f}  {fila['pregunta']}")
        if fila["recall"] < 1:
            print(f"        esperado: {fila['esperados']}  recuperado: {fila['recuperados']}")


def _imprimir_resumen(resultados: dict[str, dict[str, Any]]) -> None:
    primero = next(iter(resultados.values()))
    k = primero["k"]
    print(f"\nResumen ({len(primero['filas'])} preguntas, k={k})")
    print(f"  {'modo':<10} {'Precision@' + str(k):>12} {'Recall@' + str(k):>10}")
    for modo, r in resultados.items():
        print(f"  {modo:<10} {r['precision']:>12.3f} {r['recall']:>10.3f}")
    if primero["precision_maxima"] is not None:
        print(f"\n  Precision@{k} maxima posible con este chunking: {primero['precision_maxima']:.3f}")
        print("  (un documento con pocos chunks no puede llenar los k lugares)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Precision@k y Recall@k del recuperador hibrido")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--golden", type=Path, default=GOLDEN_SET_POR_DEFECTO)
    parser.add_argument(
        "--min-recall",
        type=float,
        default=None,
        help="sale con codigo 1 si el Recall del hibrido es menor a este valor",
    )
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from retriever import RAGSystem

    casos = cargar_golden_set(args.golden)
    rag = RAGSystem.desde_entorno(top_k=args.k)
    chunks_por_doc = dict(Counter(d.metadata["doc_id"] for d in rag.documentos))

    modos: dict[str, Callable[[str], list[Document]]] = {
        "vectorial": rag.vectorial().invoke,
        "bm25": rag.bm25().invoke,
        "hibrido": rag.retrieve,
    }
    resultados = {
        nombre: evaluar(casos, buscar, args.k, chunks_por_documento=chunks_por_doc)
        for nombre, buscar in modos.items()
    }

    _imprimir_detalle(resultados["hibrido"])
    _imprimir_resumen(resultados)

    if args.min_recall is not None and resultados["hibrido"]["recall"] < args.min_recall:
        print(f"\nFALLA: Recall@{args.k} del hibrido {resultados['hibrido']['recall']:.3f} < {args.min_recall}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
