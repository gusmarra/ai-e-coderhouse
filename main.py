"""CLI del orquestador multi-agente.

python main.py preguntar "..."     # una consulta, mostrando la delegacion paso a paso
python main.py demo                # la consulta de la consigna; guarda trazas/traza_demo.{json,log}
python main.py diagrama            # regenera docs/grafo.mmd (y docs/grafo.png si hay red)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from config import ROOT, TRAZAS_DIR
from graph import construir_grafo
from llm import crear_modelo
from trazas import ejecutar, formatear_evento, guardar

#: Consulta que obliga al recorrido completo: investigar (politica) -> calcular -> sintetizar.
CONSULTA_DEMO = (
    "Un huesped reservo 4 noches a USD 150 la noche con tarifa de temporada alta y cancela 3 dias "
    "antes del check-in. Segun la politica del hotel, ¿cuanto tendria que pagar de penalidad y que "
    "plazo tenia para cancelar sin cargo?"
)


def _nombre_modelo(llm: object) -> str:
    return str(getattr(llm, "model_name", None) or getattr(llm, "model", "desconocido"))


async def _correr(pregunta: str, *, guardar_en: str | None = None) -> None:
    llm = crear_modelo()
    grafo = construir_grafo(llm)
    print(f"Pregunta: {pregunta}\n")
    traza = await ejecutar(grafo, pregunta, on_evento=lambda e: print(formatear_evento(e) + "\n"))
    print(f"Ruta: {' -> '.join(traza.ruta)}  ({traza.pasos} turnos de especialistas)")
    if guardar_en:
        guardar(traza, TRAZAS_DIR / f"{guardar_en}.json", TRAZAS_DIR / f"{guardar_en}.log", modelo=_nombre_modelo(llm))
        print(f"Traza guardada en {TRAZAS_DIR / guardar_en}.json / .log")


def _diagrama(destino: Path) -> None:
    """Mermaid del grafo real (sale de la propia definicion, no se dibuja a mano)."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    class ModeloNulo(FakeListChatModel):
        """Para compilar el grafo sin API key: solo importa la estructura, nunca se invoca."""

        def with_structured_output(self, *args: object, **kwargs: object) -> ModeloNulo:  # type: ignore[override]
            return self

        def bind_tools(self, *args: object, **kwargs: object) -> ModeloNulo:  # type: ignore[override]
            return self

    grafo = construir_grafo(ModeloNulo(responses=["-"]))
    destino.mkdir(parents=True, exist_ok=True)
    mermaid = grafo.get_graph().draw_mermaid()
    (destino / "grafo.mmd").write_text(mermaid, encoding="utf-8")
    print(mermaid)
    try:
        (destino / "grafo.png").write_bytes(grafo.get_graph().draw_mermaid_png())
        print(f"\nPNG guardado en {destino / 'grafo.png'}")
    except Exception as exc:  # draw_mermaid_png usa la API web de mermaid.ink
        print(f"\n(no se pudo generar el PNG: {type(exc).__name__}; el .mmd alcanza)")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # tildes en consolas Windows
    parser = argparse.ArgumentParser(description="Orquestador multi-agente (LangGraph)")
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("demo", help="corre la consulta de la consigna y guarda la traza")
    preguntar = sub.add_parser("preguntar", help="hace una consulta y muestra la delegacion")
    preguntar.add_argument("pregunta")
    sub.add_parser("diagrama", help="genera el diagrama Mermaid del grafo")
    args = parser.parse_args()
    match args.comando:
        case "demo":
            asyncio.run(_correr(CONSULTA_DEMO, guardar_en="traza_demo"))
        case "preguntar":
            asyncio.run(_correr(args.pregunta))
        case "diagrama":
            _diagrama(ROOT / "docs")


if __name__ == "__main__":
    main()
