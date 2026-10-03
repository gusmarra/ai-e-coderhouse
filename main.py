"""CLI del agente ReAct.

python main.py demo                              # corre los escenarios y escribe trazas/
python main.py preguntar "..." --thread-id ana   # una pregunta; el thread recuerda
python main.py chat --thread-id ana              # conversacion interactiva
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config import TRAZAS_DIR, ruta_checkpoints
from demo import ejecutar_demo
from grafo import construir_grafo
from llm import crear_modelo
from trazas import ejecutar_turno, formatear_paso


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agente de razonamiento ciclico (LangGraph + SQLite)")
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("demo", help="corre los escenarios de la consigna y guarda la traza en trazas/")
    preguntar = sub.add_parser("preguntar", help="hace una pregunta y termina")
    preguntar.add_argument("pregunta")
    chat = sub.add_parser("chat", help="conversacion interactiva (Enter vacio para salir)")
    for p in (preguntar, chat):
        p.add_argument("--thread-id", default="default", help="misma id = misma conversacion")
    return parser


async def _chat(thread_id: str, db: Path) -> None:
    llm = crear_modelo()
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        grafo = construir_grafo(llm, saver)
        while pregunta := (await asyncio.to_thread(input, "\nUsuario: ")).strip():
            await ejecutar_turno(grafo, pregunta, thread_id, on_paso=lambda p: print(formatear_paso(p)))


async def _preguntar(pregunta: str, thread_id: str, db: Path) -> None:
    llm = crear_modelo()
    async with AsyncSqliteSaver.from_conn_string(str(db)) as saver:
        grafo = construir_grafo(llm, saver)
        await ejecutar_turno(grafo, pregunta, thread_id, on_paso=lambda p: print(formatear_paso(p)))


async def _demo() -> None:
    llm = crear_modelo()
    nombre = str(getattr(llm, "model_name", None) or getattr(llm, "model", "desconocido"))
    await ejecutar_demo(
        llm,
        db_path=ruta_checkpoints().with_name("demo_checkpoints.sqlite"),
        destino_json=TRAZAS_DIR / "traza_ejemplo.json",
        destino_log=TRAZAS_DIR / "traza_ejemplo.log",
        nombre_modelo=nombre,
    )
    print(f"\nTraza guardada en {TRAZAS_DIR}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]  # tildes en consolas Windows
    args = _parser().parse_args()
    match args.comando:
        case "demo":
            asyncio.run(_demo())
        case "preguntar":
            asyncio.run(_preguntar(args.pregunta, args.thread_id, ruta_checkpoints()))
        case "chat":
            asyncio.run(_chat(args.thread_id, ruta_checkpoints()))


if __name__ == "__main__":
    main()
