"""Demo reproducible: corre los escenarios de la consigna y guarda la traza ReAct.

Escenarios:
  A. Razonamiento multi-paso + memoria. El agente tiene que encadenar dos herramientas
     (buscar_cliente -> buscar_pedidos). La 2da pregunta ("y el ultimo?") depende del
     contexto, y la 3ra se hace con un checkpointer RECIEN abierto sobre el mismo archivo
     SQLite: simula reiniciar el proceso y comprueba que el estado sobrevivio a disco.
  B. Ciclo de retorno. Un id inexistente y un apellido ambiguo: el agente tiene que leer el
     error / la ambiguedad, reintentar con otra estrategia o pedir una aclaracion.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config import RECURSION_LIMIT
from grafo import construir_grafo
from trazas import Turno, ejecutar_turno, formatear_paso

__all__ = ["ejecutar_demo"]

# (thread_id, sesiones). Cada sesion abre su propia conexion SQLite (= un "proceso" distinto).
ESCENARIOS: list[tuple[str, list[list[str]]]] = [
    (
        "demo-multipaso",
        [
            [
                "¿Cuántos pedidos tuvo Laura Gómez y cuál fue el total?",
                "¿Y el último?",
            ],
            ["Cambiando de tema un segundo: ¿de qué clienta estábamos hablando y cuánto era su total?"],
        ],
    ),
    (
        "demo-errores",
        [["¿Cuántos pedidos tuvo el cliente 999?", "Pasame los pedidos de García"]],
    ),
]


def _borrar_db(db_path: Path) -> None:
    for sufijo in ("", "-wal", "-shm"):
        Path(f"{db_path}{sufijo}").unlink(missing_ok=True)


def _guardar(traza: dict[str, Any], lineas: list[str], destino_json: Path, destino_log: Path) -> None:
    destino_json.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": mismos fines de linea en Windows y Linux (el repo se versiona con LF).
    destino_json.write_text(json.dumps(traza, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    destino_log.write_text("\n".join(lineas).lstrip("\n") + "\n", encoding="utf-8", newline="\n")


async def ejecutar_demo(
    llm: BaseChatModel,
    db_path: Path,
    destino_json: Path,
    destino_log: Path,
    *,
    nombre_modelo: str = "desconocido",
) -> dict[str, Any]:
    """Corre los escenarios, imprime la traza en vivo y la persiste como JSON y log."""
    await asyncio.to_thread(_borrar_db, db_path)  # arrancar de cero: sin threads heredados

    lineas: list[str] = []

    def emitir(texto: str) -> None:
        print(texto)
        lineas.append(texto)

    escenarios_json: list[dict[str, Any]] = []
    for thread_id, sesiones in ESCENARIOS:
        emitir(f"\n=== thread_id={thread_id} ===")
        turnos: list[Turno] = []
        for n, preguntas in enumerate(sesiones):
            if n:
                emitir("\n  ... se cierra el checkpointer y se reabre el archivo (reinicio simulado) ...")
            async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
                grafo = construir_grafo(llm, saver)
                for pregunta in preguntas:
                    emitir(f"\nUsuario: {pregunta}")
                    turno = await ejecutar_turno(
                        grafo, pregunta, thread_id, on_paso=lambda p: emitir(formatear_paso(p))
                    )
                    turnos.append(turno)
        escenarios_json.append({"thread_id": thread_id, "turnos": [t.a_dict() for t in turnos]})

    traza = {
        "modelo": nombre_modelo,
        "fecha": datetime.now().astimezone().isoformat(timespec="seconds"),
        "recursion_limit": RECURSION_LIMIT,
        "escenarios": escenarios_json,
    }
    await asyncio.to_thread(_guardar, traza, lineas, destino_json, destino_log)
    return traza
