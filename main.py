"""Demo del recuperador hibrido contra Pinecone (y, opcionalmente, generacion con LLM).

Uso::

    python main.py                                   # pregunta de ejemplo
    python main.py "¿Cuánto cuesta el valet parking?"
    python main.py "¿Dónde está el DEA?" --categoria seguridad   # filtro por metadata
    python main.py "..." --generar                   # ademas redacta la respuesta con el LLM

Requiere haber corrido antes `python ingest.py` (ver README).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import warnings

from retriever import RAGSystem
from schemas import RespuestaRAG

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(name)s: %(message)s")
for ruidoso in ("httpx", "httpcore", "openai", "google_genai", "urllib3"):
    logging.getLogger(ruidoso).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")

PREGUNTA_EJEMPLO = "¿Cuáles son los horarios de check-in y check-out, y cuánto cuesta un late check-out?"


def mostrar_recuperados(rag: RAGSystem, pregunta: str, categoria: str | None) -> None:
    print(f"\nPregunta: {pregunta}" + (f"  [categoria={categoria}]" if categoria else ""))
    print(f"Top-{rag.top_k} (BM25 + vectorial):")
    for i, doc in enumerate(rag.retrieve(pregunta, categoria=categoria), start=1):
        m = doc.metadata
        print(f"  {i}. {m.get('source')} · {m.get('seccion')} · {m.get('categoria')}")
        print(f"     {doc.page_content[:110].strip()!r}...")


def mostrar_respuesta(resultado: RespuestaRAG) -> None:
    print(f"\nRespuesta: {resultado.respuesta}")
    print(f"Encontrado en contexto: {resultado.encontrado_en_contexto}")


async def generar(pregunta: str, rag: RAGSystem) -> RespuestaRAG:
    from rag import get_rag_response

    return await get_rag_response(pregunta, retriever=rag)


def main() -> int:
    parser = argparse.ArgumentParser(description="Consulta al recuperador hibrido")
    parser.add_argument("pregunta", nargs="?", default=PREGUNTA_EJEMPLO)
    parser.add_argument("--categoria", help="filtra por metadata: reservas | reglamento | servicios | seguridad")
    parser.add_argument("--generar", action="store_true", help="redacta la respuesta con el LLM (gasta cuota)")
    args = parser.parse_args()

    rag = RAGSystem.desde_entorno()
    mostrar_recuperados(rag, args.pregunta, args.categoria)
    if args.generar:
        mostrar_respuesta(asyncio.run(generar(args.pregunta, rag)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
