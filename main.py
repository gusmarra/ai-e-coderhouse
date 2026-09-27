"""Mini script de prueba del pipeline RAG, contra la API real.

Corre dos demos, tal como pide la consigna:

1. **Camino feliz**: una pregunta cuya respuesta esta en los documentos
   institucionales del Hotel Bahia Serena (`data/`): politica de reservas,
   reglamento interno, servicios y comodidades, protocolo de seguridad.
2. **Pregunta trampa**: una pregunta sobre algo que los documentos no
   cubren. Verifica que el modelo conteste "no lo se" en vez de alucinar.

La primera corrida ademas puebla `./vectorstore` (ingesta); las siguientes
reabren la coleccion existente sin volver a embeber nada.

Uso::

    cp .env.example .env      # y completar GEMINI_API_KEY (u OPENAI_API_KEY)
    pip install -r requirements.txt
    python main.py             # ingesta (si hace falta) + las dos demos
    python main.py --forzar    # reindexa data/ desde cero antes de correr
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import warnings

from dotenv import load_dotenv

from rag import get_rag_response
from ingest import ingerir
from llm import proveedores_disponibles
from schemas import RespuestaRAG

# En consolas Windows (cp1252) un acento en la respuesta del modelo haria
# fallar el print; con errors="replace" el script nunca muere por eso.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(name)s: %(message)s")
logging.getLogger("clase_3").setLevel(logging.INFO)
for ruidoso in ("httpx", "httpcore", "openai", "google_genai", "chromadb", "urllib3"):
    logging.getLogger(ruidoso).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")
log = logging.getLogger("clase_3.demo")

PREGUNTA_CAMINO_FELIZ = "¿Cuáles son los horarios de check-in y check-out, y cuánto cuesta un late check-out?"

PREGUNTA_TRAMPA = "¿El hotel ofrece servicio de guardería o cuidado de niños (kids club)?"


def titulo(texto: str) -> None:
    print(f"\n{'=' * 72}\n{texto}\n{'=' * 72}")


def mostrar(pregunta: str, resultado: RespuestaRAG) -> None:
    print(f"Pregunta: {pregunta}")
    print(f"Encontrado en contexto: {resultado.encontrado_en_contexto}")
    print(f"Respuesta: {resultado.respuesta}")
    if resultado.referencias:
        print("Referencias:")
        for ref in resultado.referencias:
            print(f"  - {ref.fuente}: {ref.fragmento!r}")
    else:
        print("Referencias: (ninguna)")
    print("\nJSON (RespuestaRAG, Pydantic):")
    print(resultado.model_dump_json(indent=2))


async def demo_camino_feliz() -> None:
    titulo("1. Camino feliz: la respuesta esta en los documentos")
    inicio = time.perf_counter()
    resultado = await get_rag_response(PREGUNTA_CAMINO_FELIZ)
    print(f"-> {(time.perf_counter() - inicio) * 1000:.0f} ms\n")
    mostrar(PREGUNTA_CAMINO_FELIZ, resultado)
    if not resultado.encontrado_en_contexto:
        log.warning("se esperaba encontrado_en_contexto=True: revisar la ingesta o el top_k")


async def demo_pregunta_trampa() -> None:
    titulo("2. Pregunta trampa: no deberia estar en los documentos")
    print("Ningun documento del hotel menciona un servicio de guarderia/kids club; se espera 'no lo se'.\n")
    inicio = time.perf_counter()
    resultado = await get_rag_response(PREGUNTA_TRAMPA)
    print(f"-> {(time.perf_counter() - inicio) * 1000:.0f} ms\n")
    mostrar(PREGUNTA_TRAMPA, resultado)
    if resultado.encontrado_en_contexto:
        log.warning("el modelo afirmo tener la respuesta: posible alucinacion, revisar el prompt")
    else:
        print("\n-> correcto: el modelo no alucino una respuesta que no esta en los documentos.")


async def main() -> int:
    load_dotenv()

    disponibles = proveedores_disponibles()
    if not disponibles:
        print("No hay ninguna API key en el entorno. Copia .env.example a .env y completala.")
        print("Para probar la maquinaria sin red: python validate_offline.py")
        return 1
    print(f"Proveedores con key: {', '.join(disponibles)}")

    titulo("0. Ingesta (solo reindexa si vectorstore/ esta vacio o se paso --forzar)")
    ingerir(forzar="--forzar" in sys.argv[1:])

    for demo in (demo_camino_feliz, demo_pregunta_trampa):
        try:
            await demo()
        except Exception as exc:  # una demo que falla no deberia cortar el resto
            log.error("la demo %s fallo: %s: %s", demo.__name__, type(exc).__name__, exc)

    titulo("Listo")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
