"""Mini script de prueba del pipeline, contra la API real.

Corre cuatro demos:

1. **Camino feliz**: un log de error -> objeto validado.
2. **Prueba de estres**: un texto ambiguo, sin ninguna tecnologia. El esquema
   lo rechaza, la cadena reintenta y termina fallando de forma controlada.
3. **Concurrencia**: varios textos en paralelo con `process_batch`.
4. **finish_reason**: la misma cadena con `max_tokens=16` para forzar un corte
   por limite de tokens y ver como se detecta antes de parsear.

Uso::

    cp .env.example .env      # y completar la key del proveedor a usar
    pip install -r requirements.txt
    python main.py                 # las cuatro demos, ~10 llamadas a la API
    python main.py anthropic       # fuerza un proveedor
    python main.py gemini 1 2      # solo las demos 1 y 2
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import warnings
from collections.abc import Awaitable, Callable

from dotenv import load_dotenv
from langchain_core.runnables import Runnable

from chain import (
    SalidaNoValidaError,
    construir_cadena,
    crear_modelo,
    process_batch,
    process_text,
    proveedores_disponibles,
)
from schemas import ExtraccionTecnica

LOG_DE_ERROR = """
[2026-03-11 02:14:07] ERROR api-gateway: 504 Gateway Timeout en POST /v1/orders.
El pool de conexiones de PostgreSQL quedo saturado (100/100) porque el cache de
Redis se invalido entero tras el deploy y todas las requests de FastAPI fueron a
la base. Los workers de Celery acumulan 18k tareas en la cola y el healthcheck de
Kubernetes esta reiniciando los pods cada 90 segundos.
"""

ARQUITECTURA = """
El servicio de recomendaciones corre en Python 3.12 con FastAPI detras de un
Nginx. Los embeddings se guardan en Qdrant y los modelos se sirven con ONNX
Runtime. La ingesta es un DAG de Airflow que lee de Kafka cada 15 minutos.
Todavia no hay metricas de latencia por endpoint, pero el sistema responde bien.
"""

AMBIGUO = """
Ayer fui a la panaderia de la esquina y estaba cerrada, asi que volvi caminando
por el parque. Habia bastante viento y se me hizo tarde para almorzar.
"""

LOTE = [
    "El endpoint de login tarda 8 segundos: Django hace una query N+1 contra MySQL.",
    "Documentacion interna: el frontend usa React con Vite y se despliega en Vercel.",
    AMBIGUO,
]

# En consolas Windows (cp1252) un acento en la respuesta del modelo haria
# fallar el print; con errors="replace" el script nunca muere por eso.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(name)s: %(message)s")
logging.getLogger("clase_2").setLevel(logging.INFO)
# El log de los SDKs se mezcla con la salida de las demos: solo errores.
for ruidoso in ("httpx", "httpcore", "openai", "anthropic", "google_genai", "urllib3"):
    logging.getLogger(ruidoso).setLevel(logging.ERROR)
# Los modelos con sampling fijo avisan que ignoran temperature: es esperado.
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")
log = logging.getLogger("clase_2.demo")


def titulo(texto: str) -> None:
    print(f"\n{'=' * 72}\n{texto}\n{'=' * 72}")


def mostrar(resultado: ExtraccionTecnica) -> None:
    print(resultado.model_dump_json(indent=2))


# ----------------------------------------------------------------------
# 1. Camino feliz
# ----------------------------------------------------------------------
async def demo_extraccion(cadena: Runnable) -> None:
    titulo("1. Extraccion sobre un log de error")
    inicio = time.perf_counter()
    resultado = await process_text(LOG_DE_ERROR, cadena=cadena)
    print(f"-> {(time.perf_counter() - inicio) * 1000:.0f} ms")
    mostrar(resultado)

    titulo("1b. Extraccion sobre una descripcion de arquitectura")
    mostrar(await process_text(ARQUITECTURA, cadena=cadena))


# ----------------------------------------------------------------------
# 2. Prueba de estres: texto sin ninguna tecnologia
# ----------------------------------------------------------------------
async def demo_texto_ambiguo(cadena: Runnable) -> None:
    titulo("2. Prueba de estres: texto ambiguo (se esperan reintentos y fallo)")
    print("El esquema exige al menos una tecnologia; este texto no tiene ninguna.")
    print("Cada intento deberia loguear 'la salida no cumple el esquema'.\n")

    try:
        mostrar(await process_text(AMBIGUO, cadena=cadena))
        print(
            "\n-> el modelo invento una tecnologia para cumplir el contrato. "
            "Es el limite del enfoque: la validacion atrapa el JSON mal formado, "
            "no el JSON bien formado y falso. Eso se ataca en el prompt "
            "(o agregando un campo de confianza al esquema), no con reintentos."
        )
    except SalidaNoValidaError as exc:
        print(f"\n-> fallo controlado tras agotar los reintentos: {type(exc).__name__}: {exc}")


# ----------------------------------------------------------------------
# 3. Concurrencia
# ----------------------------------------------------------------------
async def demo_lote(cadena: Runnable) -> None:
    titulo("3. Lote concurrente (abatch, los errores vuelven como dato)")
    inicio = time.perf_counter()
    resultados = await process_batch(LOTE, cadena=cadena)
    print(f"-> {len(LOTE)} textos en {(time.perf_counter() - inicio) * 1000:.0f} ms\n")

    for texto, resultado in zip(LOTE, resultados, strict=True):
        etiqueta = " ".join(texto.split())[:60]
        if isinstance(resultado, Exception):
            print(f"[FALLO] {etiqueta}...\n        {type(resultado).__name__}: {resultado}\n")
        else:
            print(f"[OK]    {etiqueta}...\n        {resultado.model_dump_json()}\n")


# ----------------------------------------------------------------------
# 4. finish_reason: respuesta cortada por limite de tokens
# ----------------------------------------------------------------------
async def demo_truncado(proveedor: str | None) -> None:
    titulo("4. finish_reason: max_tokens=16 fuerza una respuesta incompleta")
    print("Sin mirar finish_reason esto seria un error de parseo confuso.\n")

    cadena = construir_cadena(
        crear_modelo(proveedor, max_tokens=16),
        max_intentos=2,  # no tiene sentido insistir: el limite no va a cambiar
    )
    try:
        mostrar(await process_text(LOG_DE_ERROR, cadena=cadena))
        print("\n-> el proveedor devolvio algo valido igual (poco probable)")
    except SalidaNoValidaError as exc:
        print(f"\n-> detectado antes de parsear: {type(exc).__name__}: {exc}")
    except Exception as exc:  # el SDK puede rechazar max_tokens tan bajo
        print(f"\n-> el proveedor rechazo la llamada: {type(exc).__name__}: {exc}")


async def main() -> int:
    load_dotenv()

    # Argumentos sueltos: un nombre de proveedor y/o los numeros de demo.
    argumentos = sys.argv[1:]
    proveedor = next((a for a in argumentos if not a.isdigit()), None)
    pedidas = [a for a in argumentos if a.isdigit()]

    disponibles = proveedores_disponibles()
    if not disponibles:
        print("No hay ninguna API key en el entorno. Copia .env.example a .env y completala.")
        print("Para probar la maquinaria sin red: python validate_offline.py")
        return 1
    print(f"Proveedores con key: {', '.join(disponibles)}")

    # Una sola cadena para las tres primeras demos: armarla por demo no aporta
    # nada y multiplica las llamadas a la API (los planes gratuitos son cortos).
    cadena = construir_cadena(crear_modelo(proveedor))
    demos: dict[str, Callable[[], Awaitable[None]]] = {
        "1": lambda: demo_extraccion(cadena),
        "2": lambda: demo_texto_ambiguo(cadena),
        "3": lambda: demo_lote(cadena),
        "4": lambda: demo_truncado(proveedor),
    }

    for numero in pedidas or list(demos):
        demo = demos.get(numero)
        if demo is None:
            log.warning("no existe la demo %s (opciones: %s)", numero, ", ".join(demos))
            continue
        try:
            await demo()
        except Exception as exc:  # una demo que falla no deberia cortar el resto
            log.error("la demo %s fallo: %s: %s", numero, type(exc).__name__, exc)

    titulo("Listo")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
