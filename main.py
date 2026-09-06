"""Script de validacion del cliente.

Corre, para cada proveedor con API key en el entorno:

1. modo normal (`generate`),
2. modo streaming (`stream_text`),
3. manejo de errores (`generate_safe` con una key invalida),
4. concurrencia real (`asyncio.gather` sobre varias preguntas).

Uso::

    cp .env.example .env      # y completar las keys que se quieran probar
    pip install -r requirements.txt
    python main.py                       # todos los proveedores disponibles
    python main.py openai anthropic      # solo esos
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time

from dotenv import load_dotenv

from llm_client import (
    AsyncLLMManager,
    ChatMessage,
    ErrorResponse,
    LLMError,
    ModelResponse,
    Provider,
    RetryConfig,
    available_providers,
)

PREGUNTA = "Que es la entropia? Responde en dos oraciones."
SISTEMA = "Sos un profesor de fisica. Se breve, claro y concreto."

# Los modelos actuales de Anthropic razonan por defecto y ese razonamiento
# consume `max_tokens`: con un techo bajo la respuesta puede volver vacia.
# De ahi que las demos pidan holgura aunque la respuesta sea corta.
MAX_TOKENS = 2000

# En consolas Windows (cp1252) un acento en la respuesta del modelo haria
# fallar el print; con errors="replace" el script nunca muere por eso.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("llm_client").setLevel(logging.INFO)
# El log HTTP de los SDKs se mezcla con el texto del streaming: queda en WARNING.
for ruidoso in ("httpx", "httpx2", "google_genai", "openai", "anthropic"):
    logging.getLogger(ruidoso).setLevel(logging.WARNING)
log = logging.getLogger("demo")


def titulo(texto: str) -> None:
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}")


# ----------------------------------------------------------------------
# 1. Modo normal
# ----------------------------------------------------------------------
async def demo_normal(proveedor: Provider) -> None:
    titulo(f"[{proveedor}] modo normal (await generate)")
    mensajes = [ChatMessage.system(SISTEMA), ChatMessage.user(PREGUNTA)]

    async with AsyncLLMManager(proveedor, temperature=0.3, max_tokens=MAX_TOKENS) as client:
        respuesta = await client.generate(mensajes)

    print(respuesta.content)
    print(
        f"\n-> modelo={respuesta.model} tokens={respuesta.usage.total_tokens} "
        f"({respuesta.usage.input_tokens} in / {respuesta.usage.output_tokens} out) "
        f"latencia={respuesta.latency_ms:.0f}ms intentos={respuesta.attempts} "
        f"finish={respuesta.finish_reason}"
    )


# ----------------------------------------------------------------------
# 2. Modo streaming
# ----------------------------------------------------------------------
async def demo_streaming(proveedor: Provider) -> None:
    titulo(f"[{proveedor}] modo streaming (async for)")
    mensajes = [ChatMessage.system(SISTEMA), ChatMessage.user(PREGUNTA)]

    inicio = time.perf_counter()
    primer_token_ms: float | None = None
    tokens = 0

    async with AsyncLLMManager(proveedor, temperature=0.3, max_tokens=MAX_TOKENS) as client:
        # `stream()` devuelve StreamChunk (con usage al final);
        # `stream_text()` es el atajo que solo emite el texto.
        async for chunk in client.stream(mensajes):
            if chunk.delta:
                if primer_token_ms is None:
                    primer_token_ms = (time.perf_counter() - inicio) * 1000
                print(chunk.delta, end="", flush=True)
                tokens += 1
            elif chunk.is_final:
                print(
                    f"\n\n-> chunks={tokens} "
                    f"primer_token={primer_token_ms or 0:.0f}ms "
                    f"total={(time.perf_counter() - inicio) * 1000:.0f}ms "
                    f"tokens={chunk.usage.total_tokens if chunk.usage else 0} "
                    f"finish={chunk.finish_reason}"
                )


# ----------------------------------------------------------------------
# 3. Errores que no rompen el programa
# ----------------------------------------------------------------------
async def demo_errores(proveedor: Provider) -> None:
    titulo(f"[{proveedor}] manejo de errores (key invalida)")
    print(
        "Esta demo usa una API key falsa A PROPOSITO: todo lo que sigue\n"
        "(incluido el WARNING) es el error esperado, capturado y estructurado.\n"
    )

    # Un solo intento: una key invalida no es transitoria, reintentar no sirve.
    async with AsyncLLMManager(
        proveedor,
        api_key="sk-clave-invalida-a-proposito",
        retry=RetryConfig(max_attempts=1),
    ) as client:
        # generate_safe() devuelve el error como dato en lugar de lanzarlo.
        resultado = await client.generate_safe("hola")
        match resultado:
            case ErrorResponse() as error:
                print(
                    f"error capturado -> tipo={error.error_type} "
                    f"status={error.status_code} retryable={error.retryable} "
                    f"intentos={error.attempts}"
                )
                print(f"mensaje: {error.message[:160]}")
            case ModelResponse() as ok:  # pragma: no cover - no deberia pasar
                print(f"respondio igual?: {ok.content[:80]}")

        # La variante con excepcion, para quien prefiera try/except.
        try:
            await client.generate("hola")
        except LLMError as exc:
            print(f"excepcion tipada -> {type(exc).__name__}: {str(exc)[:120]}")

    print("el programa sigue vivo despues de los dos errores")


# ----------------------------------------------------------------------
# 4. Concurrencia: la prueba de que no bloqueamos el event loop
# ----------------------------------------------------------------------
async def demo_concurrencia(proveedor: Provider) -> None:
    titulo(f"[{proveedor}] tres preguntas en paralelo (asyncio.gather)")
    preguntas = [
        "Define entropia en una oracion.",
        "Define entalpia en una oracion.",
        "Define la energia libre de Gibbs en una oracion.",
    ]

    inicio = time.perf_counter()
    async with AsyncLLMManager(proveedor, temperature=0.0, max_tokens=MAX_TOKENS) as client:
        # generate_safe + gather: si una falla, las otras dos igual llegan.
        resultados = await asyncio.gather(*(client.generate_safe(p) for p in preguntas))
    transcurrido = (time.perf_counter() - inicio) * 1000

    for pregunta, resultado in zip(preguntas, resultados, strict=True):
        if isinstance(resultado, ErrorResponse):
            print(f"- {pregunta} -> FALLO ({resultado.error_type})")
        else:
            print(f"- {pregunta} -> {resultado.content.strip()[:90]}")
    print(f"\n-> 3 llamadas concurrentes en {transcurrido:.0f}ms (secuencial seria ~3x)")


# ----------------------------------------------------------------------
async def main() -> int:
    load_dotenv()  # lee .env del directorio actual

    pedidos = [p.lower() for p in sys.argv[1:]]
    disponibles = available_providers()
    proveedores = [p for p in disponibles if not pedidos or p.value in pedidos]

    if not proveedores:
        print(
            "No hay proveedores para probar.\n"
            "Copia .env.example a .env y completa al menos una key:\n"
            "  OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY"
        )
        if pedidos:
            print(f"\nPedidos: {', '.join(pedidos)}")
            print(f"Con key en el entorno: {', '.join(p.value for p in disponibles) or 'ninguno'}")
        return 1

    print(f"Proveedores a probar: {', '.join(p.value for p in proveedores)}")

    fallas = 0
    for proveedor in proveedores:
        for demo in (demo_normal, demo_streaming, demo_errores, demo_concurrencia):
            try:
                await demo(proveedor)
            except LLMError as exc:
                fallas += 1
                log.error("%s/%s fallo: %s", proveedor, demo.__name__, exc)
            except Exception:  # el script de validacion no debe abortar
                fallas += 1
                log.exception("%s/%s error inesperado", proveedor, demo.__name__)

    titulo("resultado")
    print("todo OK" if fallas == 0 else f"{fallas} demo(s) con error (ver logs arriba)")
    return 0 if fallas == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
