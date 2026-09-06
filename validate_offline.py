"""Validacion offline: sin red, sin API keys y sin los SDKs instalados.

Implementa un proveedor falso sobre `BaseLLMClient` y verifica la maquinaria
transversal: validacion Pydantic, normalizacion de mensajes, streaming,
reintentos con backoff y errores devueltos como dato.

    python validate_offline.py
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, ClassVar

from pydantic import ValidationError

from llm_client import (
    ChatMessage,
    ErrorResponse,
    ModelResponse,
    RateLimitError,
    RetryConfig,
    Role,
)
from llm_client.base import BaseLLMClient
from llm_client.exceptions import AuthenticationError, LLMError
from llm_client.schemas import Conversation, ModelConfig, StreamChunk, Usage

fallas: list[str] = []


def check(condicion: bool, descripcion: str) -> None:
    print(f"  {'ok  ' if condicion else 'FALLA'} {descripcion}")
    if not condicion:
        fallas.append(descripcion)


class ErrorFalsoDelSdk(Exception):
    """Simula la excepcion nativa de un SDK."""

    def __init__(self, kind: str) -> None:
        super().__init__(f"error simulado: {kind}")
        self.kind = kind


class FakeClient(BaseLLMClient):
    """Proveedor de prueba: devuelve texto fijo y falla cuando se le pide."""

    provider_name: ClassVar[str] = "fake"
    default_model: ClassVar[str] = "fake-1"
    api_key_env: ClassVar[str] = "FAKE_API_KEY"

    def __init__(self, *, fallas_previas: int = 0, kind: str = "rate_limit", **kw: Any) -> None:
        kw.setdefault("api_key", "test")
        super().__init__(**kw)
        self.fallas_previas = fallas_previas
        self.kind = kind
        self.llamadas = 0
        self.ultima_conversacion: Conversation | None = None
        self.ultima_config: ModelConfig | None = None

    def _quizas_fallar(self) -> None:
        self.llamadas += 1
        if self.llamadas <= self.fallas_previas:
            raise ErrorFalsoDelSdk(self.kind)

    async def _agenerate(self, conversation: Conversation, config: ModelConfig) -> ModelResponse:
        self.ultima_conversacion = conversation
        self.ultima_config = config
        self._quizas_fallar()
        await asyncio.sleep(0.01)  # simula la latencia de red, sin bloquear
        return ModelResponse(
            content=f"respuesta a: {conversation.turns[-1].content}",
            provider=self.provider_name,
            model=config.model,
            usage=Usage(input_tokens=7, output_tokens=3),
            finish_reason="stop",
        )

    async def _astream(
        self, conversation: Conversation, config: ModelConfig
    ) -> AsyncIterator[StreamChunk]:
        self.ultima_conversacion = conversation
        self._quizas_fallar()
        for indice, palabra in enumerate(["la ", "entropia ", "mide ", "el ", "desorden"]):
            await asyncio.sleep(0)
            yield StreamChunk(
                delta=palabra, index=indice, provider=self.provider_name, model=config.model
            )
        yield StreamChunk(
            index=5,
            provider=self.provider_name,
            model=config.model,
            is_final=True,
            usage=Usage(input_tokens=7, output_tokens=5),
            finish_reason="stop",
        )

    def _translate_error(self, exc: BaseException) -> LLMError | None:
        if isinstance(exc, ErrorFalsoDelSdk):
            if exc.kind == "rate_limit":
                return RateLimitError(str(exc), status_code=429)
            return AuthenticationError(str(exc), status_code=401)
        return None


# ----------------------------------------------------------------------


def test_schemas() -> None:
    print("\n1. Validacion Pydantic")

    try:
        ChatMessage(role="user", content="   ")
        check(False, "content en blanco deberia rechazarse")
    except ValidationError:
        check(True, "content en blanco rechazado")

    try:
        ModelConfig(model="x", temperature=9)
        check(False, "temperature=9 deberia rechazarse")
    except ValidationError:
        check(True, "temperature fuera de rango rechazada")

    try:
        ModelConfig(model="x", top_k=40)  # type: ignore[call-arg]
        check(False, "parametro desconocido deberia rechazarse (extra=forbid)")
    except ValidationError:
        check(True, "parametro desconocido rechazado (usar extra={...})")

    conv = Conversation.coerce("hola")
    check(conv.messages[0].role is Role.USER, "un string suelto se vuelve mensaje de usuario")

    conv = Conversation.coerce(
        [
            ChatMessage.system("se breve"),
            {"role": "user", "content": "a"},
            ChatMessage.user("b"),
        ]
    )
    check(conv.system_prompt == "se breve", "el prompt de sistema se extrae aparte")
    check(len(conv.turns) == 2, "los turnos excluyen el system")
    check(
        len(conv.merged_turns()) == 1 and conv.merged_turns()[0].content == "a\n\nb",
        "dos mensajes de usuario seguidos se fusionan (Anthropic exige alternancia)",
    )

    try:
        Conversation.coerce([ChatMessage.system("solo system")])
        check(False, "conversacion con solo system deberia rechazarse")
    except ValidationError:
        check(True, "conversacion con solo system rechazada")

    usage = Usage(input_tokens=10, output_tokens=5)
    check(usage.total_tokens == 15, "total_tokens es campo calculado")


async def test_generate() -> None:
    print("\n2. generate()")
    async with FakeClient(temperature=0.5) as client:
        respuesta = await client.generate([ChatMessage.system("s"), ChatMessage.user("hola")])
    check(respuesta.content == "respuesta a: hola", "contenido normalizado")
    check(respuesta.latency_ms > 0, "latencia medida")
    check(respuesta.attempts == 1, "un solo intento cuando todo va bien")
    check(respuesta.usage.total_tokens == 10, "usage mapeado")
    check(respuesta.as_message().role is Role.ASSISTANT, "as_message() para el historial")

    client = FakeClient()
    await client.generate("hola", model="otro-modelo", temperature=0.1)
    check(
        client.ultima_config is not None and client.ultima_config.model == "otro-modelo",
        "overrides por llamada sin mutar la config de la instancia",
    )
    check(client.config.model == "fake-1", "la config de la instancia queda intacta")

    try:
        await client.generate("hola", temperature=42)
        check(False, "un override invalido deberia explotar antes de la red")
    except ValidationError:
        check(True, "override invalido se valida antes de salir a la red")


async def test_streaming() -> None:
    print("\n3. streaming")
    async with FakeClient() as client:
        piezas = [t async for t in client.stream_text("hola")]
    check("".join(piezas) == "la entropia mide el desorden", "stream_text() concatena bien")
    check(len(piezas) == 5, "el chunk final no aporta texto vacio a stream_text()")

    async with FakeClient() as client:
        chunks = [c async for c in client.stream("hola")]
    final = chunks[-1]
    check(final.is_final and final.usage is not None, "el ultimo chunk trae usage y finish_reason")
    check(
        final.usage is not None and final.usage.output_tokens == 5,
        "usage del stream disponible sin estado mutable",
    )

    async with FakeClient() as client:
        recolectado = await client.collect_stream("hola")
    check(
        recolectado.content == "la entropia mide el desorden",
        "collect_stream() arma una ModelResponse equivalente",
    )


async def test_reintentos() -> None:
    print("\n4. reintentos y errores")
    rapido = RetryConfig(max_attempts=3, initial_backoff_s=0.01, max_backoff_s=0.02)

    client = FakeClient(fallas_previas=2, kind="rate_limit", retry=rapido)
    respuesta = await client.generate("hola")
    check(respuesta.attempts == 3, "un 429 se reintenta hasta que sale bien")
    check(client.llamadas == 3, "se hicieron exactamente 3 llamadas")

    client = FakeClient(fallas_previas=99, kind="rate_limit", retry=rapido)
    resultado = await client.generate_safe("hola")
    check(isinstance(resultado, ErrorResponse), "generate_safe() devuelve el error como dato")
    check(
        isinstance(resultado, ErrorResponse) and resultado.error_type == "RateLimitError",
        "el error mantiene su tipo",
    )
    check(
        isinstance(resultado, ErrorResponse) and resultado.attempts == 3,
        "informa cuantos intentos se hicieron",
    )
    check(client.llamadas == 3, "no reintenta mas que max_attempts")

    client = FakeClient(fallas_previas=99, kind="auth", retry=rapido)
    resultado = await client.generate_safe("hola")
    check(client.llamadas == 1, "un 401 no se reintenta (no es transitorio)")
    check(
        isinstance(resultado, ErrorResponse) and not resultado.retryable,
        "el error de auth queda marcado como no reintentable",
    )

    client = FakeClient(fallas_previas=99, kind="auth", retry=rapido)
    try:
        await client.generate("hola")
        check(False, "generate() deberia lanzar LLMError")
    except AuthenticationError:
        check(True, "generate() lanza la excepcion tipada del cliente")

    client = FakeClient(fallas_previas=1, kind="rate_limit", retry=rapido)
    piezas = [t async for t in client.stream_text("hola")]
    check(
        "".join(piezas) == "la entropia mide el desorden",
        "el stream se reintenta si falla antes del primer token",
    )


async def test_concurrencia() -> None:
    print("\n5. concurrencia (event loop libre)")
    async with FakeClient() as client:
        inicio = asyncio.get_running_loop().time()
        respuestas = await asyncio.gather(*(client.generate(f"p{i}") for i in range(10)))
        transcurrido = asyncio.get_running_loop().time() - inicio
    check(len(respuestas) == 10, "10 llamadas concurrentes completadas")
    check(
        transcurrido < 0.09,
        f"10 llamadas de 10ms tardaron {transcurrido * 1000:.0f}ms (secuencial seria ~100ms)",
    )

    resultados = await asyncio.gather(
        FakeClient(fallas_previas=99, kind="auth").generate_safe("a"),
        FakeClient().generate_safe("b"),
    )
    check(
        isinstance(resultados[0], ErrorResponse) and isinstance(resultados[1], ModelResponse),
        "una llamada que falla no arrastra al resto del gather",
    )


async def main() -> int:
    print("Validacion offline (no usa red ni API keys)")
    test_schemas()
    await test_generate()
    await test_streaming()
    await test_reintentos()
    await test_concurrencia()

    print(f"\n{'-' * 60}")
    if fallas:
        print(f"{len(fallas)} verificacion(es) fallaron:")
        for falla in fallas:
            print(f"  - {falla}")
        return 1
    print("todas las verificaciones pasaron")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
