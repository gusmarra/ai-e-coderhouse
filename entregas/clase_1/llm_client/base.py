"""Clase base asincrona: reintentos, validacion y traduccion de errores.

Los proveedores concretos solo implementan tres metodos privados
(`_agenerate`, `_astream`, `_translate_error`). Todo lo transversal
--validar entrada, medir latencia, reintentar con backoff, cerrar el
cliente HTTP-- vive aca una sola vez.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import os
import random
import time
from collections.abc import AsyncIterator, Iterable
from typing import Any, ClassVar, Self

from .exceptions import ConfigurationError, LLMError, ServiceUnavailableError
from .schemas import (
    Conversation,
    ErrorResponse,
    MessageLike,
    ModelConfig,
    ModelResponse,
    RetryConfig,
    StreamChunk,
    Usage,
)

logger = logging.getLogger(__name__)

__all__ = ["BaseLLMClient"]


class BaseLLMClient(abc.ABC):
    """Interfaz comun a OpenAI, Anthropic y Gemini.

    Uso tipico::

        async with OpenAIClient() as client:
            respuesta = await client.generate("Que es la entropia?")
            print(respuesta.content)

            async for token in client.stream_text("Contame un chiste"):
                print(token, end="", flush=True)
    """

    #: Nombre corto del proveedor, usado en logs y en las respuestas.
    provider_name: ClassVar[str]
    #: Modelo por defecto si el llamador no especifica ninguno.
    default_model: ClassVar[str]
    #: Variable de entorno donde se busca la credencial.
    api_key_env: ClassVar[str]

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        config: ModelConfig | None = None,
        retry: RetryConfig | None = None,
        **config_overrides: Any,
    ) -> None:
        base_config = config or ModelConfig(model=model or self.default_model)
        updates: dict[str, Any] = dict(config_overrides)
        if model:
            updates["model"] = model
        if updates:
            # Revalidamos el merge en lugar de mutar: `model_copy` no valida.
            base_config = ModelConfig.model_validate(base_config.model_dump() | updates)
        self.config: ModelConfig = base_config
        self.retry: RetryConfig = retry or RetryConfig()

        self._api_key = api_key or os.getenv(self.api_key_env)
        if not self._api_key:
            raise ConfigurationError(
                f"falta la API key: pasala por argumento o definí {self.api_key_env}",
                provider=self.provider_name,
                model=self.config.model,
            )

    # ------------------------------------------------------------------
    # A implementar por cada proveedor
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def _agenerate(self, conversation: Conversation, config: ModelConfig) -> ModelResponse:
        """Una sola llamada no bloqueante al SDK asincrono del proveedor."""

    @abc.abstractmethod
    def _astream(
        self, conversation: Conversation, config: ModelConfig
    ) -> AsyncIterator[StreamChunk]:
        """Generador asincrono de chunks del proveedor."""

    @abc.abstractmethod
    def _translate_error(self, exc: BaseException) -> LLMError | None:
        """Convierte una excepcion nativa del SDK en una `LLMError`.

        Devuelve `None` si la excepcion no es del SDK (ahi se propaga tal cual).
        """

    async def aclose(self) -> None:  # noqa: B027 - opcional a proposito
        """Libera el cliente HTTP subyacente. Sobrescribir si aplica."""

    # ------------------------------------------------------------------
    # API publica
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: MessageLike | Iterable[MessageLike],
        **overrides: Any,
    ) -> ModelResponse:
        """Genera una respuesta completa. Reintenta los errores transitorios.

        Lanza `LLMError` si se agotan los intentos. Si preferís no manejar
        excepciones, usá `generate_safe()`.
        """
        conversation = Conversation.coerce(messages)
        config = self._resolve_config(overrides)

        ultimo: LLMError | None = None
        for intento in range(1, self.retry.max_attempts + 1):
            inicio = time.perf_counter()
            try:
                respuesta = await self._agenerate(conversation, config)
            except BaseException as exc:  # amplio a proposito: se reclasifica abajo
                error = self._clasificar(exc, config)
                if error is None:
                    raise
                ultimo = error
                error.attempts = intento
                if not await self._deberia_reintentar(error, intento):
                    raise error from exc
                continue

            latencia_ms = (time.perf_counter() - inicio) * 1000
            return respuesta.model_copy(update={"latency_ms": latencia_ms, "attempts": intento})

        # Inalcanzable: el loop siempre retorna o lanza. Red de seguridad.
        raise ultimo or ServiceUnavailableError(
            "no se obtuvo respuesta", provider=self.provider_name, model=config.model
        )

    async def generate_safe(
        self,
        messages: MessageLike | Iterable[MessageLike],
        **overrides: Any,
    ) -> ModelResponse | ErrorResponse:
        """Igual que `generate()` pero devuelve el error como dato.

        Pensado para loops y `asyncio.gather`: una key invalida o un 429
        persistente no rompen el resto del batch.
        """
        try:
            return await self.generate(messages, **overrides)
        except LLMError as exc:
            logger.warning("%s fallo: %s", self.provider_name, exc)
            return exc.to_response()

    async def stream(
        self,
        messages: MessageLike | Iterable[MessageLike],
        **overrides: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Generador asincrono de chunks.

        Los reintentos solo aplican *antes* del primer token: una vez que
        empezamos a emitir texto no se puede rehacer la llamada sin duplicar
        contenido, asi que a partir de ahi el error se propaga.
        """
        conversation = Conversation.coerce(messages)
        config = self._resolve_config(overrides)

        for intento in range(1, self.retry.max_attempts + 1):
            emitio_algo = False
            try:
                async for chunk in self._astream(conversation, config):
                    emitio_algo = True
                    yield chunk
                return
            except BaseException as exc:  # amplio a proposito: se reclasifica abajo
                error = self._clasificar(exc, config)
                if error is None:
                    raise
                error.attempts = intento
                if emitio_algo or not await self._deberia_reintentar(error, intento):
                    raise error from exc

    async def stream_text(
        self,
        messages: MessageLike | Iterable[MessageLike],
        **overrides: Any,
    ) -> AsyncIterator[str]:
        """Azucar sobre `stream()`: solo los deltas de texto, no vacios."""
        async for chunk in self.stream(messages, **overrides):
            if chunk.delta:
                yield chunk.delta

    async def collect_stream(
        self,
        messages: MessageLike | Iterable[MessageLike],
        **overrides: Any,
    ) -> ModelResponse:
        """Consume el stream y arma una `ModelResponse` equivalente."""
        inicio = time.perf_counter()
        piezas: list[str] = []
        final: StreamChunk | None = None
        async for chunk in self.stream(messages, **overrides):
            piezas.append(chunk.delta)
            if chunk.is_final:
                final = chunk
        config = self._resolve_config(overrides)
        return ModelResponse(
            content="".join(piezas),
            provider=self.provider_name,
            model=final.model if final else config.model,
            usage=final.usage if final and final.usage else Usage(),
            finish_reason=final.finish_reason if final else None,
            latency_ms=(time.perf_counter() - inicio) * 1000,
        )

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _resolve_config(self, overrides: dict[str, Any]) -> ModelConfig:
        """Mezcla la config de la instancia con los overrides de la llamada.

        Se revalida con Pydantic para que un `temperature=9` explote acá y no
        como un 400 despues de un round trip de red.
        """
        if not overrides:
            return self.config
        return ModelConfig.model_validate(self.config.model_dump() | overrides)

    def _clasificar(self, exc: BaseException, config: ModelConfig) -> LLMError | None:
        """Normaliza cualquier excepcion a `LLMError`, o `None` si no aplica."""
        if isinstance(exc, asyncio.CancelledError):
            return None  # nunca capturamos cancelaciones del event loop
        if isinstance(exc, LLMError):
            return exc
        if isinstance(exc, TimeoutError):
            from .exceptions import LLMTimeoutError

            return LLMTimeoutError(
                f"timeout de {config.timeout_s}s",
                provider=self.provider_name,
                model=config.model,
            )
        traducido = self._translate_error(exc)
        if traducido is not None:
            traducido.provider = traducido.provider or self.provider_name
            traducido.model = traducido.model or config.model
        return traducido

    async def _deberia_reintentar(self, error: LLMError, intento: int) -> bool:
        """Duerme el backoff y dice si conviene otro intento."""
        if not error.retryable or intento >= self.retry.max_attempts:
            return False
        espera = min(
            self.retry.initial_backoff_s * self.retry.multiplier ** (intento - 1),
            self.retry.max_backoff_s,
        )
        if error.retry_after_s:  # el proveedor nos dijo cuanto esperar
            espera = min(max(espera, error.retry_after_s), self.retry.max_backoff_s)
        if self.retry.jitter:
            espera *= 0.5 + random.random()  # jitter, no criptografia
        logger.warning(
            "%s: %s (intento %d/%d), reintentando en %.2fs",
            self.provider_name,
            type(error).__name__,
            intento,
            self.retry.max_attempts,
            espera,
        )
        await asyncio.sleep(espera)
        return True

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:  # pragma: no cover - solo presentacion
        return f"{type(self).__name__}(model={self.config.model!r})"
