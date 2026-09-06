"""`AsyncLLMManager`: la fachada que elige proveedor por configuracion.

Es la puerta de entrada recomendada. Envuelve a cualquier `BaseLLMClient` y
resuelve *cual* usar en este orden:

1. el argumento `provider` que le pases;
2. la variable de entorno `LLM_PROVIDER` (configurable con `config_env`).

Si no hay ninguno de los dos falla con un mensaje claro en lugar de adivinar.

    # .env  ->  LLM_PROVIDER=anthropic
    async with AsyncLLMManager() as llm:
        print((await llm.generate("Que es la entropia?")).content)
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterable
from typing import Any, Self

from .base import BaseLLMClient
from .exceptions import ConfigurationError
from .factory import Provider, available_providers, create_client
from .schemas import ErrorResponse, MessageLike, ModelConfig, ModelResponse, StreamChunk

__all__ = ["AsyncLLMManager"]

#: Variable de entorno por defecto que decide el proveedor.
DEFAULT_CONFIG_ENV = "LLM_PROVIDER"


class AsyncLLMManager:
    """Interfaz unica sobre OpenAI, Anthropic o Gemini.

    Expone los mismos metodos que `BaseLLMClient` y delega en el cliente
    concreto, asi que cambiar de proveedor no toca el codigo que la usa:

        async with AsyncLLMManager("openai", temperature=0.3) as llm:
            respuesta = await llm.generate("Hola")            # completo
            async for token in llm.stream_text("Hola"):       # streaming
                print(token, end="")
            resultado = await llm.generate_safe("Hola")        # error como dato
    """

    def __init__(
        self,
        provider: Provider | str | None = None,
        *,
        config_env: str = DEFAULT_CONFIG_ENV,
        **client_kwargs: Any,
    ) -> None:
        elegido = provider or os.getenv(config_env)
        if not elegido:
            opciones = ", ".join(p.value for p in Provider)
            con_key = ", ".join(p.value for p in available_providers()) or "ninguno"
            raise ConfigurationError(
                f"no se indico proveedor: pasalo como argumento o defini {config_env}. "
                f"Opciones: {opciones}. Con API key en el entorno: {con_key}"
            )
        self._config_env = config_env
        self._client: BaseLLMClient = create_client(elegido, **client_kwargs)

    # ------------------------------------------------------------------
    # Estado
    # ------------------------------------------------------------------

    @property
    def provider(self) -> str:
        """Nombre del proveedor activo."""
        return self._client.provider_name

    @property
    def model(self) -> str:
        """Modelo activo."""
        return self._client.config.model

    @property
    def config(self) -> ModelConfig:
        """Configuracion de inferencia vigente."""
        return self._client.config

    @property
    def client(self) -> BaseLLMClient:
        """El cliente concreto, si hace falta algo especifico del proveedor."""
        return self._client

    async def switch(self, provider: Provider | str, **client_kwargs: Any) -> Self:
        """Cambia de proveedor en caliente, cerrando el anterior."""
        anterior = self._client
        self._client = create_client(provider, **client_kwargs)
        await anterior.aclose()
        return self

    # ------------------------------------------------------------------
    # Generacion (delega en el cliente concreto)
    # ------------------------------------------------------------------

    async def generate(
        self, messages: MessageLike | Iterable[MessageLike], **overrides: Any
    ) -> ModelResponse:
        """Respuesta completa. Lanza `LLMError` si se agotan los reintentos."""
        return await self._client.generate(messages, **overrides)

    async def generate_safe(
        self, messages: MessageLike | Iterable[MessageLike], **overrides: Any
    ) -> ModelResponse | ErrorResponse:
        """Igual, pero devuelve el error como dato en vez de lanzarlo."""
        return await self._client.generate_safe(messages, **overrides)

    def stream(
        self, messages: MessageLike | Iterable[MessageLike], **overrides: Any
    ) -> AsyncIterator[StreamChunk]:
        """Generador asincrono de `StreamChunk` (el ultimo trae el `usage`)."""
        return self._client.stream(messages, **overrides)

    def stream_text(
        self, messages: MessageLike | Iterable[MessageLike], **overrides: Any
    ) -> AsyncIterator[str]:
        """Generador asincrono de fragmentos de texto."""
        return self._client.stream_text(messages, **overrides)

    async def collect_stream(
        self, messages: MessageLike | Iterable[MessageLike], **overrides: Any
    ) -> ModelResponse:
        """Consume el stream y devuelve una `ModelResponse` equivalente."""
        return await self._client.collect_stream(messages, **overrides)

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Cierra el cliente HTTP del proveedor."""
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:  # pragma: no cover - solo presentacion
        return f"AsyncLLMManager(provider={self.provider!r}, model={self.model!r})"
