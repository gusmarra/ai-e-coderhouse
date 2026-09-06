"""Cliente Anthropic sobre `AsyncAnthropic` (SDK oficial, variante asincrona)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, ClassVar

from ..base import BaseLLMClient
from ..exceptions import (
    AuthenticationError,
    InvalidRequestError,
    LLMConnectionError,
    LLMError,
    LLMTimeoutError,
    ProviderNotInstalledError,
    RateLimitError,
    ServiceUnavailableError,
)
from ..schemas import Conversation, ModelConfig, ModelResponse, StreamChunk, Usage

logger = logging.getLogger(__name__)

__all__ = ["AnthropicClient"]


class AnthropicClient(BaseLLMClient):
    """Messages API con `await`.

    Tres diferencias de forma respecto de OpenAI que la clase absorbe:

    * el prompt de sistema va en el parametro `system`, no en `messages`;
    * los roles deben alternar, asi que usamos `merged_turns()`;
    * los modelos actuales (familia Claude 5 / 4.6+) ya no aceptan
      `temperature` ni `top_p`: la profundidad de razonamiento se controla
      con `output_config={"effort": ...}`. Si vienen seteados se ignoran
      con un warning en lugar de romper la llamada.

    Para pasar parametros propios de la API se usa `extra`::

        AnthropicClient(extra={"output_config": {"effort": "low"}})
    """

    provider_name: ClassVar[str] = "anthropic"
    default_model: ClassVar[str] = "claude-opus-5"
    api_key_env: ClassVar[str] = "ANTHROPIC_API_KEY"
    #: Parametros de sampling que la Messages API actual ya no soporta.
    _sampling_no_soportado: ClassVar[tuple[str, ...]] = ("temperature", "top_p")

    def __init__(self, *, base_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            import anthropic
        except ModuleNotFoundError as exc:  # pragma: no cover - depende del entorno
            raise ProviderNotInstalledError(
                "falta el SDK de Anthropic: pip install anthropic",
                provider=self.provider_name,
            ) from exc

        self._sdk = anthropic
        self._aviso_sampling = False
        self._client = anthropic.AsyncAnthropic(
            api_key=self._api_key,
            base_url=base_url,
            timeout=self.config.timeout_s,
            max_retries=0,  # los reintentos los maneja BaseLLMClient
        )

    async def aclose(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------------

    def _payload(self, conversation: Conversation, config: ModelConfig) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens,  # obligatorio en la Messages API
            "messages": [
                {"role": m.role.value, "content": m.content} for m in conversation.merged_turns()
            ],
        }
        if (system := conversation.system_prompt) is not None:
            payload["system"] = system
        if config.stop:
            payload["stop_sequences"] = list(config.stop)
        self._avisar_sampling(config)
        payload.update(config.extra)
        return payload

    def _avisar_sampling(self, config: ModelConfig) -> None:
        """Avisa una sola vez si se pidieron parametros que la API rechaza."""
        if self._aviso_sampling:
            return
        ignorados = [
            nombre for nombre in self._sampling_no_soportado if getattr(config, nombre) is not None
        ]
        if ignorados:
            self._aviso_sampling = True
            logger.warning(
                "anthropic: %s no se envia (la Messages API actual lo rechaza); "
                'usa extra={"output_config": {"effort": "low"|"medium"|"high"}}',
                "/".join(ignorados),
            )

    async def _agenerate(self, conversation: Conversation, config: ModelConfig) -> ModelResponse:
        mensaje = await self._client.messages.create(
            **self._payload(conversation, config),
            timeout=config.timeout_s,
        )
        return ModelResponse(
            content=_texto(mensaje),
            provider=self.provider_name,
            model=mensaje.model,
            usage=_usage(mensaje.usage),
            finish_reason=mensaje.stop_reason,
            raw=mensaje.model_dump(),
        )

    async def _astream(
        self, conversation: Conversation, config: ModelConfig
    ) -> AsyncIterator[StreamChunk]:
        indice = 0
        # `messages.stream` es un context manager asincrono (no se hace await):
        # garantiza que la conexion se cierre incluso si el consumidor corta.
        async with self._client.messages.stream(
            **self._payload(conversation, config),
            timeout=config.timeout_s,
        ) as stream:
            async for texto in stream.text_stream:
                if texto:
                    yield StreamChunk(
                        delta=texto,
                        index=indice,
                        provider=self.provider_name,
                        model=config.model,
                    )
                    indice += 1

            final = await stream.get_final_message()

        yield StreamChunk(
            index=indice,
            provider=self.provider_name,
            model=final.model,
            is_final=True,
            usage=_usage(final.usage),
            finish_reason=final.stop_reason,
        )

    # ------------------------------------------------------------------

    def _translate_error(self, exc: BaseException) -> LLMError | None:
        sdk = self._sdk
        detalle = str(exc)
        status = getattr(exc, "status_code", None)

        if isinstance(exc, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
            return AuthenticationError(detalle, status_code=status)
        if isinstance(exc, sdk.RateLimitError):
            return RateLimitError(detalle, status_code=status, retry_after_s=_retry_after(exc))
        if isinstance(exc, sdk.APITimeoutError):
            return LLMTimeoutError(detalle, status_code=status)
        if isinstance(exc, sdk.APIConnectionError):
            return LLMConnectionError(detalle)
        if isinstance(exc, (sdk.BadRequestError, sdk.NotFoundError, sdk.UnprocessableEntityError)):
            return InvalidRequestError(detalle, status_code=status)
        if isinstance(exc, (sdk.InternalServerError, sdk.APIStatusError)):
            if status is None or status >= 500 or status == 529:  # 529 = overloaded
                return ServiceUnavailableError(detalle, status_code=status)
            return LLMError(detalle, status_code=status)
        if isinstance(exc, sdk.AnthropicError):
            return LLMError(detalle)
        return None


def _texto(mensaje: Any) -> str:
    """Concatena los bloques de texto; ignora bloques de otro tipo."""
    return "".join(
        bloque.text for bloque in mensaje.content if getattr(bloque, "type", None) == "text"
    )


def _usage(bruto: Any) -> Usage:
    if bruto is None:
        return Usage()
    return Usage(
        input_tokens=getattr(bruto, "input_tokens", 0) or 0,
        output_tokens=getattr(bruto, "output_tokens", 0) or 0,
    )


def _retry_after(exc: BaseException) -> float | None:
    respuesta = getattr(exc, "response", None)
    valor = getattr(respuesta, "headers", {}).get("retry-after") if respuesta else None
    try:
        return float(valor) if valor is not None else None
    except (TypeError, ValueError):
        return None
