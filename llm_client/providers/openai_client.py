"""Cliente OpenAI sobre `AsyncOpenAI` (SDK oficial, variante asincrona)."""

from __future__ import annotations

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

__all__ = ["OpenAIClient"]


class OpenAIClient(BaseLLMClient):
    """Chat Completions con `await`: nunca bloquea el event loop."""

    provider_name: ClassVar[str] = "openai"
    default_model: ClassVar[str] = "gpt-4.1-mini"
    api_key_env: ClassVar[str] = "OPENAI_API_KEY"

    def __init__(self, *, base_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            import openai
        except ModuleNotFoundError as exc:  # pragma: no cover - depende del entorno
            raise ProviderNotInstalledError(
                "falta el SDK de OpenAI: pip install openai",
                provider=self.provider_name,
            ) from exc

        self._sdk = openai
        # `max_retries=0`: los reintentos los maneja BaseLLMClient, si los deja
        # el SDK tambien terminan siendo intentos duplicados e invisibles.
        self._client = openai.AsyncOpenAI(
            api_key=self._api_key,
            base_url=base_url,
            timeout=self.config.timeout_s,
            max_retries=0,
        )

    async def aclose(self) -> None:
        await self._client.close()

    # ------------------------------------------------------------------

    def _payload(self, conversation: Conversation, config: ModelConfig) -> dict[str, Any]:
        """Traduce el contrato interno al formato de Chat Completions."""
        payload: dict[str, Any] = {
            "model": config.model,
            # OpenAI acepta el rol `system` en la misma lista de mensajes.
            "messages": [
                {"role": m.role.value, "content": m.content} for m in conversation.messages
            ],
            "max_completion_tokens": config.max_tokens,
        }
        if config.temperature is not None:
            payload["temperature"] = config.temperature
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        if config.stop:
            payload["stop"] = list(config.stop)
        payload.update(config.extra)
        return payload

    async def _agenerate(self, conversation: Conversation, config: ModelConfig) -> ModelResponse:
        respuesta = await self._client.chat.completions.create(
            **self._payload(conversation, config),
            timeout=config.timeout_s,
        )
        eleccion = respuesta.choices[0]
        return ModelResponse(
            content=eleccion.message.content or "",
            provider=self.provider_name,
            model=respuesta.model,
            usage=_usage(respuesta.usage),
            finish_reason=eleccion.finish_reason,
            raw=respuesta.model_dump(),
        )

    async def _astream(
        self, conversation: Conversation, config: ModelConfig
    ) -> AsyncIterator[StreamChunk]:
        stream = await self._client.chat.completions.create(
            **self._payload(conversation, config),
            stream=True,
            # Sin esto, el ultimo evento del stream no trae tokens consumidos.
            stream_options={"include_usage": True},
            timeout=config.timeout_s,
        )

        indice = 0
        modelo = config.model
        usage = Usage()
        finish_reason: str | None = None

        async for evento in stream:
            modelo = evento.model or modelo
            if evento.usage is not None:
                usage = _usage(evento.usage)
            for eleccion in evento.choices:
                if eleccion.finish_reason:
                    finish_reason = eleccion.finish_reason
                delta = eleccion.delta.content if eleccion.delta else None
                if delta:
                    yield StreamChunk(
                        delta=delta,
                        index=indice,
                        provider=self.provider_name,
                        model=modelo,
                    )
                    indice += 1

        yield StreamChunk(
            index=indice,
            provider=self.provider_name,
            model=modelo,
            is_final=True,
            usage=usage,
            finish_reason=finish_reason,
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
        if isinstance(exc, sdk.InternalServerError):
            return ServiceUnavailableError(detalle, status_code=status)
        if isinstance(exc, sdk.APIStatusError):
            if status is not None and status >= 500:
                return ServiceUnavailableError(detalle, status_code=status)
            return LLMError(detalle, status_code=status)
        if isinstance(exc, sdk.OpenAIError):
            return LLMError(detalle)
        return None


def _usage(bruto: Any) -> Usage:
    """`CompletionUsage` -> `Usage`, tolerando que venga en `None`."""
    if bruto is None:
        return Usage()
    return Usage(
        input_tokens=getattr(bruto, "prompt_tokens", 0) or 0,
        output_tokens=getattr(bruto, "completion_tokens", 0) or 0,
    )


def _retry_after(exc: BaseException) -> float | None:
    """Lee el header `retry-after` si el proveedor lo mando."""
    respuesta = getattr(exc, "response", None)
    valor = getattr(respuesta, "headers", {}).get("retry-after") if respuesta else None
    try:
        return float(valor) if valor is not None else None
    except (TypeError, ValueError):
        return None
