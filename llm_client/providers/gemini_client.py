"""Cliente Gemini sobre `google-genai` (SDK oficial, interfaz `client.aio`)."""

from __future__ import annotations

import os
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
from ..schemas import Conversation, ModelConfig, ModelResponse, Role, StreamChunk, Usage

__all__ = ["GeminiClient"]

#: Gemini llama "model" al rol que OpenAI/Anthropic llaman "assistant".
_ROLES = {Role.USER: "user", Role.ASSISTANT: "model"}


class GeminiClient(BaseLLMClient):
    """Usa `client.aio`, la fachada asincrona del SDK.

    Ojo: `genai.Client().models.generate_content()` (sin `.aio`) es sincrono y
    bloquearia el event loop; siempre hay que pasar por `client.aio`.
    """

    provider_name: ClassVar[str] = "gemini"
    # gemini-2.5-flash ya no se habilita para keys nuevas; la propia API
    # redirige a 3.6-flash (tambien hay 3.7-flash y 3.8-flash disponibles).
    default_model: ClassVar[str] = "gemini-3.6-flash"
    api_key_env: ClassVar[str] = "GEMINI_API_KEY"

    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None, **kwargs: Any
    ) -> None:
        # El SDK acepta las dos variables historicas; respetamos ambas.
        api_key = api_key or os.getenv(self.api_key_env) or os.getenv("GOOGLE_API_KEY")
        super().__init__(api_key=api_key, **kwargs)
        try:
            from google import genai
            from google.genai import errors as genai_errors
        except ModuleNotFoundError as exc:  # pragma: no cover - depende del entorno
            raise ProviderNotInstalledError(
                "falta el SDK de Gemini: pip install google-genai",
                provider=self.provider_name,
            ) from exc

        self._errors = genai_errors
        self._client = genai.Client(
            api_key=self._api_key,
            http_options={"base_url": base_url} if base_url else None,
        )

    # ------------------------------------------------------------------

    def _payload(self, conversation: Conversation, config: ModelConfig) -> dict[str, Any]:
        """Arma `contents` y `config` con dicts planos (el SDK los valida)."""
        generation: dict[str, Any] = {
            "max_output_tokens": config.max_tokens,
            # El timeout del SDK va en milisegundos.
            "http_options": {"timeout": int(config.timeout_s * 1000)},
            # No usamos function calling: desactivarlo evita el warning del SDK.
            "automatic_function_calling": {"disable": True},
        }
        if (system := conversation.system_prompt) is not None:
            generation["system_instruction"] = system
        if config.temperature is not None:
            generation["temperature"] = config.temperature
        if config.top_p is not None:
            generation["top_p"] = config.top_p
        if config.stop:
            generation["stop_sequences"] = list(config.stop)
        generation.update(config.extra)

        return {
            "model": config.model,
            "contents": [
                {"role": _ROLES[m.role], "parts": [{"text": m.content}]} for m in conversation.turns
            ],
            "config": generation,
        }

    async def _agenerate(self, conversation: Conversation, config: ModelConfig) -> ModelResponse:
        respuesta = await self._client.aio.models.generate_content(
            **self._payload(conversation, config)
        )
        return ModelResponse(
            content=respuesta.text or "",
            provider=self.provider_name,
            model=config.model,
            usage=_usage(respuesta.usage_metadata),
            finish_reason=_finish_reason(respuesta),
            raw=respuesta.model_dump(mode="json", exclude_none=True),
        )

    async def _astream(
        self, conversation: Conversation, config: ModelConfig
    ) -> AsyncIterator[StreamChunk]:
        # En la fachada asincrona, `generate_content_stream` devuelve un
        # awaitable que resuelve al iterador: de ahi el `await` interno.
        stream = await self._client.aio.models.generate_content_stream(
            **self._payload(conversation, config)
        )

        indice = 0
        usage = Usage()
        finish_reason: str | None = None

        async for evento in stream:
            if evento.usage_metadata is not None:
                usage = _usage(evento.usage_metadata)
            if (motivo := _finish_reason(evento)) is not None:
                finish_reason = motivo
            if texto := (evento.text or ""):
                yield StreamChunk(
                    delta=texto,
                    index=indice,
                    provider=self.provider_name,
                    model=config.model,
                )
                indice += 1

        yield StreamChunk(
            index=indice,
            provider=self.provider_name,
            model=config.model,
            is_final=True,
            usage=usage,
            finish_reason=finish_reason,
        )

    # ------------------------------------------------------------------

    def _translate_error(self, exc: BaseException) -> LLMError | None:
        errores = self._errors
        detalle = str(exc)

        if isinstance(exc, errores.APIError):
            status = getattr(exc, "code", None)
            if status in (401, 403):
                return AuthenticationError(detalle, status_code=status)
            # Gemini reporta la key invalida como 400 API_KEY_INVALID: sin este
            # caso quedaria como InvalidRequestError y confundiria al usuario.
            if status == 400 and "api_key_invalid" in detalle.lower().replace(" ", "_"):
                return AuthenticationError(detalle, status_code=status)
            if status == 429:
                return RateLimitError(detalle, status_code=status)
            if status in (408, 504):
                return LLMTimeoutError(detalle, status_code=status)
            if isinstance(exc, errores.ServerError) or (status is not None and status >= 500):
                return ServiceUnavailableError(detalle, status_code=status)
            return InvalidRequestError(detalle, status_code=status)
        # httpx viaja dentro de google-genai; sus fallos de red son transitorios.
        if type(exc).__module__.startswith("httpx"):
            if "timeout" in type(exc).__name__.lower():
                return LLMTimeoutError(detalle)
            return LLMConnectionError(detalle)
        return None


def _usage(bruto: Any) -> Usage:
    if bruto is None:
        return Usage()
    return Usage(
        input_tokens=getattr(bruto, "prompt_token_count", 0) or 0,
        output_tokens=getattr(bruto, "candidates_token_count", 0) or 0,
    )


def _finish_reason(respuesta: Any) -> str | None:
    """`FinishReason.STOP` (enum) -> `"STOP"`."""
    candidatos = getattr(respuesta, "candidates", None) or []
    if not candidatos:
        return None
    motivo = getattr(candidatos[0], "finish_reason", None)
    if motivo is None:
        return None
    return getattr(motivo, "name", None) or str(motivo)
