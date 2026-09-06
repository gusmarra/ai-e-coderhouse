"""Cliente LLM asincrono, multi-proveedor y validado con Pydantic.

Los clientes concretos se exponen de forma perezosa (`__getattr__`) para que
importar el paquete no exija tener los tres SDKs instalados.

    from llm_client import ChatMessage, create_client

    async with create_client("openai") as client:
        print((await client.generate("Que es la entropia?")).content)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import BaseLLMClient
from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    InvalidRequestError,
    LLMConnectionError,
    LLMError,
    LLMTimeoutError,
    ProviderNotInstalledError,
    RateLimitError,
    ServiceUnavailableError,
    UnknownProviderError,
)
from .factory import Provider, available_providers, create_client
from .schemas import (
    ChatMessage,
    Conversation,
    ErrorResponse,
    ModelConfig,
    ModelResponse,
    RetryConfig,
    Role,
    StreamChunk,
    Usage,
)

if TYPE_CHECKING:  # pragma: no cover
    from .providers.anthropic_client import AnthropicClient
    from .providers.gemini_client import GeminiClient
    from .providers.openai_client import OpenAIClient

__version__ = "1.0.0"

__all__ = [
    "AnthropicClient",
    "AuthenticationError",
    "BaseLLMClient",
    "ChatMessage",
    "ConfigurationError",
    "Conversation",
    "ErrorResponse",
    "GeminiClient",
    "InvalidRequestError",
    "LLMConnectionError",
    "LLMError",
    "LLMTimeoutError",
    "ModelConfig",
    "ModelResponse",
    "OpenAIClient",
    "Provider",
    "ProviderNotInstalledError",
    "RateLimitError",
    "RetryConfig",
    "Role",
    "ServiceUnavailableError",
    "StreamChunk",
    "UnknownProviderError",
    "Usage",
    "available_providers",
    "create_client",
]

_LAZY = {
    "OpenAIClient": ".providers.openai_client",
    "AnthropicClient": ".providers.anthropic_client",
    "GeminiClient": ".providers.gemini_client",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
