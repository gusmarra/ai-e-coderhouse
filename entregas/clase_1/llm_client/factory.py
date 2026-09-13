"""Factory: elegir proveedor por string, sin importar los tres SDKs.

Los imports son perezosos a proposito: tener `openai` instalado no deberia
ser requisito para usar Anthropic.
"""

from __future__ import annotations

import os
from enum import StrEnum
from typing import Any

from .base import BaseLLMClient
from .exceptions import UnknownProviderError

__all__ = ["Provider", "available_providers", "create_client"]


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


#: proveedor -> (modulo, clase, variables de entorno aceptadas)
_REGISTRY: dict[Provider, tuple[str, str, tuple[str, ...]]] = {
    Provider.OPENAI: ("openai_client", "OpenAIClient", ("OPENAI_API_KEY",)),
    Provider.ANTHROPIC: ("anthropic_client", "AnthropicClient", ("ANTHROPIC_API_KEY",)),
    Provider.GEMINI: ("gemini_client", "GeminiClient", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
}


def _resolve(provider: Provider | str) -> Provider:
    try:
        return Provider(str(provider).lower())
    except ValueError as exc:
        validos = ", ".join(p.value for p in Provider)
        raise UnknownProviderError(
            f"proveedor desconocido: {provider!r}. Opciones: {validos}"
        ) from exc


def create_client(provider: Provider | str, **kwargs: Any) -> BaseLLMClient:
    """Devuelve el cliente del proveedor pedido, ya bajo la interfaz comun.

    Ejemplo::

        client = create_client("anthropic", model="claude-sonnet-5", temperature=0.2)
    """
    from importlib import import_module

    elegido = _resolve(provider)
    modulo, clase, _ = _REGISTRY[elegido]
    cls = getattr(import_module(f".providers.{modulo}", __package__), clase)
    return cls(**kwargs)


def available_providers() -> list[Provider]:
    """Los proveedores que tienen su API key en el entorno."""
    return [
        proveedor
        for proveedor, (_, _, envs) in _REGISTRY.items()
        if any(os.getenv(env) for env in envs)
    ]
