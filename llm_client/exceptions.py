"""Jerarquia de excepciones propia del cliente.

Objetivo: que la aplicacion nunca tenga que importar `openai.RateLimitError`
ni `anthropic.APIStatusError`. Cada proveedor traduce sus errores nativos a
estas clases, y el flag `retryable` es lo unico que la logica de reintentos
necesita mirar.
"""

from __future__ import annotations

from .schemas import ErrorResponse

__all__ = [
    "AuthenticationError",
    "ConfigurationError",
    "InvalidRequestError",
    "LLMConnectionError",
    "LLMError",
    "LLMTimeoutError",
    "ProviderNotInstalledError",
    "RateLimitError",
    "ServiceUnavailableError",
    "UnknownProviderError",
]


class LLMError(Exception):
    """Base de todos los errores del cliente."""

    #: Si es True, la capa de reintentos vuelve a intentar con backoff.
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        retry_after_s: float | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.retry_after_s = retry_after_s
        #: Cuantos intentos se hicieron antes de rendirse (lo setea el cliente).
        self.attempts = attempts

    def to_response(self, *, attempts: int | None = None) -> ErrorResponse:
        """Version dato de la excepcion, para devolver en lugar de propagar."""
        return ErrorResponse(
            provider=self.provider or "unknown",
            model=self.model,
            error_type=type(self).__name__,
            message=self.message,
            retryable=self.retryable,
            status_code=self.status_code,
            attempts=attempts or self.attempts,
        )

    def __str__(self) -> str:  # pragma: no cover - solo presentacion
        partes = [self.message]
        if self.provider:
            partes.append(f"provider={self.provider}")
        if self.model:
            partes.append(f"model={self.model}")
        if self.status_code:
            partes.append(f"status={self.status_code}")
        return " | ".join(partes)


class ConfigurationError(LLMError):
    """Falta una API key, el modelo no existe en el mapa, etc."""


class UnknownProviderError(ConfigurationError):
    """El nombre pedido a la factory no corresponde a ningun proveedor."""


class ProviderNotInstalledError(ConfigurationError):
    """El SDK oficial del proveedor no esta instalado en el entorno."""


class AuthenticationError(LLMError):
    """401/403: API key invalida o sin permisos. Reintentar no sirve."""


class InvalidRequestError(LLMError):
    """400/422: el request esta mal armado (modelo inexistente, contexto excedido)."""


class RateLimitError(LLMError):
    """429: cuota o rate limit. Transitorio, se reintenta con backoff."""

    retryable = True


class ServiceUnavailableError(LLMError):
    """5xx del proveedor. Transitorio."""

    retryable = True


class LLMConnectionError(LLMError):
    """Fallo de red antes de obtener respuesta."""

    retryable = True


class LLMTimeoutError(LLMError):
    """Se agoto `timeout_s`."""

    retryable = True
