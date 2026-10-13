"""Factories de modelos: chat (supervisor y especialistas) y embeddings (busqueda vectorial)."""

from __future__ import annotations

import os
import warnings
from importlib import import_module
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from app.config import EMBEDDING_DIMENSION

__all__ = ["crear_embeddings", "crear_modelo"]

# Los modelos "flash-lite" ignoran `temperature` y langchain avisa en cada llamada: ruido sin valor.
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")

#: proveedor -> (paquete, clase, envs de la key, modelo por defecto).
_CHAT: dict[str, tuple[str, str, tuple[str, ...], str]] = {
    "openai": ("langchain_openai", "ChatOpenAI", ("OPENAI_API_KEY",), "gpt-4.1-mini"),
    "gemini": (
        "langchain_google_genai",
        "ChatGoogleGenerativeAI",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "gemini-3.6-flash",
    ),
}
#: proveedor -> (paquete, clase, nombre del parametro de dimension, modelo por defecto).
_EMBEDDINGS: dict[str, tuple[str, str, str, str]] = {
    "openai": ("langchain_openai", "OpenAIEmbeddings", "dimensions", "text-embedding-3-small"),
    "gemini": (
        "langchain_google_genai",
        "GoogleGenerativeAIEmbeddings",
        "output_dimensionality",
        "models/gemini-embedding-001",
    ),
}

#: Un solo rate limiter para TODO el proceso: supervisor y especialistas comparten la cuota
#: del proveedor, asi que tienen que compartir tambien el espaciado de requests.
_limiter: InMemoryRateLimiter | None = None


def _proveedor() -> str:
    pedido = (os.getenv("LLM_PROVIDER") or "").lower()
    if not pedido:
        pedido = next((n for n, (_, _, envs, _) in _CHAT.items() if any(os.getenv(e) for e in envs)), "")
    if pedido not in _CHAT:
        raise ValueError(f"proveedor desconocido: {pedido!r}. Opciones: {', '.join(_CHAT)}")
    if not any(os.getenv(e) for e in _CHAT[pedido][2]):
        raise RuntimeError(f"falta la API key de {pedido}: definila en .env (ver .env.example)")
    return pedido


def crear_modelo(*, modelo: str | None = None, **kwargs: Any) -> BaseChatModel:
    """Chat model del proveedor de `LLM_PROVIDER` (o el primero con key).

    `temperature=0`: un supervisor que enruta tiene que decidir de forma reproducible.
    Si `LLM_RPM` esta definida, se espacian las llamadas (free tier de Gemini: ~5 por minuto).
    """
    global _limiter
    paquete, clase, _, por_defecto = _CHAT[_proveedor()]
    kwargs.setdefault("temperature", 0.0)
    kwargs.setdefault("max_retries", 2)
    if rpm := os.getenv("LLM_RPM"):
        _limiter = _limiter or InMemoryRateLimiter(requests_per_second=float(rpm) / 60)
        kwargs.setdefault("rate_limiter", _limiter)
    cls = getattr(import_module(paquete), clase)
    return cls(model=modelo or os.getenv("LLM_MODEL") or por_defecto, **kwargs)


def crear_embeddings() -> Embeddings:
    """Embeddings del mismo proveedor, con la dimension del indice de Pinecone."""
    paquete, clase, param_dim, por_defecto = _EMBEDDINGS[_proveedor()]
    cls = getattr(import_module(paquete), clase)
    return cls(model=os.getenv("EMBEDDING_MODEL") or por_defecto, **{param_dim: EMBEDDING_DIMENSION})
