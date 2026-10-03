"""Factory del chat model: mismo patron que `crear_modelo` de las entregas anteriores."""

from __future__ import annotations

import os
import warnings
from importlib import import_module
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

import config  # noqa: F401  (carga el .env antes de leer las keys)

__all__ = ["crear_modelo"]

# Los modelos "flash-lite" ignoran `temperature` y langchain avisa en cada llamada: ruido sin valor.
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")

#: proveedor -> (paquete, clase, envs de la key, modelo por defecto).
_PROVEEDORES: dict[str, tuple[str, str, tuple[str, ...], str]] = {
    "openai": ("langchain_openai", "ChatOpenAI", ("OPENAI_API_KEY",), "gpt-4.1-mini"),
    "gemini": (
        "langchain_google_genai",
        "ChatGoogleGenerativeAI",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "gemini-3.6-flash",
    ),
}


def _con_key(nombre: str) -> bool:
    return any(os.getenv(env) for env in _PROVEEDORES[nombre][2])


def crear_modelo(proveedor: str | None = None, *, modelo: str | None = None, **kwargs: Any) -> BaseChatModel:
    """Instancia el chat model (arg, luego `LLM_PROVIDER`, luego el primero con key).

    `temperature=0`: para un agente que decide herramientas queremos decisiones
    reproducibles, no creativas. `max_retries=2` evita que un 429 gaste cuota reintentando.
    Si `LLM_RPM` esta definida, un rate limiter espacia las llamadas para no pasar ese tope
    (el free tier de Gemini corta a ~5 requests por minuto y un agente ReAct hace varias por pregunta).
    """
    pedido = (proveedor or os.getenv("LLM_PROVIDER") or "").lower()
    if not pedido:
        pedido = next((n for n in _PROVEEDORES if _con_key(n)), "")
    if pedido not in _PROVEEDORES:
        raise ValueError(f"proveedor desconocido: {pedido!r}. Opciones: {', '.join(_PROVEEDORES)}")
    if not _con_key(pedido):
        raise RuntimeError(f"falta la API key de {pedido}: definila en .env (ver .env.example)")
    paquete, clase, _, por_defecto = _PROVEEDORES[pedido]
    cls = getattr(import_module(paquete), clase)
    kwargs.setdefault("temperature", 0.0)
    kwargs.setdefault("max_retries", 2)
    if rpm := os.getenv("LLM_RPM"):
        kwargs.setdefault("rate_limiter", InMemoryRateLimiter(requests_per_second=float(rpm) / 60))
    return cls(model=modelo or os.getenv("LLM_MODEL") or por_defecto, **kwargs)
