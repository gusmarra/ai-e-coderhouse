"""Factory de modelos: mismo patron que las entregas anteriores (`crear_modelo`
del Modulo 1 y 2), con dos familias de modelo: chat (generacion) y
embeddings (indexado de `data/` + consulta del usuario).

Pre-entrega 4: los embeddings salen siempre con `EMBEDDING_DIMENSION`
(1536) dimensiones, que es lo que espera el indice de Pinecone.

El punto critico de un RAG es que el modelo de embeddings que indexa los
documentos tiene que ser exactamente el mismo que el que embebe la pregunta
en tiempo de consulta: son espacios vectoriales distintos entre proveedores
(y a veces entre modelos del mismo proveedor), asi que mezclarlos no tira un
error — tira resultados que parecen aleatorios (la distancia coseno entre un
vector de OpenAI y uno de Gemini no significa nada). `crear_embeddings()`
resuelve el proveedor con la misma logica que `crear_modelo()`, y tanto
`ingest.py` como `rag.py` la llaman sin argumentos: mientras ambos lean
`LLM_PROVIDER` del mismo `.env`, no pueden divergir por accidente.

Solo se soportan `openai` y `gemini`: Anthropic no expone una API de
embeddings, asi que no tiene sentido incluirlo en `_PROVEEDORES_EMBEDDINGS`
(y por consistencia, tampoco en el chat de este modulo).
"""

from __future__ import annotations

import os
from functools import lru_cache
from importlib import import_module
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from config import EMBEDDING_DIMENSION

__all__ = [
    "EmbeddingsConCache",
    "crear_embeddings",
    "embeddings_por_defecto",
    "crear_modelo",
    "proveedores_disponibles",
]

#: proveedor -> (paquete, clase, envs de la key, modelo por defecto).
_PROVEEDORES_CHAT: dict[str, tuple[str, str, tuple[str, ...], str]] = {
    "openai": ("langchain_openai", "ChatOpenAI", ("OPENAI_API_KEY",), "gpt-4.1-mini"),
    "gemini": (
        "langchain_google_genai",
        "ChatGoogleGenerativeAI",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "gemini-3.6-flash",
    ),
}

_PROVEEDORES_EMBEDDINGS: dict[str, tuple[str, str, tuple[str, ...], str]] = {
    "openai": ("langchain_openai", "OpenAIEmbeddings", ("OPENAI_API_KEY",), "text-embedding-3-small"),
    "gemini": (
        "langchain_google_genai",
        "GoogleGenerativeAIEmbeddings",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "models/gemini-embedding-001",
    ),
}

#: Nombre del parametro con el que cada proveedor acepta la dimension de salida.
_PARAM_DIMENSION = {"openai": "dimensions", "gemini": "output_dimensionality"}

#: alias: LangChain llama "google" a lo que aca (y en el Modulo 1) se llama "gemini".
_ALIAS = {"google": "gemini", "google_genai": "gemini", "openai_chat": "openai"}


def proveedores_disponibles() -> list[str]:
    """Los proveedores que tienen su API key en el entorno (validos para chat y embeddings)."""
    return [
        nombre
        for nombre, (_, _, envs, _) in _PROVEEDORES_CHAT.items()
        if any(os.getenv(env) for env in envs)
    ]


def _resolver_proveedor(proveedor: str | None, tabla: dict[str, tuple[str, str, tuple[str, ...], str]]) -> str:
    pedido = proveedor or os.getenv("LLM_PROVIDER") or next(iter(proveedores_disponibles()), "")
    elegido = _ALIAS.get(pedido.lower(), pedido.lower())
    if elegido not in tabla:
        validos = ", ".join(tabla)
        con_key = ", ".join(proveedores_disponibles()) or "ninguno"
        raise ValueError(
            f"proveedor desconocido: {pedido!r}. Opciones: {validos}. "
            f"Con API key en el entorno: {con_key}"
        )
    _, _, envs, _ = tabla[elegido]
    if not any(os.getenv(env) for env in envs):
        raise RuntimeError(f"falta la API key de {elegido}: defini {' o '.join(envs)} en .env")
    return elegido


def crear_modelo(
    proveedor: str | None = None,
    *,
    modelo: str | None = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> BaseChatModel:
    """Instancia el chat model del proveedor pedido (arg, luego `LLM_PROVIDER`, luego el primero con key).

    `temperature=0` por defecto: la generacion grounded busca la respuesta mas
    fiel al contexto, no la mas creativa.
    """
    elegido = _resolver_proveedor(proveedor, _PROVEEDORES_CHAT)
    paquete, clase, _, por_defecto = _PROVEEDORES_CHAT[elegido]
    cls = getattr(import_module(paquete), clase)
    nombre_modelo = modelo or os.getenv("LLM_MODEL") or por_defecto
    return cls(model=nombre_modelo, temperature=temperature, **kwargs)


def crear_embeddings(
    proveedor: str | None = None,
    *,
    modelo: str | None = None,
    **kwargs: Any,
) -> Embeddings:
    """Instancia el modelo de embeddings del mismo proveedor que `crear_modelo()`.

    Resuelve el proveedor con la misma prioridad (arg, `LLM_PROVIDER`, primero
    con key) para que indexar y consultar usen, por construccion, el mismo
    espacio vectorial.
    """
    elegido = _resolver_proveedor(proveedor, _PROVEEDORES_EMBEDDINGS)
    paquete, clase, _, por_defecto = _PROVEEDORES_EMBEDDINGS[elegido]
    cls = getattr(import_module(paquete), clase)
    nombre_modelo = modelo or os.getenv("EMBEDDING_MODEL") or por_defecto
    kwargs.setdefault(_PARAM_DIMENSION[elegido], EMBEDDING_DIMENSION)
    return cls(model=nombre_modelo, **kwargs)


class EmbeddingsConCache(Embeddings):
    """Memoiza `embed_query`: la misma pregunta no se embebe dos veces.

    Importa en `evaluate.py`, que consulta cada pregunta con el recuperador
    vectorial y con el hibrido: sin cache seria el doble de llamadas a la API.
    """

    def __init__(self, base: Embeddings) -> None:
        self._base = base
        self._cache: dict[str, list[float]] = {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._base.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if text not in self._cache:
            self._cache[text] = self._base.embed_query(text)
        return self._cache[text]


@lru_cache(maxsize=1)
def embeddings_por_defecto() -> Embeddings:
    """Instancia compartida del proceso: evita recrear el cliente en cada llamada."""
    return EmbeddingsConCache(crear_embeddings())
