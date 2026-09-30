"""Dobles de prueba: permiten correr todo sin red ni API keys."""

from __future__ import annotations

import math
import re
from types import SimpleNamespace
from typing import Any

from langchain_core.embeddings import Embeddings

DIMENSION = 1536


class EmbeddingsFalsos(Embeddings):
    """Embedding determinista: hashea cada palabra a una dimension (bolsa de palabras).

    Textos con palabras en comun quedan cerca, suficiente para probar el
    cableado sin un modelo real. Cuenta las llamadas para verificar el cache.
    """

    def __init__(self, dimension: int = DIMENSION) -> None:
        self.dimension = dimension
        self.llamadas_query = 0

    def _vector(self, texto: str) -> list[float]:
        v = [0.0] * self.dimension
        for palabra in re.findall(r"\w+", texto.lower()):
            # sum de ordinales en vez de hash(): hash() de str cambia entre procesos
            v[sum(ord(c) * (i + 1) for i, c in enumerate(palabra)) % self.dimension] += 1.0
        return v

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.llamadas_query += 1
        return self._vector(text)


def _coseno(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norma = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norma if norma else 0.0


def _cumple(metadata: dict[str, Any], filtro: dict[str, Any] | None) -> bool:
    if not filtro:
        return True
    return all(metadata.get(campo) == cond["$eq"] for campo, cond in filtro.items())


class IndexFalso:
    """Imita la parte de `pinecone.Index` que usa el proyecto (upsert/query/delete/stats)."""

    def __init__(self) -> None:
        self.namespaces: dict[str, dict[str, dict[str, Any]]] = {}
        self.upserts: list[tuple[str, int]] = []

    def upsert(self, vectors: list[dict[str, Any]], namespace: str = "") -> None:
        ns = self.namespaces.setdefault(namespace, {})
        for v in vectors:
            ns[v["id"]] = v
        self.upserts.append((namespace, len(vectors)))

    def delete(self, delete_all: bool = False, namespace: str = "") -> None:
        if delete_all:
            self.namespaces.pop(namespace, None)

    def query(self, vector, top_k, namespace="", include_metadata=False, filter=None):  # noqa: A002
        candidatos = [
            v for v in self.namespaces.get(namespace, {}).values() if _cumple(v["metadata"], filter)
        ]
        candidatos.sort(key=lambda v: _coseno(vector, v["values"]), reverse=True)
        matches = [
            SimpleNamespace(id=v["id"], score=_coseno(vector, v["values"]), metadata=dict(v["metadata"]))
            for v in candidatos[:top_k]
        ]
        return SimpleNamespace(matches=matches)

    def describe_index_stats(self):
        return SimpleNamespace(
            namespaces={ns: SimpleNamespace(vector_count=len(vs)) for ns, vs in self.namespaces.items()}
        )


class ClientePineconeFalso:
    """Imita `pinecone.Pinecone` para probar `setup_pinecone` (crear vs reutilizar el indice)."""

    def __init__(self, existentes: dict[str, int] | None = None) -> None:
        self.indices = dict(existentes or {})
        self.creados: list[dict[str, Any]] = []

    def has_index(self, nombre: str) -> bool:
        return nombre in self.indices

    def describe_index(self, nombre: str):
        return SimpleNamespace(dimension=self.indices[nombre])

    def create_index(self, *, name, dimension, metric, spec):
        self.indices[name] = dimension
        self.creados.append({"name": name, "dimension": dimension, "metric": metric, "spec": spec})

    def Index(self, nombre: str):  # noqa: N802 - mismo nombre que el SDK
        return IndexFalso()
