from __future__ import annotations

import pytest
from langchain_core.documents import Document

from ingest import cargar_documentos, fragmentar
from tests.fakes import EmbeddingsFalsos, IndexFalso


@pytest.fixture(scope="session")
def documentos() -> list[Document]:
    return cargar_documentos()


@pytest.fixture(scope="session")
def chunks(documentos: list[Document]) -> list[Document]:
    return fragmentar(documentos)


@pytest.fixture
def embeddings() -> EmbeddingsFalsos:
    return EmbeddingsFalsos()


@pytest.fixture
def index() -> IndexFalso:
    return IndexFalso()
