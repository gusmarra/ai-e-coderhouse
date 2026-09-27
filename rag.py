"""Cadena RAG asincrona: retriever de ChromaDB + generacion grounded con LCEL.

El flujo es::

    documentos = await retriever.ainvoke(pregunta)     # ChromaDB, top_k fragmentos
    contexto   = _formatear_contexto(documentos)        # cada chunk con su fuente
    modelo_out = await (PROMPT | modelo | parser).ainvoke({...})  # LCEL
    resultado  = RespuestaRAG(..., referencias=_armar_referencias(documentos))

`PROMPT | modelo | parser` es la cadena LCEL propiamente dicha: un
`ChatPromptTemplate` con las reglas de veracidad, el chat model y un
`PydanticOutputParser` que valida `RespuestaModelo`. Las referencias no
pasan por ahi a proposito (ver `schemas.py`): se arman con los metadatos que
devuelve el retriever, no con lo que el LLM diga que uso.

`get_rag_response(query)` es el punto de entrada async que orquesta las dos
partes y devuelve un `RespuestaRAG` ya validado.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever

from ingest import ingerir
from llm import crear_modelo
from schemas import Referencia, RespuestaModelo, RespuestaRAG

__all__ = [
    "MAX_CARACTERES_FRAGMENTO",
    "PROMPT",
    "TOP_K",
    "construir_generacion",
    "get_rag_response",
    "retriever_por_defecto",
]

log = logging.getLogger("clase_3.rag")

#: Entre 3 y 5, como pide la consigna: mas fragmentos no mejora la respuesta,
#: satura el contexto ("contexto infinito") y degrada la atencion del modelo
#: (lost in the middle) ademas de acercarse al limite de tokens.
TOP_K = 4

#: El fragmento que viaja en cada `Referencia` es un preview para citar, no
#: el chunk entero (eso ya esta en el contexto que vio el modelo).
MAX_CARACTERES_FRAGMENTO = 280


# ----------------------------------------------------------------------
# Prompt: el "filtro de veracidad"
# ----------------------------------------------------------------------
SISTEMA = (
    "Sos el asistente virtual del Hotel Bahia Serena. Respondes preguntas de "
    "huespedes y personal sobre los documentos institucionales del hotel "
    "(politicas, reglamento interno, servicios y protocolos), usando "
    "exclusivamente el CONTEXTO que se te provee.\n\n"
    "Reglas estrictas:\n"
    "- Respondes UNICAMENTE con informacion presente en el CONTEXTO. No usas "
    "conocimiento externo, no completas con supuestos ni generalizas mas "
    "alla de lo que el texto dice.\n"
    "- Si el CONTEXTO no contiene la informacion necesaria para responder la "
    "pregunta, el campo 'respuesta' debe ser exactamente: "
    "'No tengo esa informacion en los documentos disponibles.' y "
    "'encontrado_en_contexto' debe ser false.\n"
    "- No inventes numeros, formulas, nombres de campos ni reglas que no "
    "esten en el CONTEXTO.\n\n"
    "{formato}"
)

HUMANO = "<contexto>\n{contexto}\n</contexto>\n\nPregunta: {pregunta}"

_parser: PydanticOutputParser[RespuestaModelo] = PydanticOutputParser(pydantic_object=RespuestaModelo)

#: Variables de entrada: `contexto`, `pregunta`. `formato` queda fijo via
#: `.partial()` con las instrucciones que genera el propio parser.
PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [("system", SISTEMA), ("human", HUMANO)]
).partial(formato=_parser.get_format_instructions())


# ----------------------------------------------------------------------
# Retriever y contexto
# ----------------------------------------------------------------------
def retriever_por_defecto(*, top_k: int = TOP_K) -> VectorStoreRetriever:
    """Abre (o puebla, si esta vacia) la coleccion de ChromaDB y devuelve su retriever."""
    return ingerir().as_retriever(search_kwargs={"k": top_k})


def _formatear_contexto(documentos: list[Document]) -> str:
    """Concatena los chunks recuperados, cada uno etiquetado con su fuente, para el prompt."""
    if not documentos:
        return "(no se encontraron fragmentos relevantes)"
    return "\n\n".join(
        f"[Fuente: {doc.metadata.get('source', 'desconocida')}]\n{doc.page_content}" for doc in documentos
    )


def _armar_referencias(documentos: list[Document]) -> list[Referencia]:
    """Referencias reales: se arman desde los metadatos del retriever, no desde
    lo que el LLM diga. Un LLM puede citar una fuente que nunca vio en el
    contexto; el retriever no."""
    return [
        Referencia(
            fuente=doc.metadata.get("source", "desconocida"),
            fragmento=(
                doc.page_content[:MAX_CARACTERES_FRAGMENTO] + "..."
                if len(doc.page_content) > MAX_CARACTERES_FRAGMENTO
                else doc.page_content
            ),
        )
        for doc in documentos
    ]


# ----------------------------------------------------------------------
# La cadena LCEL de generacion: prompt | modelo | PydanticOutputParser
# ----------------------------------------------------------------------
def construir_generacion(modelo: BaseChatModel | None = None) -> Runnable[dict[str, Any], RespuestaModelo]:
    """`PROMPT | modelo | PydanticOutputParser(RespuestaModelo)`, la cadena LCEL propiamente dicha."""
    modelo = modelo if modelo is not None else crear_modelo(temperature=0.0)
    return (PROMPT | modelo | _parser).with_config(run_name="generacion_grounded")


_generacion: Runnable[dict[str, Any], RespuestaModelo] | None = None


def _generacion_por_defecto() -> Runnable[dict[str, Any], RespuestaModelo]:
    global _generacion
    if _generacion is None:
        _generacion = construir_generacion()
    return _generacion


# ----------------------------------------------------------------------
# Entrada asincrona
# ----------------------------------------------------------------------
async def get_rag_response(
    query: str,
    *,
    retriever: VectorStoreRetriever | Runnable[str, list[Document]] | None = None,
    generacion: Runnable[dict[str, Any], RespuestaModelo] | None = None,
) -> RespuestaRAG:
    """Busca en ChromaDB, genera una respuesta grounded y la devuelve validada.

    1. Recupera los `TOP_K` fragmentos mas similares a `query` (async).
    2. Arma el contexto y llama a la cadena `PROMPT | modelo | parser`.
    3. Adjunta las referencias reales (fuente + extracto de cada chunk).

    Lanza `ValueError` si `query` esta vacia.
    """
    if not query or not query.strip():
        raise ValueError("la consulta esta vacia")

    pregunta = query.strip()
    retriever = retriever if retriever is not None else retriever_por_defecto()
    generacion = generacion if generacion is not None else _generacion_por_defecto()

    log.info("consulta: %r", pregunta)
    documentos = await retriever.ainvoke(pregunta)
    contexto = _formatear_contexto(documentos)

    respuesta_modelo = await generacion.ainvoke({"contexto": contexto, "pregunta": pregunta})

    resultado = RespuestaRAG(
        respuesta=respuesta_modelo.respuesta,
        encontrado_en_contexto=respuesta_modelo.encontrado_en_contexto,
        referencias=_armar_referencias(documentos),
    )
    log.info(
        "respuesta (encontrado_en_contexto=%s, %d chunks recuperados)",
        resultado.encontrado_en_contexto,
        len(resultado.referencias),
    )
    return resultado
