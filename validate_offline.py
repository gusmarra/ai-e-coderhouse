"""Validacion del pipeline RAG sin red, sin API keys y sin llamar a ChromaDB
ni a un proveedor real.

Reemplaza cada pieza que hable con el exterior por un doble de prueba:

* el chat model -> `FakeListChatModel` (parte de `langchain_core`), que
  devuelve texto fijo sin hacer ninguna llamada HTTP,
* el retriever -> un `RunnableLambda` que devuelve `Document`s fijos en vez
  de consultar ChromaDB.

Eso permite probar lo que realmente importa y no se puede testear contra la
API real de forma determinista ni gratis:

* el chunking (tamano y overlap de los fragmentos),
* las restricciones de los esquemas Pydantic (`RespuestaModelo`, `Referencia`,
  `RespuestaRAG`),
* que el prompt lleve las instrucciones de formato del parser,
* que `_armar_referencias` arme las citas desde los metadatos del retriever
  (y no desde lo que diga el LLM),
* la cadena LCEL completa (`PROMPT | modelo_falso | parser`) parseando un
  JSON valido de punta a punta,
* que `get_rag_response` orqueste retriever + generacion y valide la entrada.

    python validate_offline.py
"""

from __future__ import annotations

import asyncio
import sys

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from chain import (
    HUMANO,
    PROMPT,
    SISTEMA,
    _armar_referencias,
    _formatear_contexto,
    construir_generacion,
    get_rag_response,
)
from ingest import CHUNK_OVERLAP, CHUNK_SIZE, _cargar_documentos, _fragmentar
from schemas import MENSAJE_SIN_CONTEXTO, Referencia, RespuestaModelo, RespuestaRAG

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

fallas: list[str] = []


def check(condicion: bool, descripcion: str) -> None:
    print(f"  {'ok   ' if condicion else 'FALLA'} {descripcion}")
    if not condicion:
        fallas.append(descripcion)


def titulo(texto: str) -> None:
    print(f"\n{texto}")


# ----------------------------------------------------------------------
# 1. Chunking (ingest.py) sobre los documentos reales de data/
# ----------------------------------------------------------------------
def probar_chunking() -> None:
    titulo("1. Chunking de data/*.md")

    documentos = _cargar_documentos()
    check(len(documentos) >= 3, f"hay al menos 3 documentos fuente (encontrados: {len(documentos)})")
    check(
        len({d.metadata["source"] for d in documentos}) == len(documentos),
        "cada documento tiene un 'source' distinto en metadata",
    )

    chunks = _fragmentar(documentos)
    check(len(chunks) > len(documentos), f"la fragmentacion genero mas de un chunk por documento ({len(chunks)} chunks)")

    # Margen sobre CHUNK_SIZE: el splitter recursivo puede pasarse un poco
    # cuando un separador (p.ej. una fila de tabla larga) no admite un corte
    # mas fino; lo que no debe pasar es que un chunk sea varias veces el
    # tamano pedido.
    sobredimensionados = [c for c in chunks if len(c.page_content) > CHUNK_SIZE * 1.5]
    check(not sobredimensionados, f"ningun chunk supera 1.5x el chunk_size ({CHUNK_SIZE} caracteres)")
    check(CHUNK_OVERLAP > 0, "el overlap configurado es mayor a cero")
    check(
        all(c.metadata.get("source") for c in chunks),
        "todos los chunks conservan el 'source' del documento original",
    )

    fuentes_en_chunks = {c.metadata["source"] for c in chunks}
    fuentes_originales = {d.metadata["source"] for d in documentos}
    check(fuentes_en_chunks == fuentes_originales, "no se perdio ningun documento al fragmentar")


# ----------------------------------------------------------------------
# 2. Esquemas Pydantic (schemas.py)
# ----------------------------------------------------------------------
def probar_schemas() -> None:
    titulo("2. Esquemas Pydantic")

    valido = RespuestaModelo(respuesta="El Sharpe Ratio es (Rp - Rf) / sigma_p.", encontrado_en_contexto=True)
    check(valido.encontrado_en_contexto is True, "RespuestaModelo acepta una respuesta valida")

    sin_contexto = RespuestaModelo(respuesta=MENSAJE_SIN_CONTEXTO, encontrado_en_contexto=False)
    check(sin_contexto.respuesta == MENSAJE_SIN_CONTEXTO, "RespuestaModelo acepta el mensaje fijo de 'no lo se'")

    try:
        RespuestaModelo(respuesta="", encontrado_en_contexto=True)
        check(False, "RespuestaModelo rechaza una respuesta vacia")
    except ValidationError:
        check(True, "RespuestaModelo rechaza una respuesta vacia")

    try:
        RespuestaModelo(respuesta="ok", encontrado_en_contexto=True, campo_extra="no deberia existir")
        check(False, "RespuestaModelo rechaza campos extra (extra='forbid')")
    except ValidationError:
        check(True, "RespuestaModelo rechaza campos extra (extra='forbid')")

    ref = Referencia(fuente="ratio-de-sharpe.md", fragmento="Sharpe ratio = (Rp - Rf) / sigma_p")
    rag = RespuestaRAG(respuesta="...", encontrado_en_contexto=True, referencias=[ref])
    check(len(rag.referencias) == 1, "RespuestaRAG agrupa respuesta + referencias")

    vacio = RespuestaRAG(respuesta=MENSAJE_SIN_CONTEXTO, encontrado_en_contexto=False)
    check(vacio.referencias == [], "RespuestaRAG admite referencias vacias por defecto")


# ----------------------------------------------------------------------
# 3. Prompt (chain.py)
# ----------------------------------------------------------------------
def probar_prompt() -> None:
    titulo("3. Prompt de la cadena de generacion")

    check("CONTEXTO" in SISTEMA, "el prompt de sistema instruye a usar solo el CONTEXTO")
    check(MENSAJE_SIN_CONTEXTO in SISTEMA, "el prompt de sistema fija el mensaje exacto de 'no lo se'")
    check("{contexto}" in HUMANO and "{pregunta}" in HUMANO, "el prompt humano expone las variables contexto/pregunta")

    mensajes = PROMPT.format_messages(contexto="CTX", pregunta="P")
    sistema_render = mensajes[0].content
    check(
        "RespuestaModelo" in sistema_render or "respuesta" in sistema_render.lower(),
        "las instrucciones de formato del PydanticOutputParser llegan al prompt final",
    )


# ----------------------------------------------------------------------
# 4. Referencias armadas desde el retriever, no desde el LLM
# ----------------------------------------------------------------------
def probar_referencias() -> None:
    titulo("4. Contexto y referencias a partir de Documents")

    documentos = [
        Document(page_content="El Sharpe ratio se calcula como (Rp - Rf) / sigma_p.", metadata={"source": "ratio-de-sharpe.md"}),
        Document(page_content="x" * 400, metadata={"source": "operaciones-compuestas.md"}),
    ]

    contexto = _formatear_contexto(documentos)
    check("[Fuente: ratio-de-sharpe.md]" in contexto, "el contexto etiqueta cada chunk con su fuente")
    check("[Fuente: operaciones-compuestas.md]" in contexto, "el contexto incluye todos los documentos recuperados")

    referencias = _armar_referencias(documentos)
    check(len(referencias) == len(documentos), "se arma una Referencia por cada Document recuperado")
    check(referencias[0].fuente == "ratio-de-sharpe.md", "la Referencia conserva el nombre del archivo original")
    check(len(referencias[1].fragmento) < 400, "un chunk largo se recorta a un extracto en la Referencia")

    check(_formatear_contexto([]) != "", "el contexto vacio no rompe el formateo (retriever sin resultados)")
    check(_armar_referencias([]) == [], "sin documentos recuperados no hay referencias")


# ----------------------------------------------------------------------
# 5. Cadena LCEL completa con un chat model falso
# ----------------------------------------------------------------------
async def probar_generacion_lcel() -> None:
    titulo("5. Cadena LCEL (PROMPT | modelo | PydanticOutputParser) con modelo falso")

    modelo_falso = FakeListChatModel(
        responses=['{"respuesta": "El Sharpe ratio es (Rp - Rf) / sigma_p.", "encontrado_en_contexto": true}']
    )
    generacion = construir_generacion(modelo_falso)
    resultado = await generacion.ainvoke({"contexto": "...", "pregunta": "¿Como se calcula el Sharpe ratio?"})
    check(isinstance(resultado, RespuestaModelo), "el parser devuelve una instancia de RespuestaModelo")
    check(resultado.encontrado_en_contexto is True, "el JSON del modelo falso se parsea correctamente")


# ----------------------------------------------------------------------
# 6. get_rag_response: orquestacion de retriever + generacion
# ----------------------------------------------------------------------
async def probar_get_rag_response() -> None:
    titulo("6. get_rag_response (retriever + generacion, ambos dobles de prueba)")

    documentos_falsos = [
        Document(page_content="El Sharpe ratio es (Rp - Rf) / sigma_p.", metadata={"source": "ratio-de-sharpe.md"}),
        Document(page_content="Rp es el retorno promedio diario de la cartera.", metadata={"source": "ratio-de-sharpe.md"}),
    ]
    retriever_falso = RunnableLambda(lambda _pregunta: documentos_falsos)

    generacion_ok = RunnableLambda(
        lambda _entrada: RespuestaModelo(respuesta="El Sharpe ratio es (Rp - Rf) / sigma_p.", encontrado_en_contexto=True)
    )
    resultado = await get_rag_response("¿Como se calcula el Sharpe ratio?", retriever=retriever_falso, generacion=generacion_ok)
    check(isinstance(resultado, RespuestaRAG), "get_rag_response devuelve un RespuestaRAG")
    check(resultado.encontrado_en_contexto is True, "propaga encontrado_en_contexto del modelo")
    check(len(resultado.referencias) == 2, "adjunta una referencia por cada Document recuperado")

    generacion_sin_contexto = RunnableLambda(
        lambda _entrada: RespuestaModelo(respuesta=MENSAJE_SIN_CONTEXTO, encontrado_en_contexto=False)
    )
    resultado_trampa = await get_rag_response(
        "¿Cuanto cobra Alphinance de interes hipotecario?", retriever=retriever_falso, generacion=generacion_sin_contexto
    )
    check(resultado_trampa.respuesta == MENSAJE_SIN_CONTEXTO, "una pregunta trampa devuelve el mensaje fijo de 'no lo se'")

    try:
        await get_rag_response("   ", retriever=retriever_falso, generacion=generacion_ok)
        check(False, "get_rag_response rechaza una consulta vacia")
    except ValueError:
        check(True, "get_rag_response rechaza una consulta vacia")


async def main() -> int:
    probar_chunking()
    probar_schemas()
    probar_prompt()
    probar_referencias()
    await probar_generacion_lcel()
    await probar_get_rag_response()

    print("\n" + "=" * 72)
    if fallas:
        print(f"{len(fallas)} verificacion(es) fallaron:")
        for falla in fallas:
            print(f"  - {falla}")
        return 1
    print("Todas las verificaciones pasaron.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
