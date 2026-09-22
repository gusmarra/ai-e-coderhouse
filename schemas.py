"""Contratos Pydantic de la cadena RAG: lo que el LLM completa y lo que el
sistema le agrega antes de entregarlo al usuario.

Se separan dos modelos a proposito:

* `RespuestaModelo` es lo unico que el LLM redacta, y es deliberadamente
  chico (dos campos). Las referencias NO las escribe el modelo: si le
  pidieramos que las redactara, podria citar una fuente que nunca recupero
  (alucinacion de cita) o inventar un fragmento que no existe en ningun
  documento. `PydanticOutputParser` valida este objeto contra el JSON que
  devuelve el LLM.
* `RespuestaRAG` es la salida final que ve `get_rag_response()`: la
  respuesta ya validada mas las referencias, que el propio retriever arma a
  partir de los metadatos de los chunks efectivamente recuperados en
  ChromaDB (ver `chain.py::_armar_referencias`).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Referencia", "RespuestaModelo", "RespuestaRAG"]

#: Mensaje fijo que el prompt le exige al modelo cuando el contexto no alcanza.
#: Se centraliza aca para que el prompt (chain.py) y las pruebas offline
#: (validate_offline.py) usen exactamente el mismo texto.
MENSAJE_SIN_CONTEXTO = "No tengo esa informacion en los documentos disponibles."


class RespuestaModelo(BaseModel):
    """Lo que el LLM devuelve, parseado por `PydanticOutputParser`."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    respuesta: str = Field(
        min_length=1,
        max_length=2000,
        description=(
            "La respuesta a la pregunta del usuario, redactada solo con "
            "informacion presente en el CONTEXTO. Si el contexto no alcanza "
            f"para responder, el texto debe ser exactamente: {MENSAJE_SIN_CONTEXTO!r}"
        ),
    )
    encontrado_en_contexto: bool = Field(
        description=(
            "true si la respuesta se pudo construir con el CONTEXTO "
            "recuperado; false si el contexto no contenia la informacion "
            "necesaria (en ese caso 'respuesta' debe ser el mensaje fijo de "
            "'no lo se')."
        )
    )


class Referencia(BaseModel):
    """Un fragmento efectivamente recuperado por ChromaDB (no redactado por el LLM)."""

    model_config = ConfigDict(extra="forbid")

    fuente: str = Field(description="Nombre del archivo de origen del fragmento.")
    fragmento: str = Field(description="Extracto del chunk usado como contexto.")


class RespuestaRAG(BaseModel):
    """Salida final de `get_rag_response()`: respuesta validada + referencias verificables."""

    model_config = ConfigDict(extra="forbid")

    respuesta: str
    encontrado_en_contexto: bool
    referencias: list[Referencia] = Field(default_factory=list)
