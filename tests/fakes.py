"""Doble de prueba del LLM: devuelve respuestas guionadas y registra lo que recibe."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def llamada(herramienta: str, id_: str, /, **args: Any) -> AIMessage:
    """AIMessage que pide ejecutar una herramienta."""
    pedido = {"name": herramienta, "args": args, "id": id_, "type": "tool_call"}
    return AIMessage(content="", tool_calls=[pedido])


class ModeloGuionado(BaseChatModel):
    """Chat model que reproduce `guion` en orden (y repite el ultimo si se agota).

    `bind_tools` devuelve el mismo objeto: las llamadas a herramientas ya vienen en el guion.
    `recibidos` guarda los mensajes que vio en cada invocacion (para verificar la memoria).
    """

    guion: list[AIMessage]
    recibidos: list[list[BaseMessage]] = Field(default_factory=list)
    indice: int = 0

    @property
    def _llm_type(self) -> str:
        return "modelo-guionado"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ModeloGuionado:  # type: ignore[override]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.recibidos.append(list(messages))
        respuesta = self.guion[min(self.indice, len(self.guion) - 1)]
        self.indice += 1
        return ChatResult(generations=[ChatGeneration(message=respuesta.model_copy(update={"id": None}))])
