"""Dobles de prueba: LLM guionado (sin red ni keys) para el supervisor y los especialistas."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field

from app.agents.supervisor import Decision


def llamada(herramienta: str, id_: str, /, **args: Any) -> AIMessage:
    """AIMessage que pide ejecutar una herramienta."""
    return AIMessage(content="", tool_calls=[{"name": herramienta, "args": args, "id": id_, "type": "tool_call"}])


def decision(siguiente: str, instruccion: str = "", razon: str = "test") -> Decision:
    return Decision(siguiente=siguiente, instruccion=instruccion, razon=razon)  # type: ignore[arg-type]


class ModeloGuionado(BaseChatModel):
    """Reproduce `guion` en orden (repite el ultimo si se agota) y guarda lo que recibio.

    `decisiones` alimenta `with_structured_output`: asi un mismo doble sirve de supervisor
    (decisiones estructuradas + sintesis por `guion`) o de especialista (herramientas por `guion`).
    """

    guion: list[AIMessage] = Field(default_factory=list)
    decisiones: list[Decision] = Field(default_factory=list)
    recibidos: list[list[BaseMessage]] = Field(default_factory=list)
    indice: int = 0
    indice_decisiones: int = 0

    @property
    def _llm_type(self) -> str:
        return "modelo-guionado"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ModeloGuionado:  # type: ignore[override]
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[Any, Any]:  # type: ignore[override]
        def siguiente(mensajes: Any) -> Decision:
            self.recibidos.append(list(mensajes))
            d = self.decisiones[min(self.indice_decisiones, len(self.decisiones) - 1)]
            self.indice_decisiones += 1
            return d

        return RunnableLambda(siguiente)

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
