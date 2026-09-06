"""Modelos Pydantic: contrato de datos unico para todos los proveedores.

Definir esto *antes* que los clientes evita el clasico problema de andar
pasando diccionarios anidados (`{"choices": [{"message": {...}}]}`) por toda
la aplicacion: cada proveedor traduce su respuesta cruda a `ModelResponse`
y el resto del codigo nunca ve la forma nativa del SDK.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

__all__ = [
    "ChatMessage",
    "Conversation",
    "ErrorResponse",
    "MessageLike",
    "ModelConfig",
    "ModelResponse",
    "RetryConfig",
    "Role",
    "StreamChunk",
    "Usage",
]


class Role(StrEnum):
    """Roles canonicos. Cada proveedor los mapea a su propio vocabulario."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Un turno de conversacion validado."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    content: str = Field(min_length=1, description="Texto del mensaje, no vacio.")
    name: str | None = Field(default=None, description="Autor opcional (multi-agente).")

    @field_validator("content")
    @classmethod
    def _no_solo_espacios(cls, value: str) -> str:
        limpio = value.strip()
        if not limpio:
            raise ValueError("content no puede ser solo espacios en blanco")
        return limpio

    # Constructores cortos: ChatMessage.user("hola") en lugar del dict.
    @classmethod
    def system(cls, content: str) -> Self:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Self:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> Self:
        return cls(role=Role.ASSISTANT, content=content)


#: Todo lo que `BaseLLMClient` sabe convertir en una `Conversation`.
MessageLike = str | ChatMessage | Mapping[str, Any]


class Conversation(BaseModel):
    """Lista de mensajes + las normalizaciones que todo proveedor necesita."""

    model_config = ConfigDict(frozen=True)

    messages: tuple[ChatMessage, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _requiere_turno_no_system(self) -> Self:
        if all(m.role is Role.SYSTEM for m in self.messages):
            raise ValueError("la conversacion necesita al menos un mensaje user/assistant")
        return self

    @classmethod
    def coerce(cls, messages: MessageLike | Iterable[MessageLike]) -> Conversation:
        """Acepta un string, un dict, un `ChatMessage` o una secuencia de ellos."""
        if isinstance(messages, Conversation):
            return messages
        if isinstance(messages, (str, ChatMessage, Mapping)):
            items: Sequence[MessageLike] = [messages]  # type: ignore[list-item]
        else:
            items = list(messages)
        normalizados = [
            ChatMessage.user(item) if isinstance(item, str) else ChatMessage.model_validate(item)
            for item in items
        ]
        return cls(messages=tuple(normalizados))

    @property
    def system_prompt(self) -> str | None:
        """Los mensajes `system` unidos: Anthropic y Gemini los reciben aparte."""
        partes = [m.content for m in self.messages if m.role is Role.SYSTEM]
        return "\n\n".join(partes) if partes else None

    @property
    def turns(self) -> tuple[ChatMessage, ...]:
        """Solo user/assistant, en orden."""
        return tuple(m for m in self.messages if m.role is not Role.SYSTEM)

    def merged_turns(self) -> tuple[ChatMessage, ...]:
        """`turns` fusionando mensajes consecutivos del mismo rol.

        La API de Anthropic espera roles alternados; esto evita un 400 cuando
        el llamador encolo dos mensajes de usuario seguidos.
        """
        fusionados: list[ChatMessage] = []
        for mensaje in self.turns:
            if fusionados and fusionados[-1].role is mensaje.role:
                previo = fusionados[-1]
                fusionados[-1] = previo.model_copy(
                    update={"content": f"{previo.content}\n\n{mensaje.content}"}
                )
            else:
                fusionados.append(mensaje)
        return tuple(fusionados)


class ModelConfig(BaseModel):
    """Parametros de inferencia, validados antes de salir a la red."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    model: str = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0, le=200_000)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stop: tuple[str, ...] = ()
    timeout_s: float = Field(default=60.0, gt=0.0, le=600.0)
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Parametros propios del proveedor (se pasan tal cual al SDK).",
    )


class RetryConfig(BaseModel):
    """Backoff exponencial con jitter para errores transitorios."""

    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(default=3, ge=1, le=10)
    initial_backoff_s: float = Field(default=0.5, gt=0.0)
    max_backoff_s: float = Field(default=8.0, gt=0.0)
    multiplier: float = Field(default=2.0, ge=1.0)
    jitter: bool = True

    @model_validator(mode="after")
    def _coherencia_backoff(self) -> Self:
        if self.max_backoff_s < self.initial_backoff_s:
            raise ValueError("max_backoff_s debe ser >= initial_backoff_s")
        return self


class Usage(BaseModel):
    """Tokens consumidos. Un 0 significa que el proveedor no lo informo."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ModelResponse(BaseModel):
    """Respuesta completa, identica venga de OpenAI, Anthropic o Gemini."""

    model_config = ConfigDict(frozen=True)

    content: str
    provider: str
    model: str
    role: Literal[Role.ASSISTANT] = Role.ASSISTANT
    usage: Usage = Field(default_factory=Usage)
    finish_reason: str | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    attempts: int = Field(default=1, ge=1)
    # El payload crudo queda disponible para debug pero fuera de dumps y logs.
    raw: dict[str, Any] | None = Field(default=None, exclude=True, repr=False)
    ok: Literal[True] = True

    def as_message(self) -> ChatMessage:
        """Para reinyectar la respuesta en el historial."""
        return ChatMessage.assistant(self.content)


class StreamChunk(BaseModel):
    """Fragmento de streaming.

    El ultimo chunk llega con `is_final=True`, `delta=""` y el `usage` final:
    asi el consumidor obtiene los totales sin estado mutable en el cliente.
    """

    model_config = ConfigDict(frozen=True)

    delta: str = ""
    index: int = Field(default=0, ge=0)
    provider: str
    model: str
    is_final: bool = False
    usage: Usage | None = None
    finish_reason: str | None = None


class ErrorResponse(BaseModel):
    """Error con los reintentos ya agotados, en forma de dato (no de excepcion).

    Lo devuelve `generate_safe()` para que un loop principal pueda seguir
    procesando el resto de la cola sin envolver todo en try/except.
    """

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str | None = None
    error_type: str
    message: str
    retryable: bool = False
    status_code: int | None = None
    attempts: int = Field(default=1, ge=1)
    ok: Literal[False] = False
