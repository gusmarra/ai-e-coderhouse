"""Caso practico end-to-end: mesa de ayuda de una tienda online.

Toma una tanda de tickets de soporte reales-en-formato y los lleva de punta a
punta con `llm_client`, ejercitando todo lo que el cliente tiene desarrollado:

    PASO 1  Triage en lote          generate_safe + asyncio.gather (concurrencia)
    PASO 2  Salida estructurada     el texto del modelo validado con Pydantic
    PASO 3  Priorizacion            logica de negocio deterministica, sin LLM
    PASO 4  Respuesta al cliente    stream() token a token, con usage al final
    PASO 5  Conversacion multivuelta historial + ModelResponse.as_message()
    PASO 6  Resiliencia             key invalida -> ErrorResponse -> switch()
    PASO 7  Reporte                 tokens, latencia e intentos acumulados

Uso::

    python caso_practico.py                  # primer proveedor con key en .env
    python caso_practico.py gemini           # proveedor explicito
    python caso_practico.py --simulado       # sin red ni keys (respuestas fijas)
    python caso_practico.py gemini --paso 4  # correr un solo paso
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_client import (
    AsyncLLMManager,
    ChatMessage,
    ErrorResponse,
    ModelResponse,
    Provider,
    RetryConfig,
    Usage,
    available_providers,
)
from llm_client.base import BaseLLMClient
from llm_client.exceptions import AuthenticationError, LLMError
from llm_client.schemas import Conversation, ModelConfig, StreamChunk

# Los modelos de Anthropic razonan por defecto y ese razonamiento consume
# max_tokens: se pide holgura aunque la respuesta sea corta.
MAX_TOKENS = 2000

# En consolas Windows (cp1252) un acento del modelo haria fallar el print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for ruidoso in ("httpx", "google_genai", "openai", "anthropic"):
    logging.getLogger(ruidoso).setLevel(logging.WARNING)
log = logging.getLogger("caso")


# ======================================================================
# El dominio: los datos de entrada y el contrato de salida
# ======================================================================


@dataclass(frozen=True)
class Ticket:
    """Lo que entra por el canal de soporte, tal cual lo escribio el cliente."""

    id: str
    canal: str
    cliente: str
    texto: str


TICKETS: tuple[Ticket, ...] = (
    Ticket(
        id="T-1041",
        canal="email",
        cliente="Marcela D.",
        texto=(
            "Compre una notebook el 2 de septiembre, pague con tarjeta y me llego "
            "la confirmacion, pero el seguimiento dice 'etiqueta generada' hace 6 dias "
            "y no se mueve. Necesito saber si me la van a entregar o me devuelven la plata."
        ),
    ),
    Ticket(
        id="T-1042",
        canal="whatsapp",
        cliente="Jonatan R.",
        texto="hola! el cupon BIENVENIDA10 me tira invalido al pagar, lo puedo usar igual?",
    ),
    Ticket(
        id="T-1043",
        canal="web",
        cliente="Sofia L.",
        texto=(
            "Es la TERCERA vez que escribo. Me cobraron DOS VECES el mismo pedido #88213, "
            "son $340.000 retenidos de mi cuenta. Si no me responden hoy hago la denuncia "
            "en Defensa del Consumidor."
        ),
    ),
    Ticket(
        id="T-1044",
        canal="email",
        cliente="Pablo M.",
        texto=(
            "Buenas, queria saber si la cafetera Modelo X es compatible con capsulas "
            "de otras marcas y si tiene garantia oficial."
        ),
    ),
    Ticket(
        id="T-1045",
        canal="whatsapp",
        cliente="Ana T.",
        texto=(
            "Recibi el pedido #88510 pero vino el talle M y yo pedi el L. Quiero cambiarlo, "
            "todavia tiene la etiqueta puesta."
        ),
    ),
    Ticket(
        id="T-1046",
        canal="web",
        cliente="Anonimo",
        texto="asdasd",  # ruido: el modelo tiene que poder decir 'no se entiende'
    ),
)


class Categoria(StrEnum):
    ENVIO = "envio"
    FACTURACION = "facturacion"
    DEVOLUCION = "devolucion"
    PRODUCTO = "producto"
    OTRO = "otro"


class Sentimiento(StrEnum):
    POSITIVO = "positivo"
    NEUTRO = "neutro"
    NEGATIVO = "negativo"
    ENOJADO = "enojado"


class Triage(BaseModel):
    """El contrato de salida del PASO 1, validado antes de tocar el negocio.

    `extra="ignore"` porque el modelo a veces agrega campos de mas; lo que no
    se tolera es que falte un campo o que la urgencia venga fuera de rango.
    """

    model_config = ConfigDict(extra="ignore")

    categoria: Categoria
    urgencia: int = Field(ge=1, le=5, description="1 = puede esperar, 5 = critico")
    sentimiento: Sentimiento
    requiere_humano: bool
    resumen: str = Field(min_length=1, max_length=200)


@dataclass
class Resultado:
    """Un ticket ya procesado, con lo que salio bien y lo que salio mal."""

    ticket: Ticket
    triage: Triage | None = None
    fallo: str | None = None

    @property
    def ok(self) -> bool:
        return self.triage is not None


SISTEMA_TRIAGE = (
    "Sos el sistema de triage de una tienda online argentina. "
    "Clasificas tickets de soporte y devolves UNICAMENTE un objeto JSON, sin texto "
    "alrededor y sin bloques de codigo, con exactamente estas claves:\n"
    '{"categoria": "envio|facturacion|devolucion|producto|otro", '
    '"urgencia": 1-5, '
    '"sentimiento": "positivo|neutro|negativo|enojado", '
    '"requiere_humano": true|false, '
    '"resumen": "una oracion de hasta 20 palabras"}\n'
    "urgencia 5 solo para plata retenida, riesgo legal o cliente que ya reclamo varias veces. "
    "Si el texto es incomprensible: categoria 'otro', urgencia 1, requiere_humano true."
)

SISTEMA_RESPUESTA = (
    "Sos un agente de soporte de una tienda online argentina. Escribis en espanol rioplatense, "
    "trato de usted, tono calido y concreto. Nunca prometes plazos que no podes garantizar, "
    "nunca inventas numeros de pedido ni montos. Maximo 120 palabras. "
    "Cerras siempre con el proximo paso concreto y el numero de ticket."
)


# ======================================================================
# Metricas: lo que el cliente ya expone en cada respuesta
# ======================================================================


@dataclass
class Metricas:
    """Acumula lo que `ModelResponse` trae de fabrica: tokens, latencia, intentos."""

    llamadas: int = 0
    fallidas: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latencia_total_ms: float = 0.0
    reintentos: int = 0
    por_paso: dict[str, int] = field(default_factory=dict)

    def registrar(self, paso: str, resultado: ModelResponse | ErrorResponse) -> None:
        self.llamadas += 1
        self.por_paso[paso] = self.por_paso.get(paso, 0) + 1
        if isinstance(resultado, ErrorResponse):
            self.fallidas += 1
            self.reintentos += resultado.attempts - 1
            return
        self.input_tokens += resultado.usage.input_tokens
        self.output_tokens += resultado.usage.output_tokens
        self.latencia_total_ms += resultado.latency_ms
        self.reintentos += resultado.attempts - 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


METRICAS = Metricas()


# ======================================================================
# Utilidades de presentacion y parseo
# ======================================================================


def nombre_proveedor(llm: Any) -> str:
    """`AsyncLLMManager.provider` o `BaseLLMClient.provider_name`, segun venga."""
    return getattr(llm, "provider", None) or llm.provider_name


def titulo(texto: str) -> None:
    print(f"\n{'=' * 74}\n{texto}\n{'=' * 74}")


def paso(numero: int, texto: str) -> None:
    print(f"\n--- PASO {numero}: {texto} ---\n")


_CERCA_JSON = re.compile(r"^```[a-zA-Z]*\n?|```$")


def extraer_json(texto: str) -> dict[str, Any]:
    """Rescata el objeto JSON aunque el modelo lo envuelva en ``` o en prosa."""
    limpio = _CERCA_JSON.sub("", texto.strip()).strip()
    inicio, fin = limpio.find("{"), limpio.rfind("}")
    if inicio == -1 or fin <= inicio:
        raise ValueError(f"no hay JSON en la respuesta: {texto[:80]!r}")
    return json.loads(limpio[inicio : fin + 1])


# ======================================================================
# Modo simulado: el mismo caso, sin red ni API keys
# ======================================================================


class ClienteSimulado(BaseLLMClient):
    """Proveedor falso sobre `BaseLLMClient`, para correr la demo sin keys.

    Sirve para dos cosas: mostrar que agregar un proveedor son tres metodos
    privados, y que la demo funcione en una clase sin internet.
    """

    provider_name: ClassVar[str] = "simulado"
    default_model: ClassVar[str] = "demo-1"
    api_key_env: ClassVar[str] = "SIMULADO_API_KEY"

    _TRIAGES: ClassVar[dict[str, dict[str, Any]]] = {
        "T-1041": ("envio", 4, "negativo", True, "Envio detenido hace 6 dias, pide entrega o reembolso."),
        "T-1042": ("facturacion", 2, "neutro", False, "El cupon BIENVENIDA10 figura como invalido al pagar."),
        "T-1043": ("facturacion", 5, "enojado", True, "Cobro duplicado del pedido 88213, amenaza con denuncia."),
        "T-1044": ("producto", 1, "neutro", False, "Consulta de compatibilidad y garantia de la cafetera."),
        "T-1045": ("devolucion", 3, "neutro", False, "Recibio talle M en lugar de L y quiere cambiarlo."),
        "T-1046": ("otro", 1, "neutro", True, "Texto incomprensible, no se puede clasificar."),
    }

    def __init__(self, **kw: Any) -> None:
        kw.setdefault("api_key", "simulada")
        super().__init__(**kw)

    def _verificar_key(self) -> None:
        """Rechaza la key falsa del PASO 6 como lo haria un proveedor real."""
        if str(self._api_key).startswith("sk-clave-invalida"):
            raise AuthenticationError("invalid api key (simulado)", status_code=401)

    def _responder(self, conversation: Conversation) -> str:
        pedido = conversation.turns[-1].content
        sistema = conversation.system_prompt or ""
        if "triage" in sistema.lower():
            for ticket_id, datos in self._TRIAGES.items():
                if ticket_id in pedido:
                    categoria, urgencia, sentimiento, humano, resumen = datos
                    return json.dumps(
                        {
                            "categoria": categoria,
                            "urgencia": urgencia,
                            "sentimiento": sentimiento,
                            "requiere_humano": humano,
                            "resumen": resumen,
                        },
                        ensure_ascii=False,
                    )
            return json.dumps(
                {
                    "categoria": "otro",
                    "urgencia": 3,
                    "sentimiento": "neutro",
                    "requiere_humano": True,
                    "resumen": "Ticket sin guion cargado en la demo, derivado a un humano.",
                },
                ensure_ascii=False,
            )
        return (
            "Hola, gracias por escribirnos. Ya tomamos su caso y lo derivamos al equipo "
            "correspondiente, que lo va a contactar por este mismo canal. Le pedimos "
            "disculpas por la demora. Cualquier novedad, responda a este mensaje citando "
            "su numero de ticket."
        )

    async def _agenerate(self, conversation: Conversation, config: ModelConfig) -> ModelResponse:
        self._verificar_key()
        await asyncio.sleep(0.15)  # latencia simulada, sin bloquear el loop
        texto = self._responder(conversation)
        return ModelResponse(
            content=texto,
            provider=self.provider_name,
            model=config.model,
            usage=Usage(input_tokens=len(str(conversation)) // 4, output_tokens=len(texto) // 4),
            finish_reason="stop",
        )

    async def _astream(
        self, conversation: Conversation, config: ModelConfig
    ) -> AsyncIterator[StreamChunk]:
        self._verificar_key()
        texto = self._responder(conversation)
        palabras = texto.split(" ")
        for indice, palabra in enumerate(palabras):
            await asyncio.sleep(0.03)
            yield StreamChunk(
                delta=palabra + " ", index=indice, provider=self.provider_name, model=config.model
            )
        yield StreamChunk(
            index=len(palabras),
            provider=self.provider_name,
            model=config.model,
            is_final=True,
            usage=Usage(input_tokens=len(str(conversation)) // 4, output_tokens=len(palabras)),
            finish_reason="stop",
        )

    def _translate_error(self, exc: BaseException) -> LLMError | None:
        return None


def abrir_cliente(proveedor: str, **kwargs: Any) -> Any:
    """Devuelve el manager real o el cliente simulado, con la misma interfaz."""
    if proveedor == "simulado":
        return ClienteSimulado(**kwargs)
    return AsyncLLMManager(proveedor, **kwargs)


# ======================================================================
# PASO 1 + 2 - Triage en lote y salida estructurada
# ======================================================================


async def triage_de_un_ticket(llm: Any, ticket: Ticket) -> Resultado:
    """Una llamada por ticket. El error vuelve como dato, no como excepcion."""
    prompt = (
        f"Ticket {ticket.id} (canal {ticket.canal}, cliente {ticket.cliente}):\n"
        f"---\n{ticket.texto}\n---"
    )
    respuesta = await llm.generate_safe(
        [ChatMessage.system(SISTEMA_TRIAGE), ChatMessage.user(prompt)],
        temperature=0.0,  # clasificar pide determinismo, no creatividad
        max_tokens=MAX_TOKENS,
    )
    METRICAS.registrar("triage", respuesta)

    if isinstance(respuesta, ErrorResponse):
        # El proveedor fallo: el ticket no se pierde, se marca para un humano.
        return Resultado(ticket, fallo=f"{respuesta.error_type} (status {respuesta.status_code})")

    try:
        return Resultado(ticket, triage=Triage.model_validate(extraer_json(respuesta.content)))
    except (ValueError, json.JSONDecodeError, ValidationError) as exc:
        # El modelo contesto, pero fuera de contrato: tambien es un fallo manejado.
        return Resultado(ticket, fallo=f"salida fuera de contrato: {type(exc).__name__}")


async def paso_1_y_2_triage(llm: Any) -> list[Resultado]:
    paso(1, f"triage de {len(TICKETS)} tickets en paralelo (asyncio.gather + generate_safe)")
    inicio = time.perf_counter()
    resultados = await asyncio.gather(*(triage_de_un_ticket(llm, t) for t in TICKETS))
    transcurrido = (time.perf_counter() - inicio) * 1000

    paso(2, "la salida del modelo validada contra el schema Pydantic `Triage`")
    print(f"{'ticket':8} {'categoria':13} {'urg':4} {'sentimiento':12} {'humano':7} resumen")
    print("-" * 74)
    for r in resultados:
        if r.triage:
            t = r.triage
            print(
                f"{r.ticket.id:8} {t.categoria.value:13} {t.urgencia:^4} "
                f"{t.sentimiento.value:12} {'si' if t.requiere_humano else 'no':7} "
                f"{t.resumen[:50]}"
            )
        else:
            print(f"{r.ticket.id:8} {'-- FALLO --':13} {'':4} {'':12} {'si':7} {r.fallo}")

    ok = sum(r.ok for r in resultados)
    print(
        f"\n-> {ok}/{len(resultados)} clasificados en {transcurrido:.0f}ms "
        f"({len(TICKETS)} llamadas concurrentes; secuencial seria ~{len(TICKETS)}x)"
    )
    return resultados


# ======================================================================
# PASO 3 - Priorizacion: negocio deterministico sobre datos tipados
# ======================================================================


def paso_3_priorizar(resultados: list[Resultado]) -> list[Resultado]:
    paso(3, "priorizacion y ruteo (logica de negocio pura, sin LLM)")

    def clave(r: Resultado) -> tuple[int, int]:
        if not r.triage:
            return (0, 5)  # lo que fallo va primero: nadie se queda sin respuesta
        enojado = r.triage.sentimiento is Sentimiento.ENOJADO
        return (1, -(r.triage.urgencia * 2 + int(enojado)))

    ordenados = sorted(resultados, key=clave)
    for posicion, r in enumerate(ordenados, start=1):
        if not r.triage:
            destino = "cola humana (fallo tecnico)"
        elif r.triage.requiere_humano or r.triage.urgencia >= 4:
            destino = "agente senior"
        elif r.triage.categoria is Categoria.DEVOLUCION:
            destino = "equipo de devoluciones"
        else:
            destino = "respuesta automatica"
        urgencia = r.triage.urgencia if r.triage else "?"
        print(f"{posicion}. {r.ticket.id}  urgencia={urgencia}  ->  {destino}")

    print("\n-> el LLM clasifico; quien decide el ruteo sigue siendo codigo auditable")
    return ordenados


# ======================================================================
# PASO 4 - Redaccion de la respuesta en streaming
# ======================================================================


async def paso_4_respuesta_streaming(llm: Any, resultado: Resultado) -> str:
    paso(4, f"redaccion de la respuesta al ticket {resultado.ticket.id} (streaming)")
    contexto = resultado.triage.resumen if resultado.triage else resultado.ticket.texto
    prompt = (
        f"Redacta la respuesta al ticket {resultado.ticket.id} de {resultado.ticket.cliente}.\n"
        f"Motivo: {contexto}\n"
        f"Texto original del cliente: {resultado.ticket.texto}\n"
        "El caso ya fue escalado a un agente senior."
    )

    inicio = time.perf_counter()
    primer_token_ms: float | None = None
    piezas: list[str] = []
    final: StreamChunk | None = None

    async for chunk in llm.stream(
        [ChatMessage.system(SISTEMA_RESPUESTA), ChatMessage.user(prompt)],
        temperature=0.4,
        max_tokens=MAX_TOKENS,
    ):
        if chunk.delta:
            if primer_token_ms is None:
                primer_token_ms = (time.perf_counter() - inicio) * 1000
            print(chunk.delta, end="", flush=True)
            piezas.append(chunk.delta)
        elif chunk.is_final:
            final = chunk

    texto = "".join(piezas)
    total_ms = (time.perf_counter() - inicio) * 1000
    METRICAS.registrar(
        "respuesta",
        ModelResponse(
            content=texto,
            provider=nombre_proveedor(llm),
            model=final.model if final else "?",
            usage=final.usage if final and final.usage else Usage(),
            latency_ms=total_ms,
        ),
    )
    print(
        f"\n\n-> primer token en {primer_token_ms or 0:.0f}ms, total {total_ms:.0f}ms, "
        f"tokens={final.usage.total_tokens if final and final.usage else 0} "
        f"(el usage llega en el chunk final, no hay que contarlo a mano)"
    )
    return texto


# ======================================================================
# PASO 5 - Conversacion multivuelta con historial
# ======================================================================


async def paso_5_seguimiento(llm: Any, resultado: Resultado, respuesta_previa: str) -> None:
    paso(5, "el cliente responde: segunda vuelta con historial")
    replica = (
        "Gracias, pero necesito una fecha concreta. Si no esta resuelto el viernes "
        "quiero la devolucion del dinero, no un cambio."
    )
    print(f"[cliente] {replica}\n")

    # El historial se arma con los mismos objetos tipados; `as_message()` reinyecta
    # la respuesta del modelo sin pasar por diccionarios.
    historial = [
        ChatMessage.system(SISTEMA_RESPUESTA),
        ChatMessage.user(f"Ticket {resultado.ticket.id}: {resultado.ticket.texto}"),
        ChatMessage.assistant(respuesta_previa),
        ChatMessage.user(replica),
    ]
    segunda = await llm.generate_safe(historial, temperature=0.4, max_tokens=MAX_TOKENS)
    METRICAS.registrar("seguimiento", segunda)

    if isinstance(segunda, ErrorResponse):
        print(f"-> no se pudo responder: {segunda.error_type}")
        return

    print(f"[agente] {segunda.content.strip()}")
    conversacion = Conversation.coerce(historial)
    print(
        f"\n-> {len(conversacion.messages)} mensajes, "
        f"{len(conversacion.turns)} turnos user/assistant, "
        f"system extraido aparte ({len(conversacion.system_prompt or '')} chars). "
        f"latencia={segunda.latency_ms:.0f}ms intentos={segunda.attempts}"
    )


# ======================================================================
# PASO 6 - Resiliencia: el proveedor falla y la mesa sigue abierta
# ======================================================================


async def paso_6_resiliencia(proveedor: str) -> None:
    paso(6, "el proveedor falla (key invalida) y el sistema degrada sin romperse")
    print(
        "Esta parte usa una API key falsa A PROPOSITO: el WARNING que sigue\n"
        "es el error esperado, capturado y convertido en dato.\n"
    )

    # max_attempts=1: una key invalida no es transitoria, reintentar no sirve.
    async with abrir_cliente(
        proveedor, api_key="sk-clave-invalida-a-proposito", retry=RetryConfig(max_attempts=1)
    ) as caido:
        resultado = await caido.generate_safe(
            [ChatMessage.system(SISTEMA_TRIAGE), ChatMessage.user("Ticket T-1047: no me llego nada")]
        )
        METRICAS.registrar("resiliencia", resultado)
        if isinstance(resultado, ErrorResponse):
            print(
                f"error capturado -> tipo={resultado.error_type} status={resultado.status_code} "
                f"retryable={resultado.retryable} intentos={resultado.attempts}"
            )
            print(f"mensaje: {resultado.message[:140]}")

        # La variante con excepcion tipada, para quien prefiera try/except.
        try:
            await caido.generate("hola")
        except LLMError as exc:
            print(f"excepcion tipada -> {type(exc).__name__}: {str(exc)[:110]}")

    # Plan B: el mismo ticket, contra un proveedor que si responde.
    respaldo = elegir_respaldo(proveedor)
    print(f"\nfailover -> reintentando el ticket con '{respaldo}'")
    async with abrir_cliente(respaldo) as sano:
        reintento = await sano.generate_safe(
            [ChatMessage.system(SISTEMA_TRIAGE), ChatMessage.user("Ticket T-1047: no me llego nada")],
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        )
        METRICAS.registrar("resiliencia", reintento)
        if isinstance(reintento, ErrorResponse):
            print(f"-> el respaldo tambien fallo: {reintento.error_type}")
        else:
            print(f"-> respondio {reintento.provider}/{reintento.model}: {reintento.content[:120]}")

    print("\n-> ningun ticket se pierde: el fallo es un estado del sistema, no un crash")


def elegir_respaldo(proveedor: str) -> str:
    """Otro proveedor con key en el entorno; si no hay, el simulado."""
    if proveedor == "simulado":
        return "simulado"
    otros = [p.value for p in available_providers() if p.value != proveedor]
    return otros[0] if otros else proveedor


# ======================================================================
# PASO 7 - Reporte
# ======================================================================


def paso_7_reporte(resultados: list[Resultado]) -> None:
    paso(7, "reporte de la corrida")
    clasificados = sum(r.ok for r in resultados)
    escalados = sum(1 for r in resultados if r.triage and r.triage.requiere_humano)
    automaticos = clasificados - escalados

    print(f"tickets ingresados      : {len(resultados)}")
    print(f"clasificados por el LLM : {clasificados}")
    print(f"derivados a un humano   : {escalados}")
    print(f"resueltos por el sistema: {automaticos}")
    print(f"fallos manejados        : {len(resultados) - clasificados}")
    print()
    print(f"llamadas al proveedor   : {METRICAS.llamadas} ({METRICAS.fallidas} con error)")
    print(f"llamadas por paso       : {METRICAS.por_paso}")
    print(
        f"tokens                  : {METRICAS.total_tokens} "
        f"({METRICAS.input_tokens} in / {METRICAS.output_tokens} out)"
    )
    print(f"latencia acumulada      : {METRICAS.latencia_total_ms:.0f}ms")
    print(f"reintentos consumidos   : {METRICAS.reintentos}")
    print("\n-> todas estas metricas salen de ModelResponse/ErrorResponse, sin instrumentar aparte")


# ======================================================================
# Orquestacion
# ======================================================================


async def correr(proveedor: str, pasos: set[int]) -> int:
    titulo(f"CASO PRACTICO - mesa de ayuda | proveedor: {proveedor}")
    print(
        "Flujo: ingesta -> triage en lote -> validacion -> priorizacion -> "
        "respuesta en streaming\n         -> seguimiento -> resiliencia -> reporte"
    )

    resultados: list[Resultado] = []
    ordenados: list[Resultado] = []
    respuesta = ""

    async with abrir_cliente(proveedor, max_tokens=MAX_TOKENS) as llm:
        print(f"modelo: {llm.config.model}")

        if {1, 2} & pasos:
            resultados = await paso_1_y_2_triage(llm)
        if 3 in pasos and resultados:
            ordenados = paso_3_priorizar(resultados)
        if {4, 5} & pasos:
            # Si se pidio un paso suelto, se trabaja igual sobre el ticket critico.
            objetivo = next(
                (r for r in (ordenados or resultados) if r.ok),
                Resultado(TICKETS[2]),
            )
            if 4 in pasos:
                respuesta = await paso_4_respuesta_streaming(llm, objetivo)
            if 5 in pasos and respuesta:
                await paso_5_seguimiento(llm, objetivo, respuesta)

    if 6 in pasos:
        await paso_6_resiliencia(proveedor)
    if 7 in pasos and resultados:
        paso_7_reporte(resultados)

    fallos = sum(1 for r in resultados if not r.ok)
    titulo("fin del caso")
    print(
        "caso completo sin errores no manejados"
        if fallos == 0
        else f"{fallos} ticket(s) quedaron en la cola humana (comportamiento esperado)"
    )
    return 0


def elegir_proveedor(argumentos: list[str]) -> str | None:
    """Proveedor pedido por linea de comandos, o el primero con key en .env."""
    if "--simulado" in argumentos:
        return "simulado"
    pedidos = [a.lower() for a in argumentos if not a.startswith("-")]
    disponibles = available_providers()
    if pedidos:
        pedido = pedidos[0]
        if pedido not in {p.value for p in Provider}:
            print(f"proveedor desconocido: {pedido!r}. Opciones: openai, anthropic, gemini")
            return None
        if pedido not in {p.value for p in disponibles}:
            print(f"'{pedido}' no tiene API key en el entorno. Completa .env o usa --simulado")
            return None
        return pedido
    if not disponibles:
        print(
            "No hay ninguna API key en el entorno.\n"
            "  copy .env.example .env   y completa OPENAI_API_KEY / ANTHROPIC_API_KEY / "
            "GEMINI_API_KEY\n"
            "  o corre la demo sin red:  python caso_practico.py --simulado"
        )
        return None
    return disponibles[0].value


def main() -> int:
    load_dotenv()
    argumentos = sys.argv[1:]

    pasos = set(range(1, 8))
    if "--paso" in argumentos:
        indice = argumentos.index("--paso")
        try:
            pedido = int(argumentos[indice + 1])
        except (IndexError, ValueError):
            print("uso: --paso N  (N entre 1 y 7)")
            return 2
        # Los pasos 3 a 5 dependen del triage, asi que arrastran los previos.
        pasos = {1, 2, pedido} if pedido in {3, 4, 5} else {pedido}
        if pedido in {4, 5}:
            pasos |= {3}
        argumentos = argumentos[:indice] + argumentos[indice + 2 :]

    proveedor = elegir_proveedor(argumentos)
    if proveedor is None:
        return 1
    return asyncio.run(correr(proveedor, pasos))


if __name__ == "__main__":
    raise SystemExit(main())
