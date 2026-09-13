"""Cadena LCEL de extraccion: prompt | modelo estructurado | validacion, con reintentos.

El pipeline completo es una sola expresion::

    PROMPT | con_resiliencia(modelo.with_structured_output(ExtraccionTecnica, include_raw=True))

y cada eslabon tiene un motivo:

* `PROMPT` es un `ChatPromptTemplate`, no una f-string. LangChain gestiona las
  variables de entrada, lo que permite reusar la cadena, cachearla y trazarla.
* `with_structured_output(...)` convierte el esquema Pydantic en la definicion
  de herramienta que recibe el modelo: el JSON ya llega tipado.
* `include_raw=True` es la parte que no suele hacerse: sin el, un JSON cortado
  a la mitad se convierte en una excepcion opaca y perdemos el mensaje crudo.
  Con el recibimos `{"raw", "parsed", "parsing_error"}` y podemos mirar el
  `finish_reason` *antes* de confiar en el objeto.
* `_validar` traduce esos tres casos a excepciones propias, y `.with_retry()`
  vuelve a llamar al modelo solo ante esas excepciones.

Uso::

    import asyncio
    from chain import process_text

    print(asyncio.run(process_text("El worker de Celery satura Redis...")))
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from schemas import ExtraccionTecnica

__all__ = [
    "MAX_INTENTOS",
    "MAX_TOKENS",
    "PROMPT",
    "ExtraccionIncompletaError",
    "RespuestaTruncadaError",
    "RespuestaVaciaError",
    "SalidaNoValidaError",
    "cadena_por_defecto",
    "con_resiliencia",
    "construir_cadena",
    "crear_modelo",
    "process_batch",
    "process_text",
    "proveedores_disponibles",
]

log = logging.getLogger("clase_2.chain")

#: Cuantas veces se llama al modelo, como maximo, por texto de entrada.
MAX_INTENTOS = 3

#: Techo de tokens de salida por llamada. Suena generoso para un JSON de tres
#: campos, pero los modelos actuales razonan antes de responder y ese
#: razonamiento consume el mismo presupuesto: con 1024 el truncado
#: (finish_reason='max_tokens') deja de ser un caso de borde y pasa a ser el
#: caso normal. Medido con gemini-3.6-flash: 550-900 tokens por extraccion.
MAX_TOKENS = 2048


# ----------------------------------------------------------------------
# Errores: lo que dispara (y lo que no) un reintento
# ----------------------------------------------------------------------
class SalidaNoValidaError(RuntimeError):
    """Base: el modelo respondio, pero la respuesta no sirve. Se reintenta."""


class RespuestaTruncadaError(SalidaNoValidaError):
    """El modelo corto por limite de tokens: el JSON esta incompleto."""


class ExtraccionIncompletaError(SalidaNoValidaError):
    """El JSON llego entero pero no paso el contrato Pydantic."""


class RespuestaVaciaError(SalidaNoValidaError):
    """El modelo no llamo a la herramienta: no hay objeto que validar."""


# ----------------------------------------------------------------------
# 2. Prompt template modular
# ----------------------------------------------------------------------
SISTEMA = (
    "Sos un analista de sistemas. Extraes entidades tecnicas de texto libre "
    "(logs de error, descripciones de arquitectura, reportes de incidentes) y "
    "devolves siempre datos estructurados.\n\n"
    "Reglas:\n"
    "- Trabaja solo con lo que dice el texto: no agregues tecnologias que no "
    "aparezcan ni infieras un stack completo a partir de una sola pista.\n"
    "- Normaliza los nombres a su forma oficial (postgres -> PostgreSQL).\n"
    "- Si el texto no menciona ninguna tecnologia concreta, no inventes una lista.\n\n"
    "{instrucciones_formato}"
)

#: Se inyecta como variable del prompt, no concatenada: cambiar el formato de
#: salida (u otro idioma) no obliga a tocar la plantilla del sistema.
INSTRUCCIONES_FORMATO = (
    "Formato de salida: llama a la herramienta `ExtraccionTecnica` con los tres "
    "campos completos.\n"
    "- tecnologias: lista sin duplicados, al menos un elemento.\n"
    "- nivel_de_criticidad: exactamente 'baja', 'media' o 'alta'.\n"
    "- resumen_tecnico: una o dos oraciones (entre 20 y 400 caracteres) que "
    "nombren al menos una de las tecnologias listadas.\n"
    "No escribas texto fuera de la herramienta."
)

HUMANO = "Extrae las entidades tecnicas del siguiente texto:\n\n<texto>\n{texto}\n</texto>"

#: Variable de entrada: `texto`. `instrucciones_formato` queda con un valor por
#: defecto via `.partial()`, pero se puede sobrescribir al construir la cadena.
PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [("system", SISTEMA), ("human", HUMANO)]
).partial(instrucciones_formato=INSTRUCCIONES_FORMATO)


# ----------------------------------------------------------------------
# Modelo: misma idea que el factory del Modulo 1, ahora sobre LangChain
# ----------------------------------------------------------------------
#: proveedor -> (paquete, clase, envs de la key, modelo por defecto).
#: Mismos nombres y misma variable `LLM_PROVIDER` que el factory del Modulo 1
#: (`llm_client/factory.py` de la entrega anterior), ahora sobre los chat models
#: de LangChain. El import es perezoso: usar Gemini no obliga a instalar OpenAI.
_PROVEEDORES: dict[str, tuple[str, str, tuple[str, ...], str]] = {
    "openai": ("langchain_openai", "ChatOpenAI", ("OPENAI_API_KEY",), "gpt-4.1-mini"),
    "anthropic": (
        "langchain_anthropic",
        "ChatAnthropic",
        ("ANTHROPIC_API_KEY",),
        "claude-sonnet-5",
    ),
    "gemini": (
        "langchain_google_genai",
        "ChatGoogleGenerativeAI",
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "gemini-3.6-flash",
    ),
}
#: alias: LangChain llama "google" a lo que el Modulo 1 llamaba "gemini".
_ALIAS = {"google": "gemini", "google_genai": "gemini", "openai_chat": "openai"}


def proveedores_disponibles() -> list[str]:
    """Los proveedores que tienen su API key en el entorno."""
    return [
        nombre
        for nombre, (_, _, envs, _) in _PROVEEDORES.items()
        if any(os.getenv(env) for env in envs)
    ]


def crear_modelo(
    proveedor: str | None = None,
    *,
    modelo: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    **kwargs: Any,
) -> BaseChatModel:
    """Instancia el chat model del proveedor pedido.

    Resuelve cual usar en el mismo orden que el `AsyncLLMManager` del Modulo 1:
    el argumento, despues `LLM_PROVIDER` y, si no hay ninguno, el primer
    proveedor que tenga su API key en el entorno.

    `temperature=0` porque esto es extraccion, no redaccion: para el mismo
    texto queremos la misma salida.
    """
    pedido = proveedor or os.getenv("LLM_PROVIDER") or next(iter(proveedores_disponibles()), "")
    elegido = _ALIAS.get(pedido.lower(), pedido.lower())
    if elegido not in _PROVEEDORES:
        validos = ", ".join(_PROVEEDORES)
        con_key = ", ".join(proveedores_disponibles()) or "ninguno"
        raise ValueError(
            f"proveedor desconocido: {pedido!r}. Opciones: {validos}. "
            f"Con API key en el entorno: {con_key}"
        )

    paquete, clase, envs, por_defecto = _PROVEEDORES[elegido]
    if not any(os.getenv(env) for env in envs):
        raise RuntimeError(f"falta la API key de {elegido}: defini {' o '.join(envs)} en .env")

    from importlib import import_module

    try:
        cls = getattr(import_module(paquete), clase)
    except ImportError as exc:  # pragma: no cover - depende de la instalacion
        raise RuntimeError(f"falta el paquete {paquete}: pip install {paquete}") from exc

    nombre_modelo = modelo or os.getenv("LLM_MODEL") or por_defecto
    log.info(
        "modelo: %s / %s (temperature=%s, max_tokens=%s)",
        elegido,
        nombre_modelo,
        temperature,
        max_tokens,
    )
    return cls(model=nombre_modelo, temperature=temperature, max_tokens=max_tokens, **kwargs)


# ----------------------------------------------------------------------
# 4. Resiliencia: finish_reason -> excepcion -> reintento
# ----------------------------------------------------------------------
#: Motivos de corte que significan "la respuesta quedo a medias".
_MOTIVOS_TRUNCADO = frozenset(
    {"length", "max_tokens", "max_output_tokens", "model_length", "content_filter", "safety"}
)


def _motivo_de_corte(mensaje: BaseMessage | None) -> str | None:
    """`finish_reason` normalizado, sea cual sea el proveedor.

    OpenAI y Gemini lo llaman `finish_reason`; Anthropic, `stop_reason`.
    """
    metadatos: dict[str, Any] = getattr(mensaje, "response_metadata", None) or {}
    for clave in ("finish_reason", "stop_reason", "finishReason"):
        valor = metadatos.get(clave)
        if valor:
            return str(valor).lower()
    return None


def _texto_crudo(mensaje: BaseMessage | None) -> str:
    """El contenido textual del mensaje, para poder mostrarlo en el error."""
    contenido = getattr(mensaje, "content", "")
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):  # bloques (Anthropic, Gemini)
        return " ".join(
            bloque.get("text", "") for bloque in contenido if isinstance(bloque, dict)
        ).strip()
    return str(contenido)


def _detalle_pydantic(error: BaseException) -> str:
    """Resume un `ValidationError` a una linea legible en el log."""
    if isinstance(error, ValidationError):
        return "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '<raiz>'}: {e['msg']}"
            for e in error.errors()[:4]
        )
    return f"{type(error).__name__}: {error}"


def _validar(salida: dict[str, Any]) -> ExtraccionTecnica:
    """Convierte `{"raw", "parsed", "parsing_error"}` en un objeto de confianza.

    Es el unico lugar donde el pipeline decide si hay que reintentar. Los tres
    modos de falla se separan a proposito: en los logs se distingue "el modelo
    se quedo sin tokens" de "el modelo alucino un campo".
    """
    crudo: BaseMessage | None = salida.get("raw")
    parseado = salida.get("parsed")
    error_de_parseo = salida.get("parsing_error")
    motivo = _motivo_de_corte(crudo)
    tokens = (getattr(crudo, "usage_metadata", None) or {}).get("output_tokens")

    log.debug("respuesta cruda: finish_reason=%s output_tokens=%s", motivo, tokens)

    # 1. Truncado. Se mira primero: si el modelo corto por falta de tokens, el
    #    `parsing_error` de abajo describe el sintoma y no la causa.
    if motivo in _MOTIVOS_TRUNCADO:
        log.warning("respuesta cortada por el proveedor (finish_reason=%s): reintento", motivo)
        raise RespuestaTruncadaError(
            f"el modelo corto la respuesta (finish_reason={motivo!r}); "
            "subi max_tokens o acorta el texto de entrada"
        )

    # 2. JSON completo pero fuera de contrato (lista vacia, enum invalido...).
    if error_de_parseo is not None:
        detalle = _detalle_pydantic(error_de_parseo)
        log.warning("la salida no cumple el esquema: %s", detalle)
        raise ExtraccionIncompletaError(f"salida invalida: {detalle}") from error_de_parseo

    # 3. El modelo contesto en prosa y nunca llamo a la herramienta.
    if parseado is None:
        log.warning("el modelo no devolvio estructura (finish_reason=%s)", motivo)
        raise RespuestaVaciaError(
            f"el modelo no llamo a la herramienta; respondio: {_texto_crudo(crudo)[:200]!r}"
        )

    log.info(
        "validacion ok: %d tecnologias, criticidad=%s (finish_reason=%s, output_tokens=%s)",
        len(parseado.tecnologias),
        parseado.nivel_de_criticidad.value,
        motivo,
        tokens,
    )
    return parseado


def con_resiliencia(
    estructurado: Runnable[Any, dict[str, Any]],
    *,
    max_intentos: int = MAX_INTENTOS,
    espera_exponencial: bool = True,
) -> Runnable[Any, ExtraccionTecnica]:
    """Agrega validacion + reintento a un runnable que devuelve `include_raw`.

    Esta separado de `construir_cadena` para poder probar la resiliencia sin
    red: `validate_offline.py` le pasa un runnable falso.

    Solo se reintenta ante `SalidaNoValidaError` (o `ValidationError`). Un 401 o
    un prompt mal formado fallan rapido, que es lo que se quiere; los errores de
    red y los 429 ya los reintenta el SDK del proveedor.

    `espera_exponencial=False` saca el backoff entre intentos: solo lo usan los
    tests offline, donde no hay a quien darle tiempo de recuperarse.
    """
    return (
        (estructurado | RunnableLambda(_validar, name="validar_extraccion"))
        .with_retry(
            retry_if_exception_type=(SalidaNoValidaError, ValidationError),
            stop_after_attempt=max_intentos,
            wait_exponential_jitter=espera_exponencial,
        )
        .with_config(run_name="extraccion_resiliente")
    )


# ----------------------------------------------------------------------
# 3. La cadena LCEL
# ----------------------------------------------------------------------
def construir_cadena(
    modelo: BaseChatModel | None = None,
    *,
    prompt: ChatPromptTemplate = PROMPT,
    max_intentos: int = MAX_INTENTOS,
) -> Runnable[dict[str, Any], ExtraccionTecnica]:
    """Ensambla `prompt | modelo.with_structured_output(...) | validacion+reintento`."""
    modelo = modelo if modelo is not None else crear_modelo()
    estructurado = modelo.with_structured_output(ExtraccionTecnica, include_raw=True)
    return (prompt | con_resiliencia(estructurado, max_intentos=max_intentos)).with_config(
        run_name="pipeline_extraccion"
    )


@lru_cache(maxsize=1)
def cadena_por_defecto() -> Runnable[dict[str, Any], ExtraccionTecnica]:
    """Cadena compartida del proceso: se arma una vez, no una por request.

    Es perezosa a proposito: importar `chain` no deberia exigir una API key.
    """
    return construir_cadena()


# ----------------------------------------------------------------------
# 5. Entrada asincrona
# ----------------------------------------------------------------------
async def process_text(
    text: str,
    *,
    cadena: Runnable[dict[str, Any], ExtraccionTecnica] | None = None,
) -> ExtraccionTecnica:
    """Extrae las entidades tecnicas de `text` y devuelve el objeto validado.

    Lanza `SalidaNoValidaError` si el modelo no logro producir algo valido en
    `MAX_INTENTOS` llamadas, y `ValueError` si el texto de entrada esta vacio.
    """
    if not text or not text.strip():
        raise ValueError("el texto de entrada esta vacio")

    ejecutable = cadena if cadena is not None else cadena_por_defecto()
    log.info("procesando texto de %d caracteres", len(text))
    resultado = await ejecutable.ainvoke({"texto": text.strip()})
    log.info("resultado: %s", resultado.model_dump_json())
    return resultado


async def process_batch(
    textos: list[str],
    *,
    cadena: Runnable[dict[str, Any], ExtraccionTecnica] | None = None,
    max_concurrency: int = 4,
) -> list[ExtraccionTecnica | Exception]:
    """Varios textos en paralelo. Devuelve la excepcion como dato, no la levanta.

    `abatch` con `return_exceptions=True` evita que un texto ambiguo tire abajo
    el lote entero.
    """
    ejecutable = cadena if cadena is not None else cadena_por_defecto()
    log.info("procesando lote de %d textos (max_concurrency=%d)", len(textos), max_concurrency)
    return await ejecutable.abatch(
        [{"texto": t.strip()} for t in textos],
        config={"max_concurrency": max_concurrency},
        return_exceptions=True,
    )
