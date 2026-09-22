"""Validacion del pipeline sin red, sin API keys y sin SDKs de proveedores.

Reemplaza la capa `model.with_structured_output(..., include_raw=True)` por un
runnable falso que devuelve exactamente la misma forma
(`{"raw", "parsed", "parsing_error"}`). Eso permite probar lo que realmente
importa y no se puede testear contra la API real de forma determinista:

* las restricciones del esquema Pydantic,
* la deteccion de `finish_reason` truncado,
* cuantas veces reintenta `.with_retry()` y ante que excepciones,
* la recuperacion cuando el segundo intento sale bien.

    python validate_offline.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from chain import (
    PROMPT,
    ExtraccionIncompletaError,
    RespuestaTruncadaError,
    RespuestaVaciaError,
    SalidaNoValidaError,
    con_resiliencia,
    process_batch,
    process_text,
)
from schemas import ExtraccionTecnica, NivelDeCriticidad

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Los WARNING que salen durante la corrida son parte de lo que se esta
# probando: cada uno es un reintento que el pipeline decidio hacer. Van a
# stderr, asi que `python validate_offline.py 2>/dev/null` deja solo los checks.
logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(name)s: %(message)s")

fallas: list[str] = []


def check(condicion: bool, descripcion: str) -> None:
    print(f"  {'ok   ' if condicion else 'FALLA'} {descripcion}")
    if not condicion:
        fallas.append(descripcion)


def titulo(texto: str) -> None:
    print(f"\n{texto}")


VALIDO = {
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": (
        "API con cache en Redis y persistencia en PostgreSQL; "
        "cuello de botella en conexiones concurrentes."
    ),
}


# ----------------------------------------------------------------------
# Dobles de prueba
# ----------------------------------------------------------------------
def salida_ok(**overrides: Any) -> dict[str, Any]:
    """Lo que devuelve `with_structured_output(include_raw=True)` cuando todo va bien."""
    return {
        "raw": AIMessage(
            content="",
            response_metadata={"finish_reason": "stop"},
            usage_metadata={"input_tokens": 120, "output_tokens": 48, "total_tokens": 168},
        ),
        "parsed": ExtraccionTecnica(**{**VALIDO, **overrides}),
        "parsing_error": None,
    }


def salida_truncada(clave: str = "finish_reason", valor: str = "length") -> dict[str, Any]:
    """Respuesta cortada: el JSON quedo a medias y no hay objeto parseado."""
    return {
        "raw": AIMessage(content='{"tecnologias": ["Fast', response_metadata={clave: valor}),
        "parsed": None,
        "parsing_error": ValueError("Unterminated string starting at line 1"),
    }


def error_de_esquema() -> ValidationError:
    """Un `ValidationError` real, el que produce una lista de tecnologias vacia."""
    try:
        ExtraccionTecnica(**{**VALIDO, "tecnologias": []})
    except ValidationError as exc:
        return exc
    raise AssertionError("se esperaba un ValidationError")


def salida_invalida() -> dict[str, Any]:
    return {
        "raw": AIMessage(content="", response_metadata={"finish_reason": "stop"}),
        "parsed": None,
        "parsing_error": error_de_esquema(),
    }


def salida_sin_herramienta() -> dict[str, Any]:
    """El modelo contesto en prosa en lugar de llamar a la herramienta."""
    return {
        "raw": AIMessage(
            content="No estoy seguro de que tecnologias menciona el texto.",
            response_metadata={"finish_reason": "stop"},
        ),
        "parsed": None,
        "parsing_error": None,
    }


class ModeloFalso:
    """Devuelve una respuesta por intento y cuenta cuantas veces lo llamaron.

    La ultima respuesta de la lista se repite, asi `[truncada()]` simula un
    proveedor que falla siempre y `[truncada(), ok()]` uno que se recupera.
    """

    def __init__(self, *respuestas: dict[str, Any]) -> None:
        self.respuestas = list(respuestas)
        self.llamadas = 0

    def __call__(self, _entrada: Any) -> dict[str, Any]:
        indice = min(self.llamadas, len(self.respuestas) - 1)
        self.llamadas += 1
        return self.respuestas[indice]

    def runnable(self) -> Runnable[Any, dict[str, Any]]:
        return RunnableLambda(self.__call__, name="modelo_falso")


def cadena_falsa(modelo: ModeloFalso, *, max_intentos: int = 3) -> Runnable[dict, Any]:
    """El pipeline real con la unica pieza que necesita red reemplazada."""
    return PROMPT | con_resiliencia(
        modelo.runnable(), max_intentos=max_intentos, espera_exponencial=False
    )


# ----------------------------------------------------------------------
# 1. Esquema
# ----------------------------------------------------------------------
def probar_esquema() -> None:
    titulo("1. Esquema Pydantic (schemas.py)")

    valido = ExtraccionTecnica(**VALIDO)
    check(valido.tecnologias == ["FastAPI", "Redis", "PostgreSQL"], "instancia valida")
    check(valido.nivel_de_criticidad is NivelDeCriticidad.ALTA, "el enum se resuelve desde el str")
    check(
        valido.model_dump()["nivel_de_criticidad"] == "alta",
        "serializa el nivel como string plano",
    )

    normalizado = ExtraccionTecnica(
        **{**VALIDO, "tecnologias": ["  Redis ", "redis", "PostgreSQL", "Fast   API"]}
    )
    check(
        normalizado.tecnologias == ["Redis", "PostgreSQL", "Fast API"],
        "deduplica sin distinguir mayusculas y colapsa espacios",
    )

    casos: list[tuple[str, dict[str, Any]]] = [
        ("rechaza la lista de tecnologias vacia", {"tecnologias": []}),
        ("rechaza tecnologias que quedan vacias al limpiar", {"tecnologias": ["  ", ""]}),
        ("rechaza un nivel de criticidad inventado", {"nivel_de_criticidad": "critica"}),
        ("rechaza un resumen demasiado corto", {"resumen_tecnico": "poco"}),
        (
            "rechaza un resumen que no nombra ninguna tecnologia",
            {"resumen_tecnico": "El sistema presenta algunos problemas de rendimiento."},
        ),
        ("rechaza campos que no estan en el contrato", {"severidad": "alta"}),
    ]
    for descripcion, override in casos:
        try:
            ExtraccionTecnica(**{**VALIDO, **override})
        except ValidationError:
            check(True, descripcion)
        else:
            check(False, descripcion)


# ----------------------------------------------------------------------
# 2. Prompt template
# ----------------------------------------------------------------------
def probar_prompt() -> None:
    titulo("2. Prompt template (ChatPromptTemplate, sin f-strings)")

    check(PROMPT.input_variables == ["texto"], "la unica variable requerida es 'texto'")

    mensajes = PROMPT.format_messages(texto="Redis se quedo sin memoria")
    check(len(mensajes) == 2, "genera un mensaje system y uno human")
    check(mensajes[0].type == "system" and mensajes[1].type == "human", "roles correctos")
    check(
        "llama a la herramienta `ExtraccionTecnica`" in mensajes[0].content,
        "las instrucciones de formato entran por variable, no concatenadas",
    )
    check(
        "<texto>\nRedis se quedo sin memoria\n</texto>" in mensajes[1].content,
        "el texto de entrada va delimitado dentro del mensaje human",
    )

    otro = PROMPT.partial(instrucciones_formato="RESPONDE EN INGLES")
    check(
        "RESPONDE EN INGLES" in otro.format_messages(texto="x")[0].content,
        "las instrucciones de formato se pueden sobrescribir sin tocar la plantilla",
    )


# ----------------------------------------------------------------------
# 3. Validacion y reintentos
# ----------------------------------------------------------------------
async def probar_resiliencia() -> None:
    titulo("3. Validacion de la salida y reintentos (.with_retry)")

    modelo = ModeloFalso(salida_ok())
    resultado = await cadena_falsa(modelo).ainvoke({"texto": "un log cualquiera"})
    check(isinstance(resultado, ExtraccionTecnica), "camino feliz: devuelve el objeto validado")
    check(modelo.llamadas == 1, "camino feliz: una sola llamada al modelo")

    modelo = ModeloFalso(salida_truncada())
    try:
        await cadena_falsa(modelo).ainvoke({"texto": "x"})
        check(False, "respuesta truncada: deberia fallar")
    except RespuestaTruncadaError:
        check(True, "detecta finish_reason='length' antes de intentar parsear")
    check(modelo.llamadas == 3, f"reintenta hasta 3 veces (llamadas={modelo.llamadas})")

    modelo = ModeloFalso(salida_truncada("stop_reason", "max_tokens"))
    try:
        await cadena_falsa(modelo, max_intentos=1).ainvoke({"texto": "x"})
        check(False, "stop_reason de Anthropic: deberia fallar")
    except RespuestaTruncadaError:
        check(True, "detecta el stop_reason='max_tokens' de Anthropic")
    check(modelo.llamadas == 1, "max_intentos=1 no reintenta")

    modelo = ModeloFalso(salida_invalida())
    try:
        await cadena_falsa(modelo).ainvoke({"texto": "x"})
        check(False, "salida fuera de contrato: deberia fallar")
    except ExtraccionIncompletaError as exc:
        check("tecnologias" in str(exc), "el error nombra el campo que fallo la validacion")
        check(isinstance(exc.__cause__, ValidationError), "conserva el ValidationError original")
    check(modelo.llamadas == 3, "reintenta tambien ante un JSON fuera de contrato")

    modelo = ModeloFalso(salida_sin_herramienta())
    try:
        await cadena_falsa(modelo, max_intentos=2).ainvoke({"texto": "x"})
        check(False, "sin tool call: deberia fallar")
    except RespuestaVaciaError as exc:
        check("No estoy seguro" in str(exc), "muestra la respuesta en prosa del modelo")
    check(modelo.llamadas == 2, "reintenta cuando el modelo no llama a la herramienta")

    # Lo importante del reintento: que el segundo intento sirva de algo.
    modelo = ModeloFalso(salida_truncada(), salida_invalida(), salida_ok())
    resultado = await cadena_falsa(modelo).ainvoke({"texto": "x"})
    check(
        isinstance(resultado, ExtraccionTecnica) and modelo.llamadas == 3,
        "se recupera en el tercer intento tras dos fallas distintas",
    )

    # Y que no reintente lo que no tiene sentido reintentar.
    def explota(_entrada: Any) -> dict[str, Any]:
        explota.llamadas += 1  # type: ignore[attr-defined]
        raise PermissionError("401 invalid api key")

    explota.llamadas = 0  # type: ignore[attr-defined]
    cadena = PROMPT | con_resiliencia(RunnableLambda(explota), espera_exponencial=False)
    try:
        await cadena.ainvoke({"texto": "x"})
        check(False, "un error no contemplado deberia propagarse")
    except PermissionError:
        check(True, "un error de autenticacion se propaga tal cual")
    check(explota.llamadas == 1, "no reintenta ante errores que no son de formato")  # type: ignore[attr-defined]


# ----------------------------------------------------------------------
# 4. API asincrona
# ----------------------------------------------------------------------
async def probar_api_async() -> None:
    titulo("4. process_text / process_batch")

    modelo = ModeloFalso(salida_ok())
    resultado = await process_text("  un log con espacios  ", cadena=cadena_falsa(modelo))
    check(isinstance(resultado, ExtraccionTecnica), "process_text devuelve el objeto validado")

    try:
        await process_text("   ", cadena=cadena_falsa(ModeloFalso(salida_ok())))
        check(False, "texto vacio: deberia fallar")
    except ValueError:
        check(True, "process_text rechaza un texto vacio sin llamar al modelo")

    modelo = ModeloFalso(salida_ok())
    lote = await process_batch(["texto uno", "texto dos"], cadena=cadena_falsa(modelo))
    check(len(lote) == 2, "process_batch devuelve un resultado por texto")
    check(all(isinstance(r, ExtraccionTecnica) for r in lote), "process_batch valida cada salida")

    modelo = ModeloFalso(salida_invalida())
    lote = await process_batch(["a", "b"], cadena=cadena_falsa(modelo, max_intentos=1))
    check(
        all(isinstance(r, SalidaNoValidaError) for r in lote),
        "process_batch devuelve los errores como dato, sin cortar el lote",
    )

    # Concurrencia real: dos textos procesados en paralelo.
    resultados = await asyncio.gather(
        process_text("uno", cadena=cadena_falsa(ModeloFalso(salida_ok()))),
        process_text("dos", cadena=cadena_falsa(ModeloFalso(salida_ok()))),
    )
    check(len(resultados) == 2, "asyncio.gather sobre process_text funciona")


# ----------------------------------------------------------------------
async def main() -> int:
    print("=" * 72)
    print("Validacion offline del pipeline de extraccion (sin red, sin API keys)")
    print("=" * 72)

    probar_esquema()
    probar_prompt()
    await probar_resiliencia()
    await probar_api_async()

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
