"""Agente de Analisis/Computo: procesa los datos que entrego el investigador.

Herramientas acotadas a su rol (no accede a los documentos; trabaja sobre el texto que recibe):

* `calcular`: aritmetica exacta. Los LLM fallan en cuentas; una herramienta no.
* `extraer_cifras`: saca montos, porcentajes, horas y plazos de un texto, para no copiarlos "a ojo".
* `analizar_sentimiento`: tono (positivo/negativo/neutro) por lexico en espanol, deterministico.
"""

from __future__ import annotations

import ast
import operator
import re
import unicodedata
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool

from app.agents.common import NodoEspecialista, crear_nodo_especialista
from app.state import EstadoOrquestador

__all__ = [
    "HERRAMIENTAS_ANALISTA",
    "PROMPT_ANALISTA",
    "analizar_sentimiento",
    "calcular",
    "crear_analista",
    "evaluar_expresion",
    "extraer_cifras",
]

PROMPT_ANALISTA = """\
Sos el ANALISTA del orquestador del Hotel Bahia Serena. Recibis datos ya recopilados por el
investigador y los procesas: calculos, extraccion de cifras o valoracion del tono.

Reglas:
- Trabajas SOLO con los datos que te pasan. No tenes acceso a los documentos ni a internet.
  Si falta un dato para calcular, decilo explicitamente en vez de suponerlo.
- Toda cuenta se hace con `calcular`; nunca de cabeza. Para sacar numeros de un texto usa
  `extraer_cifras`; para el tono de un texto, `analizar_sentimiento`.
- Mostra el procedimiento: la formula con los numeros reemplazados y el resultado.
- Indica de que dato (y fuente, si la traen) sale cada numero que uses.
- Si recibis una consigna de REFINAMIENTO, corregi tu analisis anterior en lugar de repetirlo.
- Respuesta breve: resultado primero, despues el desarrollo.
"""

# ----------------------------------------------------------------------
# calcular: evaluador aritmetico seguro (nunca `eval`)
# ----------------------------------------------------------------------
_BINARIOS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARIOS: dict[type[ast.unaryop], Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_EXPONENTE = 100


def _evaluar(nodo: ast.AST) -> float:
    if isinstance(nodo, ast.Expression):
        return _evaluar(nodo.body)
    if isinstance(nodo, ast.Constant) and isinstance(nodo.value, int | float) and not isinstance(nodo.value, bool):
        return nodo.value
    if isinstance(nodo, ast.BinOp) and type(nodo.op) in _BINARIOS:
        izq, der = _evaluar(nodo.left), _evaluar(nodo.right)
        if isinstance(nodo.op, ast.Pow) and abs(der) > _MAX_EXPONENTE:
            raise ValueError("exponente demasiado grande")
        return _BINARIOS[type(nodo.op)](izq, der)
    if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in _UNARIOS:
        return _UNARIOS[type(nodo.op)](_evaluar(nodo.operand))
    raise ValueError("solo se admiten numeros, + - * / // % ** y parentesis")


def evaluar_expresion(expresion: str) -> float:
    """Evalua aritmetica pura. Lanza ValueError ante cualquier otra cosa (nombres, llamadas, etc.)."""
    limpia = expresion.strip().replace(",", ".").replace("^", "**")
    if not limpia:
        raise ValueError("la expresion esta vacia")
    try:
        return _evaluar(ast.parse(limpia, mode="eval"))
    except SyntaxError as exc:
        raise ValueError(f"expresion invalida: {expresion!r}") from exc
    except ZeroDivisionError as exc:
        raise ValueError("division por cero") from exc


@tool(parse_docstring=True)
def calcular(expresion: str) -> dict[str, Any]:
    """Calcula una expresion aritmetica exacta.

    Usala para CUALQUIER cuenta: totales de estadia, porcentajes, penalidades, promedios,
    diferencias. Admite numeros, + - * / // % ** y parentesis. Los porcentajes se escriben
    como multiplicacion: el 50% de 4 noches a 150 es "0.5 * 4 * 150".

    Devuelve {"expresion", "resultado"}. Si la expresion no es valida devuelve {"error": ...}:
    corregila y reintenta.

    Args:
        expresion: Aritmetica pura, por ejemplo "(4 * 150) * 0.5". Sin variables ni funciones.
    """
    try:
        resultado = evaluar_expresion(expresion)
    except ValueError as exc:
        return {"error": str(exc), "expresion": expresion}
    return {"expresion": expresion, "resultado": round(resultado, 4)}


# ----------------------------------------------------------------------
# extraer_cifras
# ----------------------------------------------------------------------
_PATRONES_CIFRAS: dict[str, re.Pattern[str]] = {
    "porcentajes": re.compile(r"\d+(?:[.,]\d+)?\s?%"),
    "montos": re.compile(r"(?:US?D|U\$S|\$)\s?\d[\d.,]*|\d[\d.,]*\s?(?:USD|dolares|pesos)", re.IGNORECASE),
    "horarios": re.compile(r"\b\d{1,2}:\d{2}\s?(?:hs)?", re.IGNORECASE),
    "plazos": re.compile(r"\b\d+\s+(?:dias?|noches?|horas?|semanas?|meses|anos?)(?:\s+habiles)?\b", re.IGNORECASE),
}


def _sin_tildes(texto: str) -> str:
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


@tool(parse_docstring=True)
def extraer_cifras(texto: str) -> dict[str, list[str]]:
    """Extrae porcentajes, montos, horarios y plazos de un texto.

    Usala cuando tengas un texto con datos del hotel (los del investigador) y necesites los
    numeros exactos antes de calcular, en vez de transcribirlos a mano.

    Devuelve {"porcentajes": [...], "montos": [...], "horarios": [...], "plazos": [...]}; una
    lista vacia significa que no se encontro nada de ese tipo.

    Args:
        texto: Texto del que sacar las cifras, tal como lo entrego el investigador.
    """
    plano = _sin_tildes(texto)
    return {tipo: [m.group().strip().rstrip(".,") for m in rx.finditer(plano)] for tipo, rx in _PATRONES_CIFRAS.items()}


# ----------------------------------------------------------------------
# analizar_sentimiento
# ----------------------------------------------------------------------
_POSITIVAS = frozenset(
    "gratis gratuito gratuita cortesia incluido incluida incluye sin cargo disponible bienvenido bienvenida "
    "comodidad comodo comoda flexible beneficio confirmado confirmada seguro segura permitido permitida "
    "reembolso reembolsable favorable excelente".split()
)
_NEGATIVAS = frozenset(
    "prohibido prohibida prohibe penalidad cargo multa sancion restriccion restringido restringida "
    "obligatorio obligatoria rechazo rechazada cancelacion perdida danos queja "
    "incumplimiento denegado denegada riesgo peligro".split()
)


@tool(parse_docstring=True)
def analizar_sentimiento(texto: str) -> dict[str, Any]:
    """Valora el tono de un texto: positivo (favorable al huesped), negativo (restrictivo) o neutro.

    Usala cuando se pida evaluar si una politica o un comentario es restrictivo, amable,
    favorable, etc. Es un analisis por lexico en espanol: sirve para comparar textos entre si,
    no reemplaza una lectura fina.

    Devuelve {"etiqueta", "puntaje" (-1 a 1), "positivas", "negativas"} con las palabras que
    pesaron en cada sentido, para poder justificar la conclusion.

    Args:
        texto: Texto a valorar (una politica, un comentario, un parrafo).
    """
    palabras = re.findall(r"[a-z]+", _sin_tildes(texto.lower()))
    positivas = [p for p in palabras if p in _POSITIVAS]
    negativas = [p for p in palabras if p in _NEGATIVAS]
    total = len(positivas) + len(negativas)
    puntaje = round((len(positivas) - len(negativas)) / total, 3) if total else 0.0
    etiqueta = "positivo" if puntaje > 0.2 else "negativo" if puntaje < -0.2 else "neutro"
    return {
        "etiqueta": etiqueta,
        "puntaje": puntaje,
        "positivas": sorted(set(positivas)),
        "negativas": sorted(set(negativas)),
    }


HERRAMIENTAS_ANALISTA: list[BaseTool] = [calcular, extraer_cifras, analizar_sentimiento]


def _entrada(estado: EstadoOrquestador) -> str:
    """Consigna + SOLO el hallazgo del investigador (no sus busquedas ni el historial completo)."""
    hallazgos = estado.get("hallazgos", {})
    partes = [
        f"Pedido original del usuario (solo contexto): {estado['pregunta']}",
        f"Tu consigna ahora: {estado['instruccion']}",
    ]
    if datos := hallazgos.get("investigador"):
        partes.append(f"Datos del investigador:\n{datos['contenido']}")
    else:
        partes.append("Datos del investigador: (todavia no hay)")
    if previo := hallazgos.get("analista"):
        partes.append(f"Tu analisis anterior, a refinar segun la consigna:\n{previo['contenido']}")
    return "\n\n".join(partes)


def crear_analista(llm: BaseChatModel) -> NodoEspecialista:
    return crear_nodo_especialista("analista", llm, HERRAMIENTAS_ANALISTA, PROMPT_ANALISTA, _entrada)
