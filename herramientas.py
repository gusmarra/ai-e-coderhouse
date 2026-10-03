"""Fase 1 - El contrato de herramientas.

El LLM NO ve el codigo de estas funciones: ve unicamente el nombre, la firma
(con sus tipos) y el docstring. Por eso los docstrings dicen *cuando* usar cada
herramienta, *que* devuelve y *que hacer* ante un error. Si el agente no usa una
herramienta que esperabas, el problema casi siempre esta en el docstring.

La "base de datos" es un diccionario en memoria: simula una consulta real
(es asincrona, valida el input y falla con errores utiles) sin necesitar infraestructura.
"""

from __future__ import annotations

import asyncio
import unicodedata
from typing import Any, TypedDict

from langchain_core.tools import BaseTool, tool

__all__ = ["HERRAMIENTAS", "buscar_cliente", "buscar_pedidos"]


class Pedido(TypedDict):
    id: int
    fecha: str  # ISO: ordena igual alfabetica y cronologicamente
    total: int  # pesos argentinos
    estado: str


CLIENTES: dict[int, str] = {
    101: "Ana Pérez",
    102: "Laura Gómez",
    103: "Carlos García",
    104: "María García",
}

PEDIDOS: dict[int, list[Pedido]] = {
    101: [{"id": 4001, "fecha": "2026-07-12", "total": 8900, "estado": "entregado"}],
    102: [
        {"id": 5001, "fecha": "2026-08-03", "total": 4200, "estado": "entregado"},
        {"id": 5002, "fecha": "2026-08-21", "total": 6300, "estado": "entregado"},
        {"id": 5003, "fecha": "2026-09-18", "total": 4000, "estado": "en camino"},
    ],
    103: [
        {"id": 6001, "fecha": "2026-06-30", "total": 7000, "estado": "entregado"},
        {"id": 6002, "fecha": "2026-09-02", "total": 5000, "estado": "cancelado"},
    ],
    104: [],
}


def _normalizar(texto: str) -> str:
    """Minusculas y sin tildes: 'Gómez' y 'gomez' tienen que matchear igual."""
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sin_tildes.lower().strip()


@tool(parse_docstring=True)
async def buscar_cliente(nombre: str) -> dict[str, Any]:
    """Busca clientes por nombre o apellido y devuelve su cliente_id.

    Usala cuando el usuario nombre a un cliente por su nombre (por ejemplo
    "Laura Gomez" o "Garcia") y todavia no tengas su cliente_id numerico: las demas
    herramientas piden el id, no el nombre. No la uses si el usuario ya dio el id.

    Devuelve {"cantidad": n, "coincidencias": [{"cliente_id": int, "nombre": str}]}.
    Si hay mas de una coincidencia, NO elijas una por tu cuenta: pedile al usuario
    que aclare cual es. Si no hay ninguna, devuelve {"error": ...}: pedi el dato
    de nuevo o proba con solo el apellido.

    Args:
        nombre: Nombre completo, o solo nombre o apellido, del cliente. No distingue tildes ni mayusculas.
    """
    await asyncio.sleep(0)  # simula la ida y vuelta a la base
    buscado = _normalizar(nombre)
    if not buscado:
        return {"error": "El nombre esta vacio. Pedile al usuario el nombre del cliente."}
    partes = buscado.split()
    coincidencias = [
        {"cliente_id": cid, "nombre": nom}
        for cid, nom in CLIENTES.items()
        if all(p in _normalizar(nom) for p in partes)
    ]
    if not coincidencias:
        return {"error": f"No hay clientes que coincidan con {nombre!r}.", "coincidencias": []}
    return {"cantidad": len(coincidencias), "coincidencias": coincidencias}


@tool(parse_docstring=True)
async def buscar_pedidos(cliente_id: int) -> dict[str, Any]:
    """Consulta los pedidos de un cliente por su cliente_id numerico.

    Usala para responder cuantos pedidos tuvo un cliente, cuanto gasto en total,
    cual fue su ultimo pedido o en que estado esta. Requiere el cliente_id (un
    entero como 102); si solo tenes el nombre, primero usa buscar_cliente.

    Devuelve {"cliente_id", "cliente", "cantidad_pedidos", "total", "pedidos": [...]}
    donde "total" es la suma en pesos de todos los pedidos y cada pedido trae
    id, fecha (AAAA-MM-DD), total y estado. Si el cliente no existe devuelve
    {"error": ...}: en ese caso el id estaba mal, busca al cliente por nombre
    con buscar_cliente o pedile el dato al usuario; no inventes cifras.

    Args:
        cliente_id: Identificador numerico del cliente, por ejemplo 102.
    """
    await asyncio.sleep(0)
    if cliente_id not in CLIENTES:
        return {
            "error": f"No existe un cliente con id {cliente_id}.",
            "sugerencia": "Si tenes el nombre del cliente, buscalo con buscar_cliente.",
        }
    pedidos = PEDIDOS.get(cliente_id, [])
    return {
        "cliente_id": cliente_id,
        "cliente": CLIENTES[cliente_id],
        "cantidad_pedidos": len(pedidos),
        "total": sum(p["total"] for p in pedidos),
        "pedidos": pedidos,
    }


HERRAMIENTAS: list[BaseTool] = [buscar_cliente, buscar_pedidos]
