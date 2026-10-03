from __future__ import annotations

from herramientas import HERRAMIENTAS, buscar_cliente, buscar_pedidos


async def test_buscar_pedidos_devuelve_cantidad_y_total() -> None:
    r = await buscar_pedidos.ainvoke({"cliente_id": 102})
    assert r["cantidad_pedidos"] == 3
    assert r["total"] == 14500
    assert r["pedidos"][-1]["id"] == 5003


async def test_buscar_pedidos_id_inexistente_devuelve_error_accionable() -> None:
    r = await buscar_pedidos.ainvoke({"cliente_id": 999})
    assert "error" in r
    assert "buscar_cliente" in r["sugerencia"]


async def test_buscar_pedidos_cliente_sin_pedidos() -> None:
    r = await buscar_pedidos.ainvoke({"cliente_id": 104})
    assert (r["cantidad_pedidos"], r["total"]) == (0, 0)


async def test_buscar_cliente_ignora_tildes_y_mayusculas() -> None:
    r = await buscar_cliente.ainvoke({"nombre": "laura GOMEZ"})
    assert r["coincidencias"] == [{"cliente_id": 102, "nombre": "Laura Gómez"}]


async def test_buscar_cliente_ambiguo_devuelve_varias_coincidencias() -> None:
    r = await buscar_cliente.ainvoke({"nombre": "García"})
    assert r["cantidad"] == 2


async def test_buscar_cliente_sin_resultados_devuelve_error() -> None:
    r = await buscar_cliente.ainvoke({"nombre": "Nadie Inexistente"})
    assert "error" in r


def test_las_herramientas_tienen_descripcion_para_el_llm() -> None:
    for h in HERRAMIENTAS:
        assert len(h.description) > 150, f"docstring demasiado corto en {h.name}"
        assert h.args  # el schema de argumentos se genera desde la firma tipada
