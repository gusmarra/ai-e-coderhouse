"""Prueba de carga: N peticiones concurrentes contra la API y resumen de latencias.

    python scripts/carga.py                  # 5 peticiones (la consigna)
    python scripts/carga.py -n 5 --aprobar   # ademas aprueba las que piden aprobacion humana

Cada peticion es un POST /tasks (que vuelve al instante) y despues polling de GET /tasks/{id}.
El costo por ejecucion y la latencia p95 los calcula el dashboard (Phoenix/LangSmith) a partir de
las trazas de esta misma corrida; este script imprime la vista del cliente para contrastar.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from typing import Any

import httpx

CONSULTAS = [
    "Un huesped reservo 4 noches a USD 150 con tarifa de temporada alta y cancela 3 dias antes del "
    "check-in. ¿Cuanto paga de penalidad y que plazo tenia para cancelar sin cargo?",
    "¿Cual es el horario de check-in y check-out y cuanto cuesta el late check-out?",
    "¿Se admiten mascotas? Si es asi, ¿cuanto cuesta por noche una mascota durante 3 noches a USD 20?",
    "¿Que debo hacer ante una alarma de incendio en el hotel?",
    "¿Que incluye el desayuno y cual es el horario del spa?",
]
FINALES = {"DONE", "FAILED"}


async def una(cliente: httpx.AsyncClient, i: int, pregunta: str, aprobar: bool) -> dict[str, Any]:
    t0 = time.perf_counter()
    job_id = (await cliente.post("/tasks", json={"pregunta": pregunta})).json()["job_id"]
    t_encolado = time.perf_counter() - t0
    aprobado = False
    while True:
        job = (await cliente.get(f"/tasks/{job_id}")).json()
        if job["status"] in FINALES:
            break
        if job["status"] == "AWAITING_APPROVAL" and aprobar and not aprobado:
            await cliente.post(f"/tasks/{job_id}/approve", json={"aprobado": True, "comentario": "carga"})
            aprobado = True
        await asyncio.sleep(1)
    return {
        "i": i,
        "job_id": job_id,
        "status": job["status"],
        "encolado_s": t_encolado,
        "total_s": time.perf_counter() - t0,
        "error": job.get("error", ""),
    }


async def main(url: str, n: int, aprobar: bool) -> None:
    preguntas = [CONSULTAS[i % len(CONSULTAS)] for i in range(n)]
    async with httpx.AsyncClient(base_url=url, timeout=30) as cliente:
        t0 = time.perf_counter()
        res = await asyncio.gather(*(una(cliente, i, p, aprobar) for i, p in enumerate(preguntas, 1)))
        pared = time.perf_counter() - t0
    for r in res:
        print(
            f"#{r['i']} {r['status']:<18} encolado en {r['encolado_s'] * 1000:6.0f} ms"
            f" · total {r['total_s']:6.1f} s · {r['job_id']} {r['error']}"
        )
    tiempos = sorted(r["total_s"] for r in res)
    p95 = tiempos[min(len(tiempos) - 1, int(0.95 * len(tiempos)))]
    print(
        f"\n{n} peticiones concurrentes en {pared:.1f} s · mediana {statistics.median(tiempos):.1f} s · p95 {p95:.1f} s"
    )
    print(f"Fallidas: {sum(r['status'] == 'FAILED' for r in res)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--aprobar", action="store_true", help="aprueba automaticamente las tareas criticas")
    a = ap.parse_args()
    asyncio.run(main(a.url, a.n, a.aprobar))
