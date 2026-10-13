"""Estado de los trabajos en Redis: un hash por job (`job:<id>`) con TTL.

Estados: PENDING -> RUNNING -> (AWAITING_APPROVAL -> PENDING -> RUNNING)* -> DONE | FAILED.
Todo es `redis.asyncio`: ninguna llamada bloquea el event loop de FastAPI.
"""

from __future__ import annotations

import json
import time
from enum import StrEnum
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import WatchError

__all__ = ["EstadoJob", "JobStore"]

TTL_SEGUNDOS = 24 * 3600
_CAMPOS_JSON = ("pendiente", "contribuciones")


class EstadoJob(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    DONE = "DONE"
    FAILED = "FAILED"


def _clave(job_id: str) -> str:
    return f"job:{job_id}"


class JobStore:
    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    async def crear(self, job_id: str, pregunta: str) -> None:
        ahora = str(time.time())
        await self.redis.hset(  # type: ignore[misc,arg-type]
            _clave(job_id),
            mapping={
                "id": job_id,
                "status": EstadoJob.PENDING.value,
                "pregunta": pregunta,
                "creado": ahora,
                "actualizado": ahora,
            },
        )
        await self.redis.expire(_clave(job_id), TTL_SEGUNDOS)

    async def actualizar(self, job_id: str, **campos: Any) -> None:
        mapping = {
            k: json.dumps(v, ensure_ascii=False) if k in _CAMPOS_JSON else ("" if v is None else str(v))
            for k, v in campos.items()
        }
        mapping["actualizado"] = str(time.time())
        await self.redis.hset(_clave(job_id), mapping=mapping)  # type: ignore[misc,arg-type]

    async def obtener(self, job_id: str) -> dict[str, Any] | None:
        crudo: dict[str, str] = await self.redis.hgetall(_clave(job_id))  # type: ignore[misc,assignment]
        if not crudo:
            return None
        return {k: (json.loads(v) if k in _CAMPOS_JSON and v else v) for k, v in crudo.items()}

    async def transicion(self, job_id: str, desde: EstadoJob, hacia: EstadoJob) -> bool:
        """Compare-and-set del estado (WATCH/MULTI). False si el job no estaba en `desde`."""
        async with self.redis.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(_clave(job_id))
                if await pipe.hget(_clave(job_id), "status") != desde.value:
                    return False
                pipe.multi()
                pipe.hset(_clave(job_id), mapping={"status": hacia.value, "actualizado": str(time.time())})
                await pipe.execute()
                return True
            except WatchError:
                return False
