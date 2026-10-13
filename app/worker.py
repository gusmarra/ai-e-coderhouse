"""Worker en segundo plano: saca tareas de una cola y las corre sin bloquear la API.

Patron: la API solo escribe `PENDING` en Redis y pone el job en una `asyncio.Queue`; N consumidores
(`WORKER_CONCURRENCIA`) los ejecutan con `grafo.ainvoke` (async nativo: el event loop nunca se
frena). Es la version "un solo proceso" que pide la consigna; para escalar a varios procesos el
mismo contrato (estado en Redis + checkpointer en Redis) se mueve tal cual a Arq o Celery.

Regla de oro: toda salida de `_ejecutar` termina en un estado final o de espera en Redis. Cualquier
excepcion (LLM caido, timeout, recursion limit) pasa a `FAILED` con el error, asi el cliente que
hace polling nunca queda esperando para siempre.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from app.config import JOB_TIMEOUT_S, RECURSION_LIMIT, WORKER_CONCURRENCIA
from app.graph import estado_inicial
from app.jobs import EstadoJob, JobStore
from app.observability import config_traza

__all__ = ["Worker"]

log = logging.getLogger("orquestador.worker")


class Worker:
    def __init__(
        self,
        grafo: CompiledStateGraph,
        store: JobStore,
        *,
        concurrencia: int = WORKER_CONCURRENCIA,
        timeout: float = JOB_TIMEOUT_S,
    ) -> None:
        self.grafo = grafo
        self.store = store
        self.concurrencia = concurrencia
        self.timeout = timeout
        self._cola: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        self._consumidores: list[asyncio.Task[None]] = []

    # --- ciclo de vida -------------------------------------------------
    def iniciar(self) -> None:
        self._consumidores = [
            asyncio.create_task(self._consumir(), name=f"worker-{i}") for i in range(self.concurrencia)
        ]

    async def detener(self) -> None:
        for t in self._consumidores:
            t.cancel()
        await asyncio.gather(*self._consumidores, return_exceptions=True)

    async def esperar_cola_vacia(self) -> None:
        """Para tests: bloquea hasta que se procesaron todos los jobs encolados."""
        await self._cola.join()

    # --- encolado (lo que usa la API) ----------------------------------
    async def encolar(self, job_id: str, pregunta: str) -> None:
        await self.store.crear(job_id, pregunta)
        await self._cola.put((job_id, estado_inicial(pregunta)))

    async def encolar_reanudacion(self, job_id: str, aprobado: bool, comentario: str) -> None:
        await self._cola.put((job_id, Command(resume={"aprobado": aprobado, "comentario": comentario})))

    # --- ejecucion -----------------------------------------------------
    async def _consumir(self) -> None:
        while True:
            job_id, entrada = await self._cola.get()
            try:
                await self._ejecutar(job_id, entrada)
            finally:
                self._cola.task_done()

    async def _ejecutar(self, job_id: str, entrada: Any) -> None:
        try:
            await self.store.actualizar(job_id, status=EstadoJob.RUNNING.value, error="")
            config: dict[str, Any] = {
                "configurable": {"thread_id": job_id},
                "recursion_limit": RECURSION_LIMIT,
                **config_traza(job_id),
            }
            resultado = await asyncio.wait_for(self.grafo.ainvoke(entrada, config), self.timeout)  # type: ignore[call-overload]
            if interrupciones := resultado.get("__interrupt__"):
                # El grafo se pauso en el nodo de aprobacion: queda en Redis esperando a un humano.
                await self.store.actualizar(
                    job_id, status=EstadoJob.AWAITING_APPROVAL.value, pendiente=interrupciones[0].value
                )
                return
            await self.store.actualizar(
                job_id,
                status=EstadoJob.DONE.value,
                respuesta=resultado.get("respuesta_final", ""),
                aprobado=resultado.get("aprobado"),
                pasos=resultado.get("pasos", 0),
                contribuciones=resultado.get("contribuciones", []),
                pendiente=None,
            )
        except asyncio.CancelledError:
            await self._fallar(job_id, "cancelado: la API se estaba apagando")
            raise
        except TimeoutError:
            await self._fallar(job_id, f"timeout: la ejecucion supero {self.timeout:.0f}s")
        except Exception as exc:
            log.exception("job %s fallo", job_id)
            await self._fallar(job_id, f"{type(exc).__name__}: {exc}")

    async def _fallar(self, job_id: str, error: str) -> None:
        try:
            await self.store.actualizar(job_id, status=EstadoJob.FAILED.value, error=error, pendiente=None)
        except Exception:  # Redis caido: no hay donde reportar, que al menos quede en el log
            log.exception("no se pudo marcar FAILED el job %s (%s)", job_id, error)
