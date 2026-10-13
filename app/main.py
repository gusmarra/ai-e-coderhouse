"""API REST del orquestador multi-agente.

    POST /tasks               -> 202 {job_id}; encola y vuelve al instante
    GET  /tasks/{id}          -> estado del job (PENDING|RUNNING|AWAITING_APPROVAL|DONE|FAILED)
    POST /tasks/{id}/approve  -> aprueba o rechaza una accion critica (ver app/hitl.py)
    GET  /health              -> Redis y backend de observabilidad

Arranque: `uvicorn app.main:app`. Las dependencias (Redis, grafo, worker) se crean en `lifespan`;
los tests inyectan un `grafo` y un `redis` propios con `crear_app(...)`.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from app.config import REDIS_URL
from app.graph import checkpointer_redis, construir_grafo
from app.hitl import router as hitl_router
from app.jobs import JobStore
from app.observability import init_observabilidad
from app.worker import Worker

__all__ = ["app", "crear_app"]


class NuevaTarea(BaseModel):
    pregunta: str = Field(min_length=3, max_length=2000, description="Consulta para el equipo de agentes.")


def crear_app(*, grafo: CompiledStateGraph | None = None, redis: Redis | None = None) -> FastAPI:
    """Fabrica de la app. Sin argumentos usa Redis real, Phoenix/LangSmith y el LLM del entorno."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as pila:
            cliente = redis or Redis.from_url(REDIS_URL, decode_responses=True)
            if redis is None:
                pila.push_async_callback(cliente.aclose)
            if grafo is None:
                from app.llm import crear_modelo

                init_observabilidad()
                saver = await pila.enter_async_context(checkpointer_redis(REDIS_URL))
                llm: BaseChatModel = crear_modelo()
                g = construir_grafo(llm, checkpointer=saver)
            else:
                g = grafo
            app.state.store = JobStore(cliente)
            app.state.worker = Worker(g, app.state.store)
            app.state.worker.iniciar()
            pila.push_async_callback(app.state.worker.detener)
            yield

    app = FastAPI(title="Orquestador multi-agente · Hotel Bahia Serena", version="1.0.0", lifespan=lifespan)
    app.include_router(hitl_router)

    @app.post("/tasks", status_code=202)
    async def crear_tarea(cuerpo: NuevaTarea, request: Request) -> dict[str, str]:
        job_id = uuid.uuid4().hex
        await request.app.state.worker.encolar(job_id, cuerpo.pregunta)
        return {"job_id": job_id, "status": "PENDING"}

    @app.get("/tasks/{job_id}")
    async def estado_tarea(job_id: str, request: Request) -> dict[str, Any]:
        job = await request.app.state.store.obtener(job_id)
        if job is None:
            raise HTTPException(404, "job inexistente (o expirado: los jobs viven 24 h)")
        return job

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        await request.app.state.store.redis.ping()
        return {"redis": "ok"}

    return app


app = crear_app()
