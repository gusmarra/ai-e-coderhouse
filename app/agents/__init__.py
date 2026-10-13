"""Agentes del orquestador: supervisor, especialistas y el ejecutor (acciones criticas)."""

from app.agents.analyst_agent import crear_analista
from app.agents.executor_agent import crear_ejecutor
from app.agents.research_agent import crear_investigador
from app.agents.supervisor import crear_sintesis, crear_supervisor

__all__ = ["crear_analista", "crear_ejecutor", "crear_investigador", "crear_sintesis", "crear_supervisor"]
