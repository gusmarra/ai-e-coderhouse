"""Contrato de salida del pipeline: lo unico que el resto del codigo conoce.

El esquema cumple dos roles a la vez y por eso vale la pena escribirlo con
cuidado:

1. **Le dice al modelo que queremos.** `with_structured_output()` traduce esta
   clase a un JSON Schema y se lo pasa al LLM como definicion de herramienta,
   asi que cada `description=` es, literalmente, parte del prompt.
2. **Es el control de calidad.** Si la respuesta no respeta las restricciones,
   Pydantic levanta `ValidationError` y la cadena reintenta (ver `chain.py`).

Las restricciones son deliberadamente estrictas: preferimos un reintento a un
objeto vacio que despues rompe rio abajo.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = ["ExtraccionTecnica", "NivelDeCriticidad"]


class NivelDeCriticidad(StrEnum):
    """Escala cerrada: el modelo no puede inventar un nivel intermedio."""

    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class ExtraccionTecnica(BaseModel):
    """Entidades tecnicas extraidas de un texto libre.

    Ejemplo de instancia valida::

        ExtraccionTecnica(
            tecnologias=["FastAPI", "Redis", "PostgreSQL"],
            nivel_de_criticidad=NivelDeCriticidad.ALTA,
            resumen_tecnico=(
                "API con cache en Redis y persistencia en PostgreSQL; "
                "cuello de botella en conexiones concurrentes."
            ),
        )
    """

    # extra="forbid" es intencional: si el modelo agrega un campo que no
    # pedimos, preferimos enterarnos (y reintentar) antes que ignorarlo.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tecnologias: list[str] = Field(
        min_length=1,
        max_length=25,
        description=(
            "Nombres propios de tecnologias, servicios, lenguajes o librerias "
            "mencionados en el texto (ej: 'FastAPI', 'Redis', 'PostgreSQL'). "
            "Sin duplicados y sin categorias genericas como 'base de datos'. "
            "Si el texto no menciona ninguna, no inventes: la extraccion falla."
        ),
    )
    nivel_de_criticidad: NivelDeCriticidad = Field(
        description=(
            "Impacto operativo que describe el texto. "
            "'alta' = hay una falla, caida o riesgo inmediato en produccion; "
            "'media' = degradacion, deuda tecnica o riesgo latente; "
            "'baja' = descripcion informativa, sin incidente."
        ),
    )
    resumen_tecnico: str = Field(
        min_length=20,
        max_length=400,
        description=(
            "Una o dos oraciones en espanol explicando que hace el sistema y "
            "cual es el problema o riesgo principal. Debe nombrar al menos una "
            "de las tecnologias listadas."
        ),
    )

    @field_validator("tecnologias", mode="after")
    @classmethod
    def _normalizar_tecnologias(cls, valores: list[str]) -> list[str]:
        """Colapsa espacios, descarta vacios y deduplica sin distinguir mayusculas.

        El modelo suele devolver `["Redis", "redis "]`: normalizarlo aca evita
        que cada consumidor tenga que hacerlo de nuevo.
        """
        vistas: set[str] = set()
        limpias: list[str] = []
        for bruto in valores:
            nombre = " ".join(bruto.split())
            clave = nombre.casefold()
            if not nombre or clave in vistas:
                continue
            vistas.add(clave)
            limpias.append(nombre)
        if not limpias:
            raise ValueError("tecnologias quedo vacia despues de limpiar los valores")
        return limpias

    @model_validator(mode="after")
    def _resumen_coherente(self) -> Self:
        """El resumen tiene que hablar de lo que se extrajo.

        Es la validacion cruzada que atrapa la respuesta perezosa: una lista de
        tecnologias correcta acompanada de un resumen generico del estilo
        "el sistema presenta problemas". Si no coinciden, se reintenta.
        """
        resumen = self.resumen_tecnico.casefold()
        if not any(tec.casefold() in resumen for tec in self.tecnologias):
            raise ValueError(
                "resumen_tecnico debe mencionar al menos una de las tecnologias "
                f"extraidas: {self.tecnologias}"
            )
        return self
