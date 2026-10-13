"""Agente de Investigacion: consulta la base de conocimiento del Hotel Bahia Serena.

Herramientas acotadas a su rol (solo LEER documentos del hotel; no calcula ni valora):

* `buscar_en_documentos_hotel`: recuperador hibrido (BM25 + Pinecone) de las pre-entregas 3 y 4.
* `listar_documentos_hotel`: mapa del corpus, para saber que categoria filtrar.

El docstring de cada tool es lo UNICO que el LLM ve de ella: dice cuando usarla, que devuelve
y que hacer si no hay resultados.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, tool

from app.agents.common import NodoEspecialista, crear_nodo_especialista
from app.base_hotel import BuscadorHotel, buscador_por_defecto
from app.state import EstadoOrquestador

__all__ = ["PROMPT_INVESTIGADOR", "crear_herramientas_investigador", "crear_investigador"]

PROMPT_INVESTIGADOR = """\
Sos el INVESTIGADOR del Hotel Bahia Serena. Tu unico trabajo es encontrar informacion en los
documentos institucionales del hotel y devolverla fiel y con su fuente.

Reglas:
- Todo dato del hotel sale de `buscar_en_documentos_hotel`. Nunca uses conocimiento propio.
- Hace las busquedas que necesites (una por tema), con palabras clave del documento
  (por ejemplo "tarifa temporada alta cancelacion penalidad"), no con la pregunta entera.
- NO calcules, NO saques conclusiones, NO valores: de eso se ocupa el analista. Entregale los
  datos crudos y exactos (montos, porcentajes, plazos, condiciones, excepciones).
- Si la busqueda no devuelve lo pedido, deci explicitamente "No encontre ...": no inventes.
- Formato de respuesta: lista corta de hallazgos; cada uno con el dato textual y su
  fuente entre parentesis, por ejemplo "(fuente: politica-reservas-cancelaciones.md, seccion 4)".
- Si recibis una consigna de REFINAMIENTO, corregi o completa tu respuesta anterior en
  lugar de repetirla.
"""


def crear_herramientas_investigador(buscador: BuscadorHotel | None = None) -> list[BaseTool]:
    """Tools del investigador sobre un buscador dado (por defecto, el del entorno)."""

    def _buscador() -> BuscadorHotel:
        return buscador or buscador_por_defecto()

    @tool(parse_docstring=True)
    def buscar_en_documentos_hotel(consulta: str) -> str:
        """Busca fragmentos en los documentos institucionales del Hotel Bahia Serena.

        Usala para cualquier dato del hotel: politicas de reserva, check-in/out, cancelaciones y
        penalidades, reglamento de huespedes (mascotas, fumadores, visitas), servicios, tarifas
        de extras, protocolos de seguridad. Se puede llamar varias veces con consultas distintas.

        Devuelve hasta 5 fragmentos numerados, cada uno con "fuente: <archivo>.md" y su seccion.
        Si no hay coincidencias devuelve un aviso: reformula con otras palabras clave o informa
        que no encontraste el dato; no lo inventes.

        Args:
            consulta: Palabras clave de lo que se busca, por ejemplo "late check-out cargo".
        """
        try:
            docs = _buscador().buscar(consulta)
        except ValueError as exc:
            return f"Error: {exc}. Reformula la consulta."
        if not docs:
            return "Sin resultados. Probá con otras palabras clave."
        return "\n\n".join(
            f"[{i}] (fuente: {d.metadata.get('source', '?')}, seccion: {d.metadata.get('seccion', '?')})\n"
            f"{d.page_content.strip()}"
            for i, d in enumerate(docs, start=1)
        )

    @tool
    def listar_documentos_hotel() -> str:
        """Lista los documentos disponibles del hotel y de que trata cada uno.

        Usala si no sabes donde buscar un dato o para confirmar que un tema esta cubierto.
        No recibe parametros y devuelve una linea por documento.
        """
        return (
            "- politica-reservas-cancelaciones.md: canales de reserva, garantia, check-in/out, "
            "cancelaciones y penalidades por tarifa, reembolsos, grupos, pagos\n"
            "- reglamento-interno-huespedes.md: normas de convivencia, mascotas, fumadores, visitas, danos\n"
            "- servicios-comodidades.md: categorias de habitacion, desayuno, room service, spa, piscina, wifi\n"
            "- protocolo-seguridad-emergencias.md: evacuacion, incendios, emergencias, acceso, camaras"
        )

    return [buscar_en_documentos_hotel, listar_documentos_hotel]


def _entrada(estado: EstadoOrquestador) -> str:
    partes = [
        f"Pedido original del usuario (solo contexto): {estado['pregunta']}",
        f"Tu consigna ahora: {estado['instruccion']}",
    ]
    if previo := estado.get("hallazgos", {}).get("investigador"):
        partes.append(f"Tu respuesta anterior, a refinar segun la consigna:\n{previo['contenido']}")
    return "\n\n".join(partes)


def crear_investigador(llm: BaseChatModel, buscador: BuscadorHotel | None = None) -> NodoEspecialista:
    return crear_nodo_especialista(
        "investigador", llm, crear_herramientas_investigador(buscador), PROMPT_INVESTIGADOR, _entrada
    )
