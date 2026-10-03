# Pre-entrega 5 — Agente de razonamiento cíclico con memoria persistente

Agente **ReAct** (Reason + Act) construido con **LangGraph**: un LLM que decide por sí
mismo cuándo llamar a una herramienta, observa el resultado, y vuelve a razonar hasta tener
la respuesta. El estado de cada conversación se guarda en **SQLite** (`AsyncSqliteSaver`),
así que con el mismo `thread_id` el agente recuerda lo hablado, incluso después de
reiniciar el proceso.

El dominio es una base simulada de **clientes y pedidos** (la del ejemplo de la consigna).

## Cómo levantar el entorno

Requiere **Python 3.12+** (el CI corre 3.12 y 3.13; se desarrolló en 3.14).

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # y completar GEMINI_API_KEY (u OPENAI_API_KEY)
```

> Las API keys viven **solo** en `.env`, que está en `.gitignore`. Nunca se suben.

```bash
python main.py demo                                   # corre los escenarios y escribe trazas/
python main.py preguntar "¿Cuántos pedidos tuvo Laura Gómez?" --thread-id ana
python main.py preguntar "¿Y el último?" --thread-id ana   # mismo thread: recuerda a Laura
python main.py chat --thread-id ana                   # conversación interactiva
```

Tests (offline: usan un LLM guionado, no necesitan red ni keys):

```bash
pip install -r requirements-dev.txt
pytest && ruff check . && mypy *.py tests
```

> **Free tier de Gemini:** corta a ~5 requests por minuto y 20 por día *por modelo*. Definí
> `LLM_RPM=4` en `.env` para que un rate limiter espacie las llamadas, y si un modelo se
> agota probá otro con `LLM_MODEL=gemini-3.5-flash-lite` (la cuota es por modelo).

## Estructura

| Archivo | Rol |
|---|---|
| `herramientas.py` | **Fase 1.** Dos tools con `@tool`: `buscar_cliente` y `buscar_pedidos` (async, con errores accionables). |
| `grafo.py` | **Fases 2 y 3.** `EstadoAgente(MessagesState)`, nodo `modelo`, `ToolNode`, `tools_condition` y el checkpointer. |
| `trazas.py` | Corre un turno con `recursion_limit` y reconstruye la traza Pensamiento → Acción → Observación → Respuesta. |
| `demo.py` | Escenarios reproducibles que generan `trazas/traza_ejemplo.{json,log}`. |
| `llm.py`, `config.py` | Factory del modelo (Gemini/OpenAI), `.env`, límites. |
| `main.py` | CLI: `demo`, `preguntar`, `chat`. |
| `tests/` | 15 tests offline: herramientas, ciclo ReAct, reintento tras error, memoria, `recursion_limit`. |

## Criterios de aceptación → dónde se cumplen

| Criterio | Cómo |
|---|---|
| **Autonomía** (sin `if/else` manual) | La ruta la decide `tools_condition` leyendo si el último `AIMessage` trae `tool_calls`. No hay ningún `if` en el código que elija herramientas. |
| **Ciclo de retorno** | La arista `tools → modelo` devuelve el resultado (también los errores) al LLM. Test: `test_ciclo_de_retorno_el_error_vuelve_al_modelo_que_reintenta`. En la traza real, ante un id inexistente o un apellido ambiguo el agente **pide aclaraciones**. |
| **Resiliencia de estado** | `AsyncSqliteSaver` + `thread_id`. La demo cierra y reabre el archivo SQLite entre preguntas del mismo thread. |
| **Código limpio** | Python ≥ 3.12, type hints (`mypy` sin errores), `asyncio` de punta a punta, `ruff` limpio. |

## Traza de ejecución (ciclo ReAct real)

Generada con `gemini-3.5-flash-lite`; completa en [`trazas/traza_ejemplo.log`](trazas/traza_ejemplo.log)
y [`trazas/traza_ejemplo.json`](trazas/traza_ejemplo.json).

```text
=== thread_id=demo-multipaso ===

Usuario: ¿Cuántos pedidos tuvo Laura Gómez y cuál fue el total?
  ACCION      buscar_cliente({'nombre': 'Laura Gomez'})
  OBSERVACION {"cantidad": 1, "coincidencias": [{"cliente_id": 102, "nombre": "Laura Gómez"}]}
  ACCION      buscar_pedidos({'cliente_id': 102})
  OBSERVACION {"cliente_id": 102, "cliente": "Laura Gómez", "cantidad_pedidos": 3, "total": 14500, ...}
  RESPUESTA   Laura Gómez tuvo 3 pedidos y el total gastado fue de $14.500.

Usuario: ¿Y el último?
  ACCION      buscar_pedidos({'cliente_id': 102})
  RESPUESTA   El último pedido fue el N° 5003, realizado el 18/09/2026 por $4.000, y su estado actual es "en camino".

  ... se cierra el checkpointer y se reabre el archivo (reinicio simulado) ...

Usuario: Cambiando de tema un segundo: ¿de qué clienta estábamos hablando y cuánto era su total?
  RESPUESTA   Estábamos hablando de Laura Gómez, y su total gastado es de $14.500.

=== thread_id=demo-errores ===

Usuario: ¿Cuántos pedidos tuvo el cliente 999?
  ACCION      buscar_pedidos({'cliente_id': 999})
  OBSERVACION {"error": "No existe un cliente con id 999.", ...} [ERROR]
  RESPUESTA   No encontré ningún cliente con el ID 999. ¿Tenés el nombre o apellido del cliente para que lo busque?
```

La primera pregunta usa la herramienta **dos veces** (razonamiento multi-paso): el agente
necesita el `cliente_id`, por eso primero busca por nombre y recién después consulta los pedidos.
Las respuestas de un LLM no son deterministas: otra corrida puede variar el orden o la redacción.

## Cómo funciona, paso a paso

**1. El contrato de herramientas (`herramientas.py`).** Una tool es una función con `@tool`.
El LLM nunca ve el código: solo el nombre, los tipos de los parámetros y el *docstring*. Por eso
cada docstring dice *cuándo* usarla, *qué devuelve* y *qué hacer si falla*. Si el agente no usa
la herramienta esperada, el problema casi siempre está ahí, no en el grafo.

**2. El estado (`EstadoAgente`).** Hereda de `MessagesState`, que ya trae
`messages: Annotated[list[AnyMessage], add_messages]`. El *reducer* `add_messages` hace que un
nodo **agregue** mensajes en vez de pisar la lista (hace lo que `operator.add`, y además
actualiza por id). Un nodo devuelve solo lo nuevo: `{"messages": [respuesta]}`.

**3. El grafo (`construir_grafo`).**

```text
START → modelo ──(tools_condition)──→ tools
          ↑                              │
          └──────────────────────────────┘
          └─(sin tool_calls)→ END
```

`modelo` llama al LLM con `llm.bind_tools(...)`. `tools_condition` lee el último mensaje: si
pidió herramientas va a `tools` (un `ToolNode` que las ejecuta y agrega `ToolMessage`s), si no
termina. El ciclo `tools → modelo` es lo que hace "cíclico" al agente.

**4. La persistencia.** `graph.compile(checkpointer=AsyncSqliteSaver)` guarda el estado después
de cada paso. Al invocar con `{"configurable": {"thread_id": "ana"}}`, LangGraph carga el estado
de ese hilo y le *suma* el mensaje nuevo; otro `thread_id` arranca de cero. Usamos
`AsyncSqliteSaver` y no `SqliteSaver` porque el agente es asíncrono (`ainvoke`/`astream`): el
saver síncrono no soporta métodos async. Es el mismo almacenamiento SQLite, versión `aiosqlite`.

**5. Protecciones contra los errores comunes.**

- *Bucles infinitos*: `recursion_limit=10` en cada invocación (`config.py`). Si el modelo no
  converge, `GraphRecursionError` se captura y el turno termina con un mensaje claro
  (test: `test_recursion_limit_corta_un_loop_infinito`).
- *Estado sucio*: el estado guardado crece con cada turno, pero `recortar_contexto` limita
  lo que se **envía** al modelo (últimos 30 mensajes, empezando siempre en un mensaje del
  usuario para no dejar un `ToolMessage` huérfano). El historial completo queda en SQLite.
- *Excepciones en tools*: `ToolNode(handle_tool_errors=True)` las convierte en un `ToolMessage`
  de error, así el modelo puede reintentar en vez de romper el grafo.
- *Descripciones vagas*: un test verifica que cada herramienta tenga un docstring sustancial.
