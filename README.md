# Cliente de LLM robusto y asíncrono (Python 3.12+)

Un cliente único, `async`, para **OpenAI**, **Anthropic** y **Gemini** detrás de la
misma interfaz, con entrada y salida validadas por **Pydantic**, streaming token a
token y reintentos con backoff exponencial.

```python
import asyncio
from llm_client import ChatMessage, create_client


async def main():
    async with create_client("openai") as client:  # o "anthropic" / "gemini"
        r = await client.generate("¿Qué es la entropía?")
        print(r.content, r.usage.total_tokens)

        async for token in client.stream_text("Contame un chiste corto"):
            print(token, end="", flush=True)


asyncio.run(main())
```

## Estructura

| Archivo | Rol |
| --- | --- |
| [llm_client/schemas.py](llm_client/schemas.py) | `ChatMessage`, `Conversation`, `ModelConfig`, `RetryConfig`, `ModelResponse`, `StreamChunk`, `Usage`, `ErrorResponse` |
| [llm_client/exceptions.py](llm_client/exceptions.py) | Jerarquía de errores propia, con el flag `retryable` |
| [llm_client/base.py](llm_client/base.py) | `BaseLLMClient`: ABC + reintentos + medición + context manager |
| [llm_client/providers/openai_client.py](llm_client/providers/openai_client.py) | `AsyncOpenAI` -> Chat Completions |
| [llm_client/providers/anthropic_client.py](llm_client/providers/anthropic_client.py) | `AsyncAnthropic` -> Messages API |
| [llm_client/providers/gemini_client.py](llm_client/providers/gemini_client.py) | `google-genai` -> `client.aio` |
| [llm_client/factory.py](llm_client/factory.py) | `create_client("openai")`, `available_providers()` |
| [main.py](main.py) | Script de validación contra las APIs reales |
| [validate_offline.py](validate_offline.py) | 33 verificaciones sin red, sin keys y sin SDKs |

## Instalación

```powershell
# Windows / PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1      # si PowerShell lo bloquea:
                                 # Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
copy .env.example .env           # y completar las keys que se vayan a usar
```

```bash
# Linux / macOS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Sin activar el entorno, los scripts se pueden correr apuntando al interprete del
venv: `.venv\Scripts\python.exe validate_offline.py`.

Los SDKs se importan a demanda: si solo se va a usar Anthropic, alcanza con
`pip install pydantic python-dotenv anthropic`.

## Uso

```bash
python validate_offline.py          # no necesita keys ni red
python main.py                      # prueba todos los proveedores con key en .env
python main.py anthropic gemini     # solo esos
```

`main.py` corre, por proveedor: modo normal, streaming, manejo de errores con una
key inválida y tres llamadas en paralelo con `asyncio.gather`.

## Diseño

**1. Un contrato de datos, tres APIs.** Cada proveedor traduce su respuesta a
`ModelResponse`, así la aplicación nunca navega diccionarios anidados
(`resp["choices"][0]["message"]["content"]`). El payload crudo sigue accesible en
`.raw` para debug, pero está excluido de los dumps.

**2. La clase base concentra lo transversal.** Un proveedor nuevo implementa tres
métodos privados y hereda gratis validación, reintentos, latencia, `attempts` y
cierre del cliente HTTP:

```python
class MiProveedor(BaseLLMClient):
    provider_name = "mi-proveedor"
    default_model = "mi-modelo"
    api_key_env = "MI_API_KEY"

    async def _agenerate(self, conversation, config) -> ModelResponse: ...
    async def _astream(self, conversation, config) -> AsyncIterator[StreamChunk]: ...
    def _translate_error(self, exc) -> LLMError | None: ...
```

**3. Dos estilos de error, según el llamador.**

```python
r = await client.generate(msgs)  # lanza LLMError al agotar reintentos
r = await client.generate_safe(msgs)  # devuelve ModelResponse | ErrorResponse
```

`generate_safe()` es lo que hace que un `asyncio.gather` de 50 preguntas no se
caiga entera porque una devolvió 429.

**4. Streaming con totales al final.** `stream()` emite `StreamChunk`; el último
llega con `is_final=True` y el `usage` real. `stream_text()` es el atajo que emite
solo texto, y `collect_stream()` consume el stream y devuelve una `ModelResponse`.

**5. Reintentos donde tienen sentido.** Solo se reintenta lo transitorio (429, 5xx,
red, timeout) con backoff exponencial + jitter, respetando el header `retry-after`.
Un 401 o un 400 fallan al primer intento. En streaming se reintenta **solo antes del
primer token**: después ya se emitió texto y repetir la llamada lo duplicaría.

**6. Normalización de mensajes.** `Conversation` acepta un string, un dict, un
`ChatMessage` o una lista mezclada, extrae el prompt de sistema aparte (Anthropic y
Gemini lo reciben en su propio parámetro) y fusiona mensajes consecutivos del mismo
rol (la Messages API espera roles alternados).

## Errores comunes que este código evita

| Error | Cómo se evita |
| --- | --- |
| Bloquear el event loop con el cliente síncrono | Solo `AsyncOpenAI`, `AsyncAnthropic` y `client.aio` de Gemini; `validate_offline.py` mide 10 llamadas concurrentes para probarlo |
| Que una excepción del SDK rompa el loop principal | Toda excepción nativa se traduce a `LLMError`; `generate_safe()` la devuelve como dato |
| Reintentar lo que nunca va a funcionar | El flag `retryable` distingue 429/5xx de 401/400 |
| Reintentos duplicados e invisibles | Los SDKs se construyen con `max_retries=0`: los reintentos son los del cliente y se ven en `attempts` |
| Un `temperature=9` que se descubre como 400 tras el round trip | `ModelConfig` valida rangos antes de salir a la red, también en los overrides por llamada |
| Diccionarios anidados por toda la app | `ModelResponse` / `StreamChunk` con tipos |
| Perder el `usage` cuando se usa streaming | OpenAI recibe `stream_options={"include_usage": True}` y Anthropic usa `get_final_message()` |
| Fugas de conexiones HTTP | `async with` sobre el cliente (`aclose()` cierra el cliente del SDK) |
| Cancelaciones tragadas | `asyncio.CancelledError` nunca se captura como error de API |

## Notas por proveedor

**Anthropic.** Los modelos actuales (familia Claude 5 / 4.6+) **ya no aceptan
`temperature` ni `top_p`**: la Messages API los rechaza. El cliente los ignora con un
warning en lugar de romper la llamada; la profundidad de razonamiento se controla con
`output_config`:

```python
create_client("anthropic", extra={"output_config": {"effort": "low"}})
```

Además estos modelos razonan por defecto y ese razonamiento consume `max_tokens`: con
un techo muy bajo la respuesta puede volver vacía. Por eso `main.py` pide holgura
(`max_tokens=2000`) aunque la respuesta sea de dos oraciones.

**OpenAI.** Se envía `max_completion_tokens` (el `max_tokens` de Chat Completions está
deprecado) y `stream_options={"include_usage": True}` para recibir tokens en streaming.

**Gemini.** El rol `assistant` se mapea a `model`, el prompt de sistema va en
`system_instruction`, el timeout del SDK está en milisegundos y la key se busca en
`GEMINI_API_KEY` o `GOOGLE_API_KEY`. Una key inválida llega como `400 API_KEY_INVALID`
(no como 401), y el cliente la reclasifica igual a `AuthenticationError`.

**Modelos por defecto.** `gpt-4.1-mini`, `claude-opus-5` y `gemini-3.6-flash`. Google
retira versiones seguido: si aparece un `404 ... no longer available`, el mensaje
suele indicar el reemplazo, y siempre se puede consultar qué habilita la key:

```python
from google import genai
for m in genai.Client().models.list():
    if "generateContent" in (m.supported_actions or []):
        print(m.name)
```

Para cambiar el modelo sin tocar el código del cliente:

```python
create_client("gemini", model="gemini-3.8-flash")
```

Cualquier parámetro propio de un proveedor se pasa por `extra`, que va tal cual al SDK:

```python
create_client("openai", extra={"seed": 42, "presence_penalty": 0.5})
```

## Verificación

`validate_offline.py` implementa un proveedor falso sobre `BaseLLMClient` y verifica
33 comportamientos sin tocar la red: validaciones de Pydantic, normalización de
mensajes, streaming, reintentos (429 que sale bien al tercer intento, 429 persistente,
401 sin reintento), errores como dato y concurrencia real.

```
$ python validate_offline.py
...
todas las verificaciones pasaron
```

Los tres clientes se validaron además de punta a punta contra un servidor HTTP local
que emula las tres APIs (payload enviado, SSE de streaming, `usage`, y clasificación
de 401 / 429 / 500).
