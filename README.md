# Pipeline de extracción de entidades técnicas (LCEL + Pydantic)

Recibe un párrafo de texto sin procesar —un log de error, una descripción de
arquitectura— y devuelve un objeto **validado**, no un string con JSON adentro.

```python
import asyncio
from chain import process_text

resultado = asyncio.run(process_text(
    "504 Gateway Timeout en POST /v1/orders: el pool de PostgreSQL quedó "
    "saturado porque el caché de Redis se invalidó entero tras el deploy y "
    "todas las requests de FastAPI fueron a la base."
))
print(resultado.model_dump_json(indent=2))
```

```json
{
  "tecnologias": ["PostgreSQL", "Redis", "FastAPI"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "El pool de conexiones de PostgreSQL se saturó tras invalidarse el caché de Redis, lo que derivó todas las requests de FastAPI a la base y provocó timeouts."
}
```

`resultado` es una instancia de `ExtraccionTecnica`: si el modelo hubiera
devuelto una lista vacía, un nivel inventado o un JSON cortado a la mitad, la
cadena lo habría detectado y reintentado antes de llegar a esta línea.

## La cadena, en una expresión

```python
PROMPT | modelo.with_structured_output(ExtraccionTecnica, include_raw=True) | _validar
#                                                                      └─ .with_retry(...)
```

| Eslabón | Qué aporta |
| --- | --- |
| `PROMPT` | `ChatPromptTemplate` con dos variables (`texto`, `instrucciones_formato`). Sin f-strings: las variables las gestiona LangChain. |
| `with_structured_output(..., include_raw=True)` | El esquema Pydantic viaja como definición de herramienta; el modelo responde con JSON tipado. |
| `_validar` | Mira el `finish_reason`, el `parsing_error` y el objeto parseado, y traduce cada falla a una excepción propia. |
| `.with_retry(...)` | Vuelve a llamar al modelo **solo** ante esas excepciones, con backoff exponencial y jitter. |

## Estructura

| Archivo | Rol |
| --- | --- |
| [schemas.py](schemas.py) | `ExtraccionTecnica` y `NivelDeCriticidad`: el contrato de salida y sus restricciones |
| [chain.py](chain.py) | Prompt, factory de modelos, validación, reintentos, `process_text()` y `process_batch()` |
| [main.py](main.py) | Mini script de prueba asíncrono contra la API real (4 demos) |
| [validate_offline.py](validate_offline.py) | 36 verificaciones sin red, sin API keys y sin SDKs |

## Instalación

```powershell
# Windows / PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env           # y completar la key del proveedor a usar
```

```bash
# Linux / macOS
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

```bash
python validate_offline.py     # no necesita API key ni red
python main.py                 # las 4 demos contra la API (~10 llamadas)
python main.py anthropic       # fuerza un proveedor
python main.py gemini 1 2      # solo las demos 1 y 2
```

## El contrato (`schemas.py`)

```python
class ExtraccionTecnica(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tecnologias: list[str] = Field(min_length=1, max_length=25, description=...)
    nivel_de_criticidad: NivelDeCriticidad          # enum: baja | media | alta
    resumen_tecnico: str = Field(min_length=20, max_length=400, description=...)
```

El esquema hace dos trabajos a la vez, y por eso vale la pena escribirlo con
cuidado: cada `description=` termina dentro del prompt (es lo que
`with_structured_output()` le manda al modelo como definición de herramienta), y
las restricciones son el control de calidad de lo que vuelve.

Las restricciones son deliberadamente estrictas —un reintento es más barato que
un objeto vacío que rompe río abajo:

- `tecnologias` no puede estar vacía; se normalizan espacios y se deduplica sin
  distinguir mayúsculas (`["Redis", "redis "]` → `["Redis"]`).
- `nivel_de_criticidad` es un `StrEnum`: el modelo no puede inventar "crítica".
- `resumen_tecnico` tiene que **mencionar al menos una de las tecnologías
  extraídas** (validación cruzada con `@model_validator`). Es lo que atrapa la
  respuesta perezosa: lista correcta + resumen genérico del estilo "el sistema
  presenta problemas".
- `extra="forbid"`: un campo de más también dispara el reintento.

## La resiliencia (`chain.py`)

El pipeline distingue tres formas de fallar, y las tres reintentan:

| Situación | Excepción | Cómo se detecta |
| --- | --- | --- |
| El modelo se quedó sin tokens y cortó el JSON | `RespuestaTruncadaError` | `finish_reason` / `stop_reason` del mensaje crudo |
| El JSON llegó entero pero no cumple el contrato | `ExtraccionIncompletaError` | el `parsing_error` que devuelve `include_raw=True` |
| El modelo contestó en prosa y nunca llamó a la herramienta | `RespuestaVaciaError` | `parsed is None` |

```python
(estructurado | RunnableLambda(_validar)).with_retry(
    retry_if_exception_type=(SalidaNoValidaError, ValidationError),
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)
```

Solo se reintenta eso. Un 401 o un modelo inexistente se propagan tal cual: no
tiene sentido reintentar tres veces algo que no va a cambiar, y los 429 y los
errores de red ya los reintenta el SDK del proveedor.

### Por qué `include_raw=True`

Es la parte que se suele saltear. Sin él, un JSON cortado a la mitad llega como
una excepción de parseo opaca (`Unterminated string...`) y el mensaje original
se pierde. Con él, la cadena recibe `{"raw", "parsed", "parsing_error"}` y puede
mirar el `finish_reason` **antes** de confiar en el objeto:

```
WARNING  clase_2.chain: respuesta cortada por el proveedor (finish_reason=max_tokens): reintento
```

Cada proveedor lo nombra distinto —OpenAI y Gemini usan `finish_reason`,
Anthropic usa `stop_reason`—, así que `_motivo_de_corte()` normaliza los tres.
La demo 4 de `main.py` lo fuerza con `max_tokens=16`.

## Proveedores

`crear_modelo()` replica el factory del Módulo 1 (`llm_client/factory.py` de la
entrega anterior) sobre los chat models de LangChain: mismo nombre de variable
(`LLM_PROVIDER`), mismos nombres de proveedor y el mismo import perezoso, así
que usar Gemini no obliga a instalar los otros dos SDKs.

| `LLM_PROVIDER` | Clase | Modelo por defecto |
| --- | --- | --- |
| `openai` | `ChatOpenAI` | `gpt-4.1-mini` |
| `anthropic` | `ChatAnthropic` | `claude-sonnet-5` |
| `gemini` (alias `google`) | `ChatGoogleGenerativeAI` | `gemini-3.6-flash` |

Sin `LLM_PROVIDER`, se usa el primero que tenga API key en el entorno.
`LLM_MODEL` pisa el modelo por defecto.

`temperature=0`: esto es extracción, no redacción; para el mismo texto queremos
la misma salida.

## Dos cosas que aparecieron probando contra la API real

**`max_tokens=1024` no alcanza.** Suena generoso para un JSON de tres campos,
pero los modelos actuales razonan antes de responder y ese razonamiento sale del
mismo presupuesto: con `gemini-3.6-flash` cada extracción consumió entre 550 y
900 tokens de salida. Con 1024, el truncado dejaba de ser un caso de borde y
pasaba a ser el caso normal. El default quedó en `MAX_TOKENS = 2048`.

**La validación no atrapa la alucinación.** En la prueba de estrés (demo 2) se
le pasa un texto sin ninguna tecnología. Lo esperable es que `min_length=1`
rechace la lista vacía y se agoten los reintentos; lo que a veces pasa es que el
modelo, presionado por el contrato, devuelve `{"tecnologias": ["Panaderia"]}`.
El JSON es válido y el pipeline lo acepta. El límite es real y conviene tenerlo
presente: Pydantic verifica la **forma**, no la **verdad**. Eso se ataca en el
prompt, o agregando al esquema un campo de confianza o un booleano
`contiene_tecnologias` que el modelo pueda poner en `false` sin sentir que
incumple.

## Validación offline

`validate_offline.py` reemplaza la única pieza que necesita red —la capa
`with_structured_output`— por un runnable falso que devuelve exactamente la
misma forma. Eso permite testear de forma determinista lo que contra la API real
sería un volado: cuántas veces reintenta, ante qué excepciones, y qué pasa
cuando el segundo intento sale bien.

```
1. Esquema Pydantic (schemas.py)                        10 checks
2. Prompt template (ChatPromptTemplate, sin f-strings)   6 checks
3. Validacion de la salida y reintentos (.with_retry)   14 checks
4. process_text / process_batch                          6 checks

Todas las verificaciones pasaron.
```

## Checklist de la entrega

| Requisito | Dónde |
| --- | --- |
| Esquema Pydantic con `tecnologias`, `nivel_de_criticidad` (enum) y `resumen_tecnico` | [schemas.py](schemas.py) |
| Prompt template modular que acepta el texto y las instrucciones de formato | `PROMPT` en [chain.py](chain.py) |
| Cadena LCEL `prompt \| model.with_structured_output(Schema)` | `construir_cadena()` en [chain.py](chain.py) |
| Reintento ante JSON mal formado o incompleto (`.with_retry()`) | `con_resiliencia()` en [chain.py](chain.py) |
| Detección de `finish_reason` antes de transformar el objeto | `_validar()` / `_motivo_de_corte()` en [chain.py](chain.py) |
| `async def process_text(text)` con `.ainvoke()` y logs de validación | [chain.py](chain.py) |
| Mini script de prueba asíncrono | [main.py](main.py) y [validate_offline.py](validate_offline.py) |
