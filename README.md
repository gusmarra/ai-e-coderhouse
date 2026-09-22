# RAG end-to-end sobre los PRDs de Alphinance (ChromaDB + LCEL + Pydantic)

Un pipeline RAG completo: ingesta documentos `.md`, los persiste en
ChromaDB, y responde preguntas del usuario **solo** con lo que esos
documentos dicen. Si la respuesta no está en el contexto recuperado, el
sistema dice que no la tiene — no alucina.

```python
import asyncio
from chain import get_rag_response

resultado = asyncio.run(get_rag_response(
    "¿Cuál es la fórmula del Ratio de Sharpe y qué representa cada término?"
))
print(resultado.model_dump_json(indent=2))
```

```json
{
  "respuesta": "La fórmula del Ratio de Sharpe es Sharpe ratio = (Rp − Rf) / σp. Rp es el retorno promedio diario de la cartera, Rf la tasa libre de riesgo diaria y σp la volatilidad (desvío estándar) de los retornos.",
  "encontrado_en_contexto": true,
  "referencias": [
    {"fuente": "ratio-de-sharpe.md", "fragmento": "## Requerimiento técnico funcional\n\n### Fórmula general..."}
  ]
}
```

Y frente a algo que los documentos no cubren:

```python
asyncio.run(get_rag_response("¿Qué tasa de interés cobra Alphinance por un préstamo personal?"))
# respuesta: "No tengo esa informacion en los documentos disponibles."
# encontrado_en_contexto: false
```

## El "cerebro": PRDs funcionales de Alphinance

`data/` tiene 4 documentos de especificación funcional de la misma
plataforma (Alphinance, un sistema de inversiones), elegidos porque se
referencian entre sí y comparten vocabulario técnico — el escenario real
donde un RAG con `top_k` chico (no todo el corpus) demuestra su valor:

| Archivo | Contenido |
| --- | --- |
| [data/cuentas-origen-destino.md](data/cuentas-origen-destino.md) | Cuentas origen/destino de fondos en una operación |
| [data/operaciones-compuestas.md](data/operaciones-compuestas.md) | Operaciones que generan múltiples líneas vinculadas (transferencias, forex, FCI en especie, opciones) |
| [data/anulacion-operaciones.md](data/anulacion-operaciones.md) | Reglas de anulación, saldos negativos y recálculo FIFO |
| [data/ratio-de-sharpe.md](data/ratio-de-sharpe.md) | Widget de dashboard: fórmula y cálculo paso a paso del Sharpe ratio |

## Arquitectura

```
data/*.md  ->  ingest.py (chunking)  ->  ChromaDB (./vectorstore)
                                              |
pregunta  ->  retriever.ainvoke() (top_k=4) --+
                                              |
                                    contexto + pregunta
                                              |
                              PROMPT | modelo | PydanticOutputParser   <- chain.py (LCEL)
                                              |
                                     RespuestaModelo (LLM)
                                              |
                     + referencias reales (metadatos del retriever)
                                              |
                                      RespuestaRAG (schemas.py)
```

| Archivo | Rol |
| --- | --- |
| [schemas.py](schemas.py) | `RespuestaModelo` (lo que redacta el LLM), `Referencia` y `RespuestaRAG` (la salida final) |
| [llm.py](llm.py) | `crear_modelo()` / `crear_embeddings()`: factory multi-proveedor (openai / gemini), misma lógica de resolución para las dos familias |
| [ingest.py](ingest.py) | Módulo de ingesta: lee `data/`, fragmenta y persiste en ChromaDB (idempotente) |
| [chain.py](chain.py) | Prompt, `PROMPT \| modelo \| PydanticOutputParser` (LCEL) y `get_rag_response()` async |
| [main.py](main.py) | Demo contra la API real: ingesta + una pregunta con respuesta en los documentos + una pregunta trampa |
| [validate_offline.py](validate_offline.py) | 35 verificaciones sin red, sin API keys y sin ChromaDB real |

### Por qué las referencias no las escribe el LLM

`RespuestaModelo` (lo único que el LLM completa) solo tiene `respuesta` y
`encontrado_en_contexto`. Las `referencias` de `RespuestaRAG` se arman en
`chain.py::_armar_referencias()` a partir de los metadatos que devuelve el
retriever de ChromaDB — no de lo que el modelo diga que usó. Si dejáramos
que el LLM redactara sus propias citas, podría citar una fuente que nunca
recuperó (alucinación de cita); con esta separación, toda referencia que
aparece en la respuesta es, por construcción, un chunk que realmente estuvo
en el contexto.

Por la misma razón, `get_rag_response()` siempre adjunta las referencias
que trajo el retriever, incluso cuando `encontrado_en_contexto=false`: eso
permite ver qué fragmentos se recuperaron (y verificar que, en efecto, no
eran relevantes) en vez de esconderlos.

## Instalación

```powershell
# Windows / PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env           # y completar la key del proveedor a usar
```

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env` solo necesita una API key: `GEMINI_API_KEY` (o `OPENAI_API_KEY` si
se cambia `LLM_PROVIDER=openai`). **Importante**: el mismo proveedor se usa
para chat y para embeddings — ver la sección de "Embeddings no
coincidentes" más abajo.

## Cómo correrlo

```bash
python validate_offline.py   # sin red, sin API key: valida chunking, esquemas, prompt y la cadena LCEL con un modelo falso
python main.py                # ingesta (si hace falta) + una pregunta respondible + una pregunta trampa
python main.py --forzar       # reindexa data/ desde cero (por si se editaron los .md)
```

`python ingest.py` también se puede correr solo, para poblar
`./vectorstore` sin disparar ninguna consulta.

## Chunking

`RecursiveCharacterTextSplitter` con `chunk_size` ≈ 500 tokens y `overlap`
≈ 50 tokens (mínimo pedido por la consigna), separando primero por sección
(`##`/`###`), después por párrafo y por oración antes de cortar a lo bruto.

El splitter mide en caracteres, no en tokens exactos. En vez de
`from_tiktoken_encoder` (que en la primera corrida descarga el archivo de
encoding de tiktoken desde internet) se usa la heurística estándar de ~4
caracteres por token en prosa en español/inglés — una aproximación
documentada, elegida para que `validate_offline.py` pueda correr sin red.
Ver `ingest.py::CHUNK_SIZE`.

## Persistencia

`ingest.py::ingerir()` abre la colección de `./vectorstore` y, si ya tiene
chunks, **no vuelve a embeberlos** (evita gastar cuota de la API de
embeddings en cada corrida de `main.py`). Para reconstruirla desde cero —
por ejemplo, después de editar los `.md` de `data/` — usar `--forzar` o
`ingerir(forzar=True)`.

`./vectorstore` está en `.gitignore`: no se versiona, se reconstruye con
`python ingest.py` en cualquier máquina.

## Errores comunes que este proyecto evita a propósito

- **Contexto infinito**: `TOP_K = 4` (ver `chain.py`). Pasar 20-30
  fragmentos al prompt no mejora la respuesta: la degrada (lost in the
  middle) y acerca el límite de tokens.
- **Embeddings no coincidentes**: `llm.py::crear_embeddings()` resuelve el
  proveedor con la misma lógica que `crear_modelo()` (mismo `LLM_PROVIDER`
  del `.env`), así que `ingest.py` y `chain.py` no pueden indexar con un
  modelo de embeddings y consultar con otro por accidente. Mezclarlos no
  tira un error: tira resultados que parecen aleatorios, porque la
  distancia vectorial entre espacios de proveedores distintos no significa
  nada.
- **Falta de persistencia**: `ingerir()` verifica `coleccion.count() > 0`
  antes de reembeber (ver arriba).

## Nota sobre la cuota de la API

El entorno del curso solo tiene configurada `GEMINI_API_KEY`, de free tier:
**20 requests por día y por modelo** (separado para el modelo de chat y el
de embeddings). Una corrida completa de `main.py` gasta ~1 llamada de
embeddings (la ingesta embebe los 35 chunks en un solo batch) + 2 llamadas
de embeddings de consulta + 2 llamadas de chat — muy por debajo del límite.
Aun así, para iterar sobre chunking, esquemas o el prompt sin tocar la red,
usar `validate_offline.py`.
