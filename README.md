# RAG escalable en la nube: Pinecone + recuperador híbrido (BM25 + vectorial)

Pre-entrega 4. Sobre los 4 documentos institucionales del Hotel Bahía Serena
(`data/`), este repo:

1. **Ingesta** los documentos a un índice **Pinecone Serverless**, con el texto
   original y metadatos avanzados dentro de cada vector.
2. **Recupera** con un `RAGSystem` que combina búsqueda vectorial (Pinecone) y
   léxica (**BM25**) mediante un `EnsembleRetriever`, y devuelve el top-5.
3. **Evalúa** el recuperador con un *golden set* midiendo **Precision@k** y
   **Recall@k**.

La generación de la respuesta final (LCEL + Pydantic, entrega anterior) sigue
disponible: `python main.py "..." --generar`. El backup completo de la entrega 3
(ChromaDB) está en `entregas/clase_3/`.

## Estructura

| Archivo | Rol |
| --- | --- |
| [config.py](config.py) | Variables de entorno, dimensión (1536) y métrica del índice |
| [setup_pinecone.py](setup_pinecone.py) | Crea el índice serverless si no existe; rechaza uno con otra dimensión |
| [ingest.py](ingest.py) | `data/*.md` → chunks → embeddings → `upsert` a Pinecone (texto + metadata) |
| [retriever.py](retriever.py) | `PineconeVectorRetriever` y `RAGSystem` (`EnsembleRetriever` BM25 + vectorial) |
| [evaluate.py](evaluate.py) + [golden_set.json](golden_set.json) | Precision@k / Recall@k sobre 16 preguntas con documento esperado |
| [rag.py](rag.py), [schemas.py](schemas.py), [main.py](main.py) | Generación grounded con citas + demo por consola |
| [tests/](tests/) | Suite `pytest` offline (sin red ni API keys) |

## Replicar el índice

Requiere Python ≥ 3.12, una cuenta de Pinecone (el plan gratuito alcanza) y una
key de Gemini **o** de OpenAI.

```bash
# 1. Entorno
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2. Variables: copiar la plantilla y completar PINECONE_API_KEY + la key del proveedor
cp .env.example .env
```

`.env` (nunca se sube: está en `.gitignore`):

```
PINECONE_API_KEY=...             # console.pinecone.io -> API Keys
INDEX_NAME=hotel-bahia-serena    # nombre del índice serverless
LLM_PROVIDER=gemini              # o openai
GEMINI_API_KEY=...               # u OPENAI_API_KEY
```

```bash
# 3. Crear el índice (idempotente: si existe lo reutiliza)
python setup_pinecone.py

# 4. Ingestar (idempotente: los ids de chunk son deterministas, re-correr pisa, no duplica)
python ingest.py                 # --reset vacía el namespace antes de subir

# 5. Consultar y evaluar
python main.py "¿Cuánto cuesta el valet parking?"
python main.py "¿Dónde está el DEA?" --categoria seguridad   # filtro por metadata
python evaluate.py               # imprime el reporte en consola
```

`ingest.py` ya llama a `setup_pinecone`, así que el paso 3 es opcional: existe
para poder verificar la infraestructura por separado. El índice se crea como
`ServerlessSpec(cloud="aws", region="us-east-1")` (la única región del plan
gratuito; se cambia con `PINECONE_CLOUD` / `PINECONE_REGION`), dimensión **1536**
y métrica **coseno**.

## Cómo funciona

```
data/*.md ─► ingest.py ──► embeddings (1536 d) ──► Pinecone  namespace=hotel-bahia-serena
             (chunks + metadata)                      vector + metadata{text, source, ...}
                                                              │
pregunta ─► RAGSystem.retrieve() ─┬─► BM25Retriever (memoria, mismos chunks) ──┐
                                  └─► PineconeVectorRetriever ─────────────────┤
                                                          EnsembleRetriever (RRF, id_key=chunk_id)
                                                                               ▼
                                                                         top-5 chunks
```

**Metadata de cada vector** (todo dentro de Pinecone, sin base relacional aparte):

| campo | ejemplo | uso |
| --- | --- | --- |
| `text` | `"Late check-out: hasta las 13:00..."` | contenido del chunk |
| `source` / `doc_id` | `politica-reservas-cancelaciones.md` / `politica-reservas-cancelaciones` | cita / id estable |
| `categoria` | `reservas` · `reglamento` · `servicios` · `seguridad` | filtro `{"categoria": {"$eq": ...}}` |
| `etiquetas` | `["check-in", "cancelacion", ...]` | filtro por lista |
| `page`, `seccion` | `1`, `3. Check-in y check-out` | ubicación (los `.md` no tienen páginas: `page` es siempre 1; `seccion` es la que aporta precisión) |
| `chunk_index`, `chunk_id` | `2`, `politica-...#002` | orden y **id del vector** |

**Decisiones de diseño**

- **SDK nativo de Pinecone, no `langchain-pinecone`.** La consigna permite ambos;
  `langchain-pinecone` no publica versiones para Python 3.14 (el del entorno de
  desarrollo). `PineconeVectorRetriever` (~20 líneas en `retriever.py`) es el
  adaptador a la interfaz `BaseRetriever` de LangChain, así que el
  `EnsembleRetriever` lo consume igual.
- **Chunking**: `RecursiveCharacterTextSplitter` de ~500 tokens con 50 de overlap,
  cortando primero por encabezado `##`, luego párrafo y oración. Más chico pierde
  contexto semántico; más grande diluye el embedding.
- **Namespace**: todo el corpus vive en un namespace propio (`PINECONE_NAMESPACE`).
  Una consulta solo mira los vectores de su namespace: en un sistema multi-inquilino,
  un namespace por cliente evita resultados ruidosos y acelera la búsqueda.
- **BM25 con tokenizador propio**: minúsculas, sin acentos ni puntuación y sin
  stopwords en español (el default de `BM25Retriever` es `str.split`, con el que
  `"check-in."` y `"Check-in"` serían términos distintos).
- **Fusión**: Reciprocal Rank Fusion con pesos 0.5/0.5. Cada recuperador aporta
  10 candidatos y `RAGSystem` se queda con los 5 mejores de la fusión. Los chunks
  presentes en *ambas* listas suben, incluso por encima del rank 1 de una sola.
- **Embeddings de 1536 dimensiones con ambos proveedores**: OpenAI
  `text-embedding-3-small` es nativo; a Gemini se le pide
  `output_dimensionality=1536`. Ojo: misma dimensión **no** significa mismo
  espacio vectorial. Si cambiás de proveedor, reingestá con `--reset`.
- **Consistencia eventual**: tras el `upsert`, `ingest.py` espera hasta que
  `describe_index_stats` muestre todos los vectores; si no, una consulta inmediata
  podría no verlos.

**Límite de escala:** el BM25 de LangChain vive en memoria y se arma al iniciar
desde los mismos chunks de `data/`. Es perfecto para cientos o miles de chunks; con
millones, la parte léxica debería migrar a vectores *sparse* de Pinecone.

## Evaluación

`golden_set.json` tiene 16 pares `{"pregunta", "documento_id_esperado"}`: 10 preguntas
"semánticas" (parafrasean el texto) y 6 con **términos exactos** (`PMS`, `anafes`,
`$15.000`, teléfono interno `9`, `calle Costanera`, `shuttle`), pensadas para el caso
donde la búsqueda léxica debería aportar. Para cada pregunta se recuperan los top-k
chunks y se mide a nivel documento:

- **Recall@k**: ¿está el documento correcto entre los k recuperados? (0 o 1 por pregunta).
- **Precision@k**: de los k chunks recuperados, ¿qué fracción es del documento correcto?

El reporte compara `vectorial`, `bm25` e `hibrido` sobre las mismas preguntas.
Resultados **contra Pinecone Serverless** (aws/us-east-1) con embeddings de Gemini de 1536 d:

| k | modo | Precision@k | Recall@k |
| --- | --- | --- | --- |
| 5 | vectorial | 0.463 | 1.000 |
| 5 | bm25 | 0.400 | 1.000 |
| 5 | **híbrido** | 0.388 | 1.000 |
| 3 | vectorial | 0.625 | 0.938 |
| 3 | bm25 | 0.583 | 1.000 |
| 3 | **híbrido** | 0.583 | 1.000 |
| 1 | vectorial | 0.875 | 0.875 |
| 1 | bm25 | 0.938 | 0.938 |
| 1 | **híbrido** | 0.938 | 0.938 |

Cómo leerlo, sin maquillar:

- **Con las preguntas de términos exactos aparece la ventaja léxica.** A k=1 y k=3 el
  vectorial falla alguna pregunta (Recall@1 0.875, Recall@3 0.938) que BM25 y el
  híbrido aciertan (0.938 y 1.000). Con las 10 preguntas semánticas solas, los tres
  modos empataban en recall.
- **Recall@5 = 1.0 sigue saturado en los tres modos:** el corpus son 14 chunks y el
  top-5 cubre más de un tercio. Las diferencias reales están en k bajo.
- **Precision@5 tiene un techo de 0.70**, no de 1.0: cada documento tiene 3-4 chunks,
  así que como mucho 3-4 de los 5 recuperados pueden ser suyos. El reporte imprime
  ese techo.
- **En precisión el híbrido no le gana al vectorial** (0.388 vs 0.463 a k=5): BM25
  mete ruido léxico en la cola del ranking. Lo que el híbrido compra es no perder
  el documento correcto cuando el embedding no lo capta, a costa de algo de precisión.
- **Un fallo que ningún modo resuelve:** "¿Por qué concepto se cobran $15.000 por
  estadía?" a k=1 recupera la política de reservas. El monto se tokeniza como `15` y
  `000`, que aparecen en muchos chunks, y el embedding no asocia el monto con mascotas.
  Es un buen candidato para mejorar (tokenizar montos como un solo término).

`python evaluate.py --k 3 --min-recall 0.8` sale con código 1 si el recall
híbrido cae bajo el umbral (útil como chequeo de regresión).

## Tests

Las verificaciones que antes vivían en `validate_offline.py` (chunking, esquemas,
prompt, referencias, cadena LCEL, `get_rag_response`) ahora son tests de `pytest`,
junto con los de la entrega nueva (ingesta, setup del índice, recuperador híbrido,
métricas y golden set). Todo corre **sin red ni API keys**, con dobles de prueba
(`tests/fakes.py`: embeddings deterministas, `Index` y cliente de Pinecone falsos).

```bash
pip install -r requirements-dev.txt
pytest
```

GitHub Actions ([.github/workflows/tests.yml](.github/workflows/tests.yml)) corre la
suite en Python 3.12 y 3.13 en cada push y pull request.

## Errores comunes (y cómo están cubiertos)

| Error | Qué pasa | Cobertura |
| --- | --- | --- |
| Mismatch de dimensiones | Subir 1536 d a un índice de 768 falla en el `upsert` | `setup_pinecone` compara la dimensión del índice existente y falla con un mensaje claro (`test_rechaza_un_indice_con_otra_dimension`) |
| Ignorar el namespace | Búsquedas ruidosas y lentas | Todo `upsert` y `query` va con `namespace=` |
| Chunks mal dimensionados | Chicos: sin contexto; grandes: embedding diluido | ~500 tokens, overlap 50, corte por encabezados |
| Consultar justo después del upsert | Pinecone es eventualmente consistente | `ingest.py` espera al conteo de vectores |
| Cambiar de proveedor de embeddings | Misma dimensión, espacio distinto: resultados aleatorios | `--reset` y nota en `.env.example` |

## Nota sobre cuotas

La key de Gemini del entorno es de *free tier*. La ingesta hace **una** llamada de
embeddings para todos los chunks y `evaluate.py` cachea los embeddings de las
preguntas (cada una se embebe una vez aunque se evalúe en tres modos). Solo
`main.py --generar` usa el LLM de chat.
