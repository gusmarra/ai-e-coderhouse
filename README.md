# RAG end-to-end sobre los documentos institucionales del Hotel Bahía Serena

Un pipeline RAG completo: ingesta documentos `.md`, los persiste en
ChromaDB, y responde preguntas del usuario **solo** con lo que esos
documentos dicen. Si la respuesta no está en el contexto recuperado, el
sistema dice que no la tiene — no alucina.

```python
import asyncio
from chain import get_rag_response

resultado = asyncio.run(get_rag_response(
    "¿Cuáles son los horarios de check-in y check-out, y cuánto cuesta un late check-out?"
))
print(resultado.model_dump_json(indent=2))
```

```json
{
  "respuesta": "Check-in: a partir de las 15:00 hs. Check-out: hasta las 11:00 hs. Late check-out: sin cargo hasta las 13:00 hs, 50% de una noche entre las 13:01 y las 18:00 hs, y una noche completa después de las 18:00 hs.",
  "encontrado_en_contexto": true,
  "referencias": [
    {"fuente": "politica-reservas-cancelaciones.md", "fragmento": "## 3. Check-in y check-out\n\n- **Check-in**: a partir de las 15:00 hs...."}
  ]
}
```

Y frente a algo que los documentos no cubren:

```python
asyncio.run(get_rag_response("¿El hotel ofrece servicio de guardería o cuidado de niños (kids club)?"))
# respuesta: "No tengo esa informacion en los documentos disponibles."
# encontrado_en_contexto: false
```

## El "cerebro": documentos institucionales del Hotel Bahía Serena

`data/` tiene 4 documentos institucionales de un mismo hotel ficticio
(Hotel Bahía Serena), redactados para este proyecto, que se referencian
entre sí y comparten vocabulario operativo — el escenario donde un RAG con
`top_k` chico (no todo el corpus) demuestra su valor:

| Archivo | Contenido |
| --- | --- |
| [data/politica-reservas-cancelaciones.md](data/politica-reservas-cancelaciones.md) | Reservas, garantía, check-in/check-out, cancelaciones, reembolsos, métodos de pago |
| [data/reglamento-interno-huespedes.md](data/reglamento-interno-huespedes.md) | Normas de convivencia: horarios de silencio, mascotas, visitas, fumadores, daños |
| [data/servicios-comodidades.md](data/servicios-comodidades.md) | Categorías de habitación, restaurante, spa, piscina, wifi, estacionamiento, traslados |
| [data/protocolo-seguridad-emergencias.md](data/protocolo-seguridad-emergencias.md) | Evacuación, incendios, emergencias médicas, caja de seguridad, cámaras, control de acceso |

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

## Instalación y ejecución

```bash
# 1. Crear entorno e instalar dependencias
python -m venv .venv
source .venv/bin/activate  # o .venv\Scripts\activate en Windows
pip install -r requirements.txt

# 2. Configurar variables de entorno
cp .env.example .env       # y completar GEMINI_API_KEY (o OPENAI_API_KEY)

# 3. Ingesta: fragmenta data/*.md y los persiste en ChromaDB (./vectorstore)
python ingest.py

# 4. Pruebas y verificación
python validate_offline.py # sin red, sin API key: chunking, esquemas, prompt y cadena LCEL con modelo falso
python main.py              # una pregunta respondible + una pregunta trampa, contra la API real
```

`.env` solo necesita una API key: `GEMINI_API_KEY` (o `OPENAI_API_KEY` si
se cambia `LLM_PROVIDER=openai`). **Importante**: el mismo proveedor se usa
para chat y para embeddings — ver la sección de "Embeddings no
coincidentes" más abajo. `ingest.py` es idempotente: si `./vectorstore` ya
tiene chunks, no vuelve a embeberlos (correrlo de nuevo no hace daño); para
reconstruirlo desde cero tras editar los `.md` de `data/`, usar
`python ingest.py --forzar` o `python main.py --forzar`.

### Salida esperada (fragmento real de `python main.py`)

El log completo de una corrida real queda en la sección
["Pruebas realizadas"](#pruebas-realizadas-contra-la-api-real-de-gemini)
más abajo. Este fragmento muestra los tres elementos clave: los chunks
recuperados por el retriever (con su fuente), el fallback a "no lo sé"
cuando el contexto no alcanza, y el JSON validado por Pydantic:

```text
========================================================================
1. Camino feliz: la respuesta esta en los documentos
========================================================================
Pregunta: ¿Cuáles son los horarios de check-in y check-out, y cuánto cuesta un late check-out?
Encontrado en contexto: True
Respuesta: Los horarios del hotel son:
- Check-in: a partir de las 15:00 hs.
- Check-out: hasta las 11:00 hs.
[...]
Referencias:
  - politica-reservas-cancelaciones.md: '## 3. Check-in y check-out\n\n- **Check-in**: a partir de las 15:00 hs...'
  - politica-reservas-cancelaciones.md: '## 5. Reembolsos\n\nLos reembolsos que correspondan por cancelación...'
  - reglamento-interno-huespedes.md: '## 4. Visitas de no-huéspedes\n\nSe permite el ingreso de visitas...'
  - servicios-comodidades.md: '## 4. Piscina\n\nLa piscina exterior está disponible de 08:00 a 20:00 hs...'

========================================================================
2. Pregunta trampa: no deberia estar en los documentos
========================================================================
Pregunta: ¿El hotel ofrece servicio de guardería o cuidado de niños (kids club)?
Encontrado en contexto: False
Respuesta: No tengo esa informacion en los documentos disponibles.
Referencias:
  - servicios-comodidades.md: '## 4. Piscina\n\nLa piscina exterior está disponible de 08:00 a 20:00 hs...'
  - politica-reservas-cancelaciones.md: '## 5. Reembolsos\n\nLos reembolsos que correspondan por cancelación...'
  [...]

JSON (RespuestaRAG, Pydantic):
{
  "respuesta": "No tengo esa informacion en los documentos disponibles.",
  "encontrado_en_contexto": false,
  "referencias": [
    {
      "fuente": "servicios-comodidades.md",
      "fragmento": "## 4. Piscina\n\nLa piscina exterior está disponible de 08:00 a 20:00 hs..."
    },
    {
      "fuente": "politica-reservas-cancelaciones.md",
      "fragmento": "## 5. Reembolsos\n\nLos reembolsos que correspondan por cancelación..."
    }
  ]
}

-> correcto: el modelo no alucino una respuesta que no esta en los documentos.
```

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

`ingest.py::ingerir()` abre la colección `hotel_bahia_serena` de
`./vectorstore` y, si ya tiene chunks, **no vuelve a embeberlos** (evita
gastar cuota de la API de embeddings en cada corrida de `main.py`). Para
reconstruirla desde cero — por ejemplo, después de editar los `.md` de
`data/` — usar `--forzar` o `ingerir(forzar=True)`.

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

## Pruebas realizadas (contra la API real de Gemini)

| Pregunta | `encontrado_en_contexto` | Resultado |
| --- | --- | --- |
| ¿Cuáles son los horarios de check-in y check-out, y cuánto cuesta un late check-out? | `true` | Respondió los tres horarios y los tres tramos de cargo de late check-out, citando `politica-reservas-cancelaciones.md` |
| ¿El hotel ofrece servicio de guardería o cuidado de niños (kids club)? (pregunta trampa) | `false` | "No tengo esa informacion en los documentos disponibles." — no alucinó, a pesar de que el retriever igual devolvió 4 fragmentos de otros temas |

## Nota sobre la cuota de la API

El entorno del curso solo tiene configurada `GEMINI_API_KEY`, de free tier:
**20 requests por día y por modelo** (separado para el modelo de chat y el
de embeddings). Una corrida completa de `main.py` gasta ~1 llamada de
embeddings (la ingesta embebe todos los chunks en un solo batch) + 2
llamadas de embeddings de consulta + 2 llamadas de chat — muy por debajo
del límite. Aun así, para iterar sobre chunking, esquemas o el prompt sin
tocar la red, usar `validate_offline.py`.
