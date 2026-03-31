# IBEX35 RAG — Arquitectura y decisiones técnicas

Documento técnico detallado: por qué existe cada fichero, qué hace internamente y cómo encaja con el resto del sistema.

---

## Índice

1. [Visión general del sistema](#1-visión-general-del-sistema)
2. [Configuración central](#2-configuración-central)
3. [Pipeline de ingesta](#3-pipeline-de-ingesta)
4. [Vector store](#4-vector-store)
5. [RAG engine](#5-rag-engine)
6. [Agente financiero](#6-agente-financiero)
7. [API FastAPI](#7-api-fastapi)
8. [Redis: caché y rate limiting](#8-redis-caché-y-rate-limiting)
9. [MLflow: trazabilidad de experimentos](#9-mlflow-trazabilidad-de-experimentos)
10. [Prometheus y Grafana](#10-prometheus-y-grafana)
11. [Tests](#11-tests)
12. [Docker Compose](#12-docker-compose)
13. [Flujo completo de una petición](#13-flujo-completo-de-una-petición)

---

## 1. Visión general del sistema

Un sistema RAG (*Retrieval-Augmented Generation*) es una arquitectura que separa el **conocimiento** del **razonamiento**:

- El **conocimiento** vive en documentos (en este caso PDFs con resultados financieros de empresas del IBEX35).
- El **razonamiento** lo hace el LLM (en este caso `llama3.2` corriendo localmente con Ollama).
- El **puente** entre ambos es la búsqueda por similitud semántica sobre un vector store (ChromaDB).

La diferencia con un LLM puro es que el modelo no necesita haber visto los datos en su entrenamiento: se los pasamos como contexto en tiempo de inferencia. Esto permite responder sobre datos propietarios o muy recientes.

```
Usuario
  │
  ▼
FastAPI (src/api/main.py)
  │
  ├─── Redis: ¿está en caché? ──→ SÍ: devolver respuesta cacheada
  │                                NO: continuar
  ├─── Rate limiter: ¿ha excedido el límite de peticiones?
  │
  ▼
RAGEngine (src/rag/engine.py)
  │
  ├─── Retriever: busca los K chunks más similares a la pregunta en ChromaDB
  │       └── usa embeddings de nomic-embed-text (Ollama)
  │
  ├─── Synthesizer: manda pregunta + chunks al LLM (llama3.2)
  │       └── usa prompt template en español
  │
  └─── Devuelve respuesta + fuentes + latencia
  │
  ├─── Guardar en Redis
  ├─── Loguear métricas en Prometheus
  └─── Loguear run en MLflow
```

---

## 2. Configuración central

### `src/config.py`

**Por qué existe:** En producción nunca se hardcodean valores en el código. Los parámetros cambian entre entornos (dev, staging, prod) y no deben vivir en el código fuente. Además, los secretos (URLs, credenciales) deben inyectarse desde el entorno.

**Qué hace internamente:**

```python
class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ...
```

`pydantic-settings` hace lo siguiente automáticamente:
1. Lee las variables del fichero `.env`
2. Las sobreescribe con variables de entorno del sistema si existen
3. Valida los tipos (si `chunk_size` no es un int, lanza error en arranque)
4. Expone los valores como atributos tipados

El decorador `@lru_cache` sobre `get_settings()` garantiza que solo se crea **una instancia** de `Settings` durante toda la vida de la aplicación (patrón Singleton). Esto es importante porque `pydantic-settings` lee el disco al instanciarse; hacerlo en cada request sería ineficiente.

**Parámetros clave explicados:**

| Parámetro | Valor | Por qué ese valor |
|---|---|---|
| `chunk_size=512` | 512 tokens | Balance entre contexto suficiente por chunk y no exceder el context window del LLM |
| `chunk_overlap=64` | 64 tokens | Evita que una frase relevante quede cortada entre dos chunks |
| `similarity_top_k=5` | 5 chunks | Recuperamos 5 fragmentos; más aumenta el ruido, menos puede perder información |
| `rerank_top_n=3` | 3 chunks | Tras recuperar 5, si hubiera reranking usaríamos los 3 mejores |
| `llm_temperature=0.1` | 0.1 | Respuestas deterministas (financiero: no queremos creatividad, queremos precisión) |

### `src/logging_config.py`

**Por qué existe:** `print()` no sirve en producción. Necesitamos logs que puedan ser consumidos por sistemas de agregación de logs (Loki, CloudWatch, Datadog). El estándar es JSON estructurado.

**Qué hace internamente:**

`structlog` es una librería de logging estructurado. En lugar de:
```
2024-01-15 10:30:00 INFO Processing query
```
genera:
```json
{"event": "query_complete", "latency": 1.23, "sources": 3, "timestamp": "2024-01-15T10:30:00Z", "level": "info"}
```

La función `setup_logging()` hace tres cosas:
1. Configura `structlog` para que todos los loggers de la app usen JSON
2. Silencia librerías ruidosas (`httpx`, `chromadb`) que generarían demasiado ruido
3. Conecta el sistema de logging de Python estándar (stdlib) con structlog, para que librerías de terceros que usen `logging.getLogger()` también salgan en JSON

`contextvars` permite añadir campos a TODOS los logs de una request sin tener que pasarlos manualmente. En `main.py` hacemos:
```python
structlog.contextvars.bind_contextvars(path=request.url.path, client_ip=...)
```
Y todos los logs generados durante esa request incluirán automáticamente `path` y `client_ip`.

---

## 3. Pipeline de ingesta

### `src/ingestion/pdf_loader.py`

**Por qué existe:** LlamaIndex tiene sus propios loaders de PDF, pero no extraen los metadatos que necesitamos (empresa, ticker, número de página). Un loader propio da control total sobre qué información se asocia a cada chunk.

**Por qué PyMuPDF (fitz) y no PyPDF2 o pdfplumber:**

| Librería | Velocidad | Precisión | Tablas |
|---|---|---|---|
| PyPDF2 | Media | Baja | No |
| pdfplumber | Lenta | Alta | Sí |
| **PyMuPDF** | **Muy rápida** | **Alta** | Parcial |

PyMuPDF es la más rápida y mantiene bien la estructura del texto en PDFs de reports financieros (que son PDFs "digitales", no escaneados).

**Qué hace internamente:**

```python
def _extract_text_by_page(pdf_path: Path) -> list[dict]:
    with fitz.open(str(pdf_path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")  # extrae texto plano
            if text.strip():              # ignora páginas vacías
                pages.append({...})
```

`page.get_text("text")` extrae el texto en orden de lectura. Hay otros modos (`"html"`, `"dict"`, `"blocks"`) que preservan más estructura pero generan más ruido en los embeddings.

**Por qué una página = un Document:**

Si metiéramos todo el PDF como un solo documento, el chunk splitter lo cortaría de forma arbitraria, potencialmente mezclando texto de páginas muy distantes. Partir por páginas primero da chunking más semántico porque cada página del informe tiende a tratar un tema (resultados, deuda, guidance...).

**El `COMPANY_TICKER_MAP`:**

Mapa manual porque los nombres en los ficheros PDF (`IBERDROLA.pdf`) no coinciden exactamente con los símbolos de Yahoo Finance (`IBE.MC`). Esto permite que el agente financiero pueda ir de "la empresa mencionada en el RAG" a "el ticker en el mercado" sin búsquedas adicionales.

**`excluded_llm_metadata_keys`:**

```python
excluded_llm_metadata_keys=["page", "total_pages", "source"],
```

LlamaIndex puede incluir los metadatos en el contexto que manda al LLM. `page` y `total_pages` son datos técnicos que no aportan información financiera y consumen tokens. Los excluimos del contexto del LLM pero los mantenemos para filtrado y trazabilidad.

---

### `src/ingestion/pipeline.py`

**Por qué existe:** Separa la lógica de orquestación del ETL (extract-transform-load) de la lógica de carga de ficheros. Sigue el principio de responsabilidad única.

**El flujo ETL:**

```
PDF files                         ChromaDB
   │                                  │
   ├── load_all_pdfs()                │
   │   └── Document(text, metadata)   │
   │                                  │
   ├── SentenceSplitter               │
   │   └── Node(chunk_text, metadata) │
   │                                  │
   └── VectorStoreManager.add_nodes()─┘
         └── OllamaEmbedding → vector[768]
```

**Por qué `SentenceSplitter` y no `TokenTextSplitter`:**

`TokenTextSplitter` corta por número de tokens exacto, lo que puede cortar frases por la mitad. `SentenceSplitter` intenta respetar los límites de frase — si el chunk de 512 tokens terminaría en medio de "El EBITDA de la compañía fue", busca el final de frase más cercano. Mejor semántica → mejores embeddings → mejores resultados en retrieval.

**Por qué MLflow aquí:**

El pipeline de ingesta es el equivalente al entrenamiento en ML clásico. En MLOps se trackea todo lo que produce un modelo o índice. Aquí el "modelo" es el índice vectorial, y queremos saber:
- Con qué parámetros se creó (chunk_size, modelo de embedding)
- Cuántos documentos procesó
- Cuánto tardó
- Si hubo errores

Esto permite comparar dos ingestions con distintos parámetros y elegir la que produce mejores resultados.

---

## 4. Vector store

### `src/vectorstore/store.py`

**Por qué existe:** Abstrae ChromaDB detrás de una interfaz propia. El resto del código no sabe si el backend es ChromaDB, Qdrant, Pinecone o pgvector — solo llama a `add_nodes()` y `get_index()`. Cambiar de vector store es cambiar una clase.

**Por qué ChromaDB:**

- Open source, sin necesidad de API key
- Fácil de correr en Docker
- Se integra nativamente con LlamaIndex
- Soporta filtrado por metadatos (necesario para `company_filter`)
- Persiste en disco → los embeddings sobreviven reinicios

**Por qué cosine similarity:**

```python
metadata={"hnsw:space": "cosine"}
```

ChromaDB usa HNSW (Hierarchical Navigable Small World), un índice de aproximación de vecinos más cercanos. La métrica de similitud determina cómo se compara la query con los documentos. `cosine` mide el ángulo entre vectores (dirección), ignorando la magnitud. Es la métrica estándar para embeddings de texto porque normaliza la longitud del texto.

**Patrón lazy initialization:**

```python
def _get_client(self) -> chromadb.HttpClient:
    if self._client is None:
        self._client = chromadb.HttpClient(...)
    return self._client
```

El cliente no se crea hasta que se necesita por primera vez. Esto permite que la clase se instancie sin que ChromaDB esté disponible (útil en tests y en arranque de la aplicación antes de que los servicios estén listos).

**`get_index()` devuelve `VectorStoreIndex`:**

Este es el objeto central de LlamaIndex. Encapsula:
- La conexión al vector store
- El modelo de embeddings
- La lógica de búsqueda por similitud

Una vez creado, tanto el `RAGEngine` como el `IngestionPipeline` operan sobre este índice.

---

## 5. RAG engine

### `src/rag/prompts.py`

**Por qué existe:** Los prompts son código. Igual que no hardcodeas una query SQL en el medio de la lógica de negocio, no hardcodeas el prompt en el medio del engine.

**`QA_TEMPLATE` vs `REFINE_TEMPLATE`:**

LlamaIndex tiene varios `response_mode`:
- `compact`: mete todos los chunks en un solo prompt. Más rápido, puede perder matices.
- `refine`: primero genera una respuesta con el primer chunk, luego la refina con cada chunk adicional. Más lento pero más preciso para preguntas que requieren integrar múltiples fuentes.

El `REFINE_TEMPLATE` se usa en el modo `refine` para iterar sobre las fuentes.

**Por qué `excluded_llm_metadata_keys` en los prompts:**

Los templates de LlamaIndex insertan los metadatos del nodo en el contexto por defecto. Sin configuración, el LLM vería:
```
[Fuente: IBERDROLA.pdf, Página: 3, Total páginas: 45]
El EBITDA fue de €7.800M...
```
El dato de página consume tokens sin aportar valor al análisis financiero.

---

### `src/rag/engine.py`

**Por qué existe:** Es el núcleo del sistema. Orquesta retrieval + generación + observabilidad.

**Anatomía de `RAGEngine`:**

```
RAGEngine
├── _llm: Ollama (llama3.2)
│     └── temperature=0.1 (determinista)
│     └── request_timeout=120s (LLMs locales son lentos)
│
└── _query_engine: RetrieverQueryEngine
      ├── retriever: VectorIndexRetriever (top_k=5)
      └── synthesizer: ResponseSynthesizer
            ├── llm → Ollama
            ├── text_qa_template → QA_TEMPLATE
            └── refine_template → REFINE_TEMPLATE
```

**`_build_query_engine_for_company()`:**

```python
filters=MetadataFilters(
    filters=[ExactMatchFilter(key="company", value=company.upper())]
)
```

ChromaDB permite filtrar por metadatos antes de buscar por similitud. Si el usuario pregunta por IBERDROLA con `company_filter="IBERDROLA"`, la búsqueda vectorial solo considera los chunks que tienen `metadata["company"] == "IBERDROLA"`. Esto:
1. Reduce el contexto irrelevante que llega al LLM
2. Mejora la precisión de las respuestas
3. Reduce la latencia (menos vectores a comparar)

**`_extract_sources()`:**

LlamaIndex devuelve junto con la respuesta los nodos que usó (`response.source_nodes`). Cada nodo tiene un `score` (similitud coseno entre 0 y 1). Extraemos estos metadatos para devolvérselos al usuario — es transparencia sobre de dónde vienen los datos.

**MLflow en queries:**

```python
with mlflow.start_run(run_name="query", nested=True):
    mlflow.log_metric("latency_seconds", ...)
    mlflow.log_metric("num_sources", ...)
```

`nested=True` permite que este run sea hijo de un run padre si se llama desde dentro de otro run. Permite agrupar en MLflow todas las queries de una sesión bajo un run padre.

---

## 6. Agente financiero

### `src/agents/tools.py`

**Por qué existe:** Un RAG puro solo puede responder con lo que está en los documentos. Los PDFs de resultados financieros son datos históricos (resultados del último año). Si alguien pregunta "¿cómo cotiza IBERDROLA ahora?", el RAG no puede responder. Las tools dan acceso a datos en tiempo real.

**Por qué `yfinance`:**

- Gratuito, sin API key
- Tiene todos los valores del IBEX35 (sufijo `.MC` = Madrid)
- `fast_info` da datos básicos sin descargar todo el histórico

**Anatomía de cada tool:**

```python
def get_stock_price(company_name: str) -> str:
```

Todas las tools devuelven `str`. Esto es deliberado: el LLM consume texto, no objetos. Devolver un dict haría que el agente tuviera que serializar/deserializar. Una string formateada es directamente consumible.

**`FunctionTool.from_defaults()`:**

LlamaIndex inspecciona el docstring de la función para generar la descripción de la tool. El agente ReAct usa esta descripción para decidir cuándo usar cada tool. Un docstring malo = el agente usa mal las tools.

---

### `src/agents/financial_agent.py`

**Por qué existe:** Combina el RAG (conocimiento estático) con las tools (datos dinámicos) en un agente que puede decidir qué usar en cada momento.

**Por qué ReAct y no un agente clásico:**

ReAct (*Reasoning + Acting*) es un patrón donde el LLM alterna entre:
1. **Thought**: razona sobre qué necesita saber
2. **Action**: llama a una tool
3. **Observation**: observa el resultado
4. **Repeat** hasta tener suficiente información para responder

```
Thought: El usuario pregunta por los fundamentales y el precio de IBERDROLA.
         Necesito datos financieros del RAG y precio actual.
Action: ibex35_financial_reports("EBITDA e ingresos de IBERDROLA 2024")
Observation: EBITDA €7.800M, ingresos €47.000M...
Action: get_stock_price("IBERDROLA")
Observation: Precio €12.50, cambio +0.8%...
Thought: Tengo toda la información para responder.
Answer: IBERDROLA reportó en 2024 unos ingresos de €47.000M y un EBITDA de €7.800M...
        Actualmente cotiza a €12.50 (+0.8% hoy).
```

**`max_iterations=8`:**

Sin límite, un agente con un LLM pequeño (3B) puede entrar en bucles. 8 iteraciones es suficiente para preguntas complejas sin riesgo de loop infinito.

**RAG como una tool más:**

```python
rag_tool = QueryEngineTool(
    query_engine=self.rag_engine._query_engine,
    metadata=ToolMetadata(name="ibex35_financial_reports", description="...")
)
```

El query engine del RAG se envuelve como una tool de LlamaIndex. Para el agente, consultar los documentos es igual que consultar Yahoo Finance — solo es otra acción disponible. Esto permite que el agente decida cuándo usar datos históricos y cuándo datos en tiempo real.

---

## 7. API FastAPI

### `src/api/models.py`

**Por qué existe:** Define el contrato de la API. Pydantic v2 valida automáticamente que las peticiones entrantes cumplen el schema antes de que lleguen al handler.

**`QueryRequest` explicado:**

```python
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    company_filter: str | None = None
    use_agent: bool = False
```

- `min_length=5`: evita consultas triviales ("hi", "?") que desperdiciarían GPU
- `max_length=1000`: evita prompt injection excesivamente largo
- `company_filter`: permite pre-filtrar el vector store antes de buscar
- `use_agent`: el usuario elige explícitamente si quiere el agente (más lento, más potente)

**Por qué `SourceDoc` como modelo separado:**

```python
class SourceDoc(BaseModel):
    company: str
    ticker: str
    score: float
```

Podríamos devolver los sources como `list[dict]`, pero un modelo tipado:
1. Garantiza que siempre tienen los campos correctos
2. Aparece documentado en el OpenAPI schema (`/docs`)
3. Es más fácil de testear

---

### `src/api/dependencies.py`

**Por qué existe:** FastAPI usa *dependency injection* para gestionar recursos compartidos. El problema sin DI sería que cada request crearía una nueva instancia de `RAGEngine`, que a su vez crearía una nueva conexión a ChromaDB y cargaría el modelo de embeddings — latencia de varios segundos por request.

**`@lru_cache` en los getters:**

```python
@lru_cache
def get_rag_engine():
    ...
```

`lru_cache` convierte la función en un singleton. La primera llamada crea el objeto; las siguientes devuelven el mismo objeto cacheado. El `RAGEngine` se crea una sola vez al arrancar la app y se reutiliza en todas las requests.

**Redis como dependencia async:**

```python
async def get_redis() -> aioredis.Redis:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.from_url(settings.redis_url)
    return _redis_pool
```

`aioredis` es el cliente async de Redis. Se usa `from_url()` que internamente crea un connection pool — múltiples requests pueden usar Redis concurrentemente sin esperar.

---

### `src/api/main.py`

**Por qué existe:** Es el punto de entrada de la aplicación. Configura el servidor, los middlewares, el instrumentador de Prometheus y registra los routers.

**`lifespan` context manager:**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("app_starting", ...)
    yield
    await close_redis()
```

FastAPI moderno usa `lifespan` en lugar de `@app.on_event("startup")`. El código antes del `yield` se ejecuta al arrancar; el código después al apagar. Garantiza que Redis se cierra limpiamente al hacer `SIGTERM` (lo que hace Docker al parar un contenedor).

**El middleware de logging:**

```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    structlog.contextvars.bind_contextvars(path=..., client_ip=...)
    response = await call_next(request)
    logger.info("request_handled", status_code=..., duration=...)
```

Se ejecuta en cada request. Añade `path` y `client_ip` al contexto de structlog para que aparezcan en TODOS los logs de esa request, aunque se generen en capas internas.

**`Instrumentator`:**

```python
Instrumentator(...).instrument(app).expose(app, endpoint="/metrics")
```

`prometheus-fastapi-instrumentator` intercepta todas las requests y genera automáticamente métricas de:
- Número de requests por endpoint y status code
- Latencia por endpoint (histograma)
- Requests en curso

Lo expone en `/metrics` en el formato que Prometheus espera.

---

## 8. Redis: caché y rate limiting

### `src/api/cache.py`

**Por qué existe:** LLMs son lentos (1-30 segundos por query). Si dos usuarios hacen la misma pregunta, no tiene sentido ejecutar el pipeline completo dos veces. La caché guarda la respuesta por un tiempo configurable.

**Diseño de la clave de caché:**

```python
def _make_cache_key(question: str, company_filter: str | None) -> str:
    raw = f"{question.lower().strip()}|{company_filter or ''}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"ibex35:query:{digest}"
```

- `lower().strip()`: normaliza para que "¿Ingresos de IBERDROLA?" y "¿ingresos de iberdrola?" sean la misma clave
- `sha256`: convierte la pregunta en una clave de longitud fija (evita claves muy largas en Redis)
- `[:16]`: los primeros 16 caracteres del hash son suficientes — la probabilidad de colisión es 1 en 2^64
- Prefijo `ibex35:query:`: permite borrar todas las queries con `SCAN ibex35:query:*`

**TTL (Time-To-Live):**

```python
await redis.setex(key, ttl, json.dumps(response))
```

`setex` = SET + EXpiry. La caché expira automáticamente sin necesidad de un job de limpieza. Por defecto 1 hora (`cache_ttl_seconds=3600`). Los datos financieros cambian poco durante el día, así que 1 hora es razonable.

---

### `src/api/rate_limiter.py`

**Por qué existe:** Sin rate limiting, un usuario malicioso (o un bug en el cliente) podría hacer miles de requests por segundo, saturando la GPU con inferencias de Ollama.

**Sliding window vs fixed window:**

- **Fixed window**: "máximo 20 requests por minuto, el minuto empieza a las :00". Problema: permite 40 requests en 2 segundos (20 al final del minuto 1 + 20 al inicio del minuto 2).
- **Sliding window**: "máximo 20 requests en los últimos 60 segundos". Más justo, sin el spike del boundary.

**Implementación con Redis sorted sets:**

```python
pipe.zremrangebyscore(key, 0, window_start)  # eliminar entradas antiguas
pipe.zcard(key)                               # contar entradas actuales
pipe.zadd(key, {str(now): now})              # añadir la request actual
pipe.expire(key, window_seconds * 2)         # limpiar la clave eventualmente
```

Un sorted set de Redis almacena elementos con un score numérico. Usamos el timestamp Unix como score. Para saber cuántas requests hay en la ventana deslizante:
1. Eliminar todas las entradas con score < `now - window`
2. Contar las que quedan
3. Si el contador excede el límite, rechazar

**`pipeline()`:**

Las 4 operaciones se ejecutan en un solo round-trip a Redis (pipeline = batch de comandos). Esto es crítico para la latencia — hacer 4 round-trips individuales sería 4x más lento.

**Fail-open:**

```python
except Exception as exc:
    logger.warning("rate_limit_redis_error", error=str(exc))
    # No raise — la request continúa
```

Si Redis está caído, el rate limiter falla silenciosamente y deja pasar las requests. Decisión de diseño: mejor servir sin rate limit que no servir nada. En producción real esto dependería del threat model.

---

## 9. MLflow: trazabilidad de experimentos

**Por qué MLflow y no solo logs:**

Los logs son texto plano. MLflow estructura los datos de experimentos en una base de datos con UI, comparación entre runs, y API para consultas programáticas. Es el estándar de facto para MLOps.

**Dos tipos de runs en este sistema:**

**Ingestion runs** (`src/ingestion/pipeline.py`):
```
Run: "ingestion"
├── params: pdf_dir, chunk_size, chunk_overlap, embed_model
└── metrics: total_documents, total_nodes, avg_nodes_per_doc, duration_seconds
```
Permite responder: "¿Qué chunk_size produce más nodos de calidad?" o "¿Cuánto tardó la última ingesta?"

**Query runs** (`src/rag/engine.py`):
```
Run: "query" (nested)
├── params: question_length, llm_model, similarity_top_k, company_filter?
└── metrics: latency_seconds, num_sources, answer_length
```
Permite responder: "¿Las queries con company_filter son más rápidas?" o "¿Cuál es la latencia media del LLM?"

**`nested=True`:**

En MLflow se pueden anidar runs. Un run "sesión" podría contener múltiples runs "query". En nuestro sistema cada query es un run independiente, pero `nested=True` permite que si en el futuro se llama desde un run padre, se anide correctamente.

**PostgreSQL como backend:**

Por defecto MLflow usa un fichero SQLite. En producción usamos PostgreSQL porque:
- Soporta accesos concurrentes (múltiples workers de la API logueando simultáneamente)
- Más robusto para volúmenes grandes de experimentos
- Permite backups estándar de base de datos

---

## 10. Prometheus y Grafana

### `src/monitoring/metrics.py`

**Por qué métricas propias además de las del Instrumentator:**

`prometheus-fastapi-instrumentator` mide métricas HTTP genéricas (latencia, status codes). Nuestras métricas miden el negocio:

```python
CACHE_HITS = Counter("ibex35_rag_cache_hits_total", ...)
QUERY_LATENCY = Histogram("ibex35_rag_query_latency_seconds", labelnames=["company"], ...)
VECTOR_STORE_COUNT = Gauge("ibex35_rag_vector_store_documents_total", ...)
```

**Counter vs Histogram vs Gauge:**

- **Counter**: solo sube, nunca baja. Para eventos acumulativos (requests, errores, cache hits). En Prometheus se consulta con `rate()` para obtener velocidad.
- **Histogram**: distribuye observaciones en buckets. Para latencias y tamaños. Permite calcular percentiles (p50, p95, p99) con `histogram_quantile()`.
- **Gauge**: puede subir y bajar. Para valores que fluctúan (documentos en el vector store, memoria usada, conexiones activas).

**Buckets del histograma de latencia:**

```python
buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0)
```

Los buckets deben cubrir el rango esperado de valores con granularidad en los rangos más importantes. Sabemos que `llama3.2:3b` en una RTX 4060 tarda entre 1-10 segundos en responder, así que ponemos más buckets en ese rango.

**Labels en las métricas:**

```python
QUERY_REQUESTS.labels(company=body.company_filter or "all", cached="true").inc()
```

Los labels crean series de tiempo separadas. Esto permite en Grafana filtrar "¿cuántas queries a IBERDROLA?" vs "¿cuántas queries a SANTANDER?".

### `infra/grafana/dashboards/ibex35_rag.json`

El dashboard de Grafana está en JSON y se **provisiona automáticamente** al arrancar Grafana. No hace falta entrar a la UI y configurar nada manualmente.

**Queries PromQL relevantes:**

```promql
# Cache hit rate en %
rate(ibex35_rag_cache_hits_total[5m]) /
(rate(ibex35_rag_cache_hits_total[5m]) + rate(ibex35_rag_cache_misses_total[5m])) * 100

# Percentil 95 de latencia en los últimos 5 minutos
histogram_quantile(0.95, rate(ibex35_rag_query_latency_seconds_bucket[5m]))
```

---

## 11. Tests

### `tests/conftest.py`

**Por qué existe:** pytest carga automáticamente `conftest.py` y hace disponibles sus fixtures en todos los tests del directorio y subdirectorios. Centraliza la configuración compartida.

**`fakeredis`:**

```python
async def fake_redis():
    r = fakeredis.FakeRedis()
    yield r
```

`fakeredis` implementa el protocolo de Redis en memoria. Los tests de caché y rate limiter se ejecutan sin necesidad de un servidor Redis real. Ventajas:
- Tests más rápidos (sin I/O de red)
- Tests deterministas (sin estado compartido entre tests)
- Funcionan en CI sin servicios externos

---

### `tests/unit/test_pdf_loader.py`

Tests de la lógica de carga de PDFs. El test `test_load_pdf_as_documents` usa `@patch("src.ingestion.pdf_loader.fitz.open")` para interceptar la llamada a PyMuPDF y devolver un documento mock. Esto permite testear la lógica de extracción sin necesidad de un PDF real.

### `tests/unit/test_cache.py`

Tests del sistema de caché usando `fakeredis`. Incluye un test de TTL que espera 1.1 segundos para verificar que la clave expira correctamente.

### `tests/unit/test_rate_limiter.py`

Tests del rate limiter. Verifica que el límite se aplica por IP independientemente — un usuario saturando el sistema no bloquea a otros.

### `tests/integration/test_api.py`

Tests de los endpoints de FastAPI. Usa `TestClient` (cliente HTTP síncrono de Starlette) y mockea todas las dependencias pesadas (ChromaDB, Ollama, Redis). Testa el comportamiento HTTP: status codes, validación de parámetros, estructura de la respuesta.

---

## 12. Docker Compose

### `docker-compose.yml`

**Servicios y por qué cada uno:**

| Servicio | Imagen | Puerto | Propósito |
|---|---|---|---|
| `api` | build local | 8080 | FastAPI application |
| `ollama` | `ollama/ollama` | 11434 | LLM + embeddings local |
| `chroma` | `chromadb/chroma` | 8000 | Vector store |
| `redis` | `redis:7-alpine` | 6379 | Caché + rate limiting |
| `mlflow` | `ghcr.io/mlflow/mlflow` | 5000 | Experiment tracking UI |
| `postgres` | `postgres:16-alpine` | 5432 | Backend de MLflow |
| `prometheus` | `prom/prometheus` | 9090 | Scraping y almacenamiento de métricas |
| `grafana` | `grafana/grafana` | 3000 | Dashboards de métricas |

**GPU en Docker:**

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Esta configuración de Docker Compose le da acceso a la GPU NVIDIA al contenedor de Ollama. Requiere `nvidia-container-toolkit` instalado en el host.

**Health checks y `depends_on`:**

```yaml
depends_on:
  ollama:
    condition: service_healthy
```

`condition: service_healthy` significa que la API no arranca hasta que Ollama responde correctamente a su healthcheck. Evita el problema de que la API arranque antes de que los servicios que necesita estén listos.

**Redis configuración:**

```yaml
command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
```

- `--appendonly yes`: persiste las operaciones en disco (los datos de caché sobreviven reinicios)
- `--maxmemory 512mb`: límite de memoria para evitar que Redis consuma toda la RAM
- `--maxmemory-policy allkeys-lru`: cuando se llega al límite, elimina el elemento usado menos recientemente

### `infra/docker/Dockerfile`

**Multi-stage build:**

```dockerfile
FROM python:3.11-slim AS builder
# instala dependencias

FROM python:3.11-slim AS runtime
# solo copia el .venv y el código
```

La imagen final no contiene `uv`, compiladores ni headers de C. Solo el `.venv` ya construido y el código fuente. Esto reduce el tamaño de la imagen de ~2GB a ~400MB.

**Usuario no-root:**

```dockerfile
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser
```

Por seguridad, el proceso de la API no corre como root. Si hay una vulnerabilidad de ejecución remota de código, el atacante obtiene permisos de `appuser`, no de root.

---

## 13. Flujo completo de una petición

Para concluir, el flujo completo de `POST /api/v1/query`:

```
1. Cliente → POST /api/v1/query {"question": "¿Cuál es el EBITDA de IBERDROLA?"}

2. main.py middleware
   └── log_requests: registra path + IP en contextvars
   └── Instrumentator: empieza a medir latencia HTTP

3. routes/query.py::query_rag()
   │
   ├── get_redis() → obtiene conexión Redis del pool
   │
   ├── check_rate_limit(request, redis, settings)
   │   └── ZREMRANGEBYSCORE + ZCARD + ZADD + EXPIRE (pipeline Redis)
   │   └── Si count >= 20: raise HTTPException 429
   │
   ├── get_cached_response(redis, question, company_filter)
   │   └── SHA256(question.lower()) → busca en Redis
   │   └── Si HIT: CACHE_HITS.inc() + devolver respuesta cacheada
   │
   ├── [CACHE MISS] CACHE_MISSES.inc()
   │
   ├── get_rag_engine() → singleton RAGEngine (ya instanciado)
   │
   └── QUERY_LATENCY.labels(company="all").time():
       └── rag_engine.query("¿Cuál es el EBITDA de IBERDROLA?")
           │
           ├── mlflow.start_run("query")
           │   └── log_param(llm_model, similarity_top_k, ...)
           │
           ├── retriever.retrieve(question)
           │   └── OllamaEmbedding("¿Cuál es el EBITDA de IBERDROLA?")
           │       → vector[768] (nomic-embed-text)
           │   └── ChromaDB.query(vector, top_k=5)
           │       → [chunk_iberdrola_p3, chunk_iberdrola_p7, ...]
           │
           ├── synthesizer.synthesize(question, nodes)
           │   └── QA_TEMPLATE.format(context=chunks, query=question)
           │   └── Ollama(llama3.2).complete(prompt)
           │       → "El EBITDA de IBERDROLA en 2024 fue de €7.800M..."
           │
           └── mlflow.log_metric(latency, num_sources, answer_length)

4. set_cached_response(redis, question, response, ttl=3600)
   └── SETEX ibex35:query:{hash} 3600 {json}

5. QUERY_REQUESTS.labels(company="all", cached="false").inc()

6. return QueryResponse(answer=..., sources=[...], latency=1.8, from_cache=False)

7. Instrumentator: registra latencia HTTP en histograma Prometheus
8. log_requests: loguea {"event": "request_handled", "status_code": 200, "duration": 1.82}
```

Total: ~1.8 segundos para una query sin caché en RTX 4060. Con caché: ~5ms.
