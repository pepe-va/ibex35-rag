# IBEX35 RAG

Sistema RAG para consultar resultados financieros de empresas del IBEX35 con documentos PDF, embeddings locales y un stack completo de observabilidad.

## Qué hace

El proyecto combina:

- búsqueda híbrida (densa + sparse BM25) sobre informes financieros PDF
- reranking con cross-encoder para precisión máxima
- generación de respuestas con Ollama
- un agente financiero con memoria de sesión que mezcla RAG con datos de mercado en tiempo real
- observabilidad con métricas, logs y trazas

Flujo principal:

```text
PDFs (data/rag-data/) → Docling → Markdown → chunking → embeddings → Qdrant
Pregunta → API → extract_filters (LLM) → hybrid search → rerank → Ollama → respuesta
```

## Stack actual

| Componente | Tecnología |
|---|---|
| LLM | Ollama (`qwen2.5:7b`) |
| Embeddings | Ollama (`nomic-embed-text:latest`) |
| Extracción PDF | Docling (Markdown estructurado + tablas + imágenes) |
| Framework RAG | LangChain + LangGraph |
| Vector store | Qdrant (hybrid dense cosine + sparse BM25) |
| Reranking | `BAAI/bge-reranker-base` (cross-encoder, CPU) |
| API | FastAPI + Pydantic v2 |
| Agente | LangGraph `create_agent` con `MemorySaver` (memoria de sesión) |
| Caché y rate limit | Redis |
| Logs | Loki + Promtail |
| Métricas | Prometheus |
| Trazas | OpenTelemetry + Tempo |
| Dashboards | Grafana |
| Alertas | Alertmanager |
| Infraestructura | Docker Compose |

El núcleo RAG funciona en local y no necesita API keys externas. El modo agente usa `yfinance`, así que para datos de mercado sí necesita conectividad saliente.

## Requisitos

- Docker y Docker Compose
- GPU NVIDIA con `nvidia-container-toolkit` si quieres acelerar Ollama por GPU
- fichero `.env` a partir de [.env.example](.env.example)

## Arranque rápido

```bash
cp .env.example .env
docker compose up -d
```

La ingesta de PDFs se ejecuta en dos pasos:

```bash
# 1. Extracción Docling (PDFs → Markdown). Solo hace falta una vez.
OLLAMA_BASE_URL=http://localhost:11434 QDRANT_HOST=localhost \
  uv run python scripts/ingest.py --extract-only

# 2. Ingesta vectorial en Qdrant
OLLAMA_BASE_URL=http://localhost:11434 QDRANT_HOST=localhost \
  uv run python scripts/ingest.py
```

O a través de la API una vez levantada:

```bash
curl -X POST http://localhost:8080/api/v1/ingest \
  -H "Content-Type: application/json" -d '{}'
```

## URLs útiles

| Servicio | URL |
|---|---|
| API | http://localhost:8080 |
| API docs | http://localhost:8080/docs |
| Qdrant | http://localhost:6333/dashboard |
| Ollama | http://localhost:11434 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Loki | http://localhost:3100 |
| Tempo | http://localhost:3200 |
| Alertmanager | http://localhost:9093 |

Las credenciales de Grafana salen de `.env`.

## Endpoints principales

### `POST /api/v1/query`

Consulta RAG sobre los PDFs indexados. Los filtros (empresa, año, trimestre) se extraen automáticamente de la pregunta mediante el LLM.

```bash
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Cuál fue el EBITDA de IBERDROLA en 2024?"
  }'
```

Respuesta típica:

```json
{
  "answer": "El EBITDA de IBERDROLA en 2024 fue ...",
  "sources": [
    {
      "company": "IBERDROLA",
      "ticker": "IBE.MC",
      "score": 0.91
    }
  ],
  "latency_seconds": 1.8,
  "from_cache": false,
  "query": "¿Cuál fue el EBITDA de IBERDROLA en 2024?"
}
```

Campos opcionales del body:

| Campo | Tipo | Descripción |
|---|---|---|
| `question` | string | Pregunta (mín. 5, máx. 1000 caracteres) |
| `company_filter` | string\|null | Fuerza el filtro de empresa (si no se pasa, se extrae automáticamente) |
| `thread_id` | string\|null | ID de sesión para memoria de conversación en el agente |
| `use_agent` | bool | Usar el agente ReAct (más lento, combina RAG + datos en tiempo real) |

### `POST /api/v1/agent`

Modo agente para combinar la base documental con datos de mercado. Tiene memoria de sesión.

```bash
curl -X POST http://localhost:8080/api/v1/agent \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Compara el beneficio neto de BBVA con su cotización actual",
    "thread_id": "sesion-123"
  }'
```

### `POST /api/v1/ingest`

Lanza la ingesta/indexación de los Markdown extraídos en Qdrant.

```bash
curl -X POST http://localhost:8080/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Health y métricas

```bash
curl http://localhost:8080/health
curl http://localhost:8080/health/live
curl http://localhost:8080/health/ready
curl http://localhost:8080/metrics
```

## Desarrollo local

```bash
uv sync --dev
cp .env.example .env
```

Para ejecutar la API fuera de Docker, ajusta `.env` para usar `localhost` en lugar de nombres de servicio Docker:

```bash
# OLLAMA_BASE_URL=http://localhost:11434
# QDRANT_HOST=localhost
# REDIS_URL=redis://localhost:6379/0

docker compose up -d qdrant redis ollama prometheus grafana loki tempo otel-collector alertmanager
uv run uvicorn src.api.main:app --reload --port 8080
```

Comandos útiles:

```bash
uv run pytest tests/ -v
uv run pytest tests/unit/ -v
uv run pytest tests/integration/ -v
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
```

## Estructura del proyecto

```text
src/
  api/            FastAPI, rutas, dependencias, caché y rate limiting
  agents/         agente financiero con memoria (LangGraph) y herramientas de mercado
  ingestion/      extracción Docling y pipeline de ingesta vectorial
  monitoring/     métricas Prometheus
  rag/            motor RAG: hybrid search, reranking, generación
  vectorstore/    integración con Qdrant (hybrid dense+BM25)

data/
  rag-data/
    markdown/     Markdown extraído por empresa (Docling)
    tables/       Tablas con contexto por empresa
    images/       Imágenes de páginas visuales

infra/
  alertmanager/
  docker/
  grafana/
  loki/
  otel/
  prometheus/
  promtail/
  tempo/

docs/
  documentación técnica y funcional
```

## Documentación

Toda la documentación vive en [docs](docs/):

- [docs/infra.md](docs/infra.md)
- [docs/RAG.md](docs/RAG.md)
- [docs/observability.md](docs/observability.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/DEVELOPER_MANUAL.md](docs/DEVELOPER_MANUAL.md)
- [docs/USER_MANUAL.md](docs/USER_MANUAL.md)

## Empresas incluidas

ACCIONA, ACERINOX, ACS, AENA, AMADEUS, ARCELORMITTAL, BBVA, CELLNEX, ENAGAS, ENDESA, FERROVIAL, FLUIDRA, GRIFOLS, IAG, IBERDROLA, INDITEX, INDRA, LOGISTA, MAPFRE, MERLINPROPERTIES, NATURGY, PUIG, REDEIA, ROVI, SACYR, SANTANDER, SOLARIA, TELEFONICA y UNICAJA.
