# IBEX35 RAG

Sistema RAG (*Retrieval-Augmented Generation*) para consultar resultados financieros de las empresas del IBEX35. Permite hacer preguntas en lenguaje natural sobre los informes anuales y combinarlas con datos de mercado en tiempo real.

## Stack

| Componente | Tecnología |
|---|---|
| LLM + Embeddings | Ollama (`llama3.2`, `nomic-embed-text`) |
| RAG framework | LlamaIndex |
| Vector store | ChromaDB |
| API | FastAPI + Pydantic v2 |
| Caché + Rate limiting | Redis |
| Experiment tracking | MLflow |
| Métricas | Prometheus + Grafana |
| Infraestructura | Docker Compose |

Todo corre localmente. No requiere API keys externas.

## Requisitos

- Docker y Docker Compose
- NVIDIA GPU con `nvidia-container-toolkit` (para Ollama con GPU)
- Ollama con los modelos descargados:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Arranque rápido

```bash
# 1. Clonar y entrar al directorio
git clone <repo>
cd RAG

# 2. Configurar entorno
cp .env.example .env

# 3. Levantar todos los servicios
docker compose up -d

# 4. Esperar a que estén listos (~60s) y verificar
curl http://localhost:8080/health

# 5. Indexar los PDFs (solo la primera vez)
docker compose exec api python scripts/ingest.py
```

## Servicios disponibles

| Servicio | URL | Descripción |
|---|---|---|
| API | http://localhost:8080/docs | Swagger UI con todos los endpoints |
| Grafana | http://localhost:3000 | Dashboards (`admin` / `ibex35rag`) |
| MLflow | http://localhost:5000 | Experiment tracking |
| Prometheus | http://localhost:9090 | Métricas raw |
| ChromaDB | http://localhost:8000 | Vector store |

## Endpoints principales

### `POST /api/v1/query` — Consulta RAG

Busca en los informes financieros y genera una respuesta con el LLM.

```bash
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Cuál fue el EBITDA de IBERDROLA en 2024?",
    "company_filter": "IBERDROLA"
  }'
```

```json
{
  "answer": "El EBITDA de IBERDROLA en 2024 fue de 7.800M€...",
  "sources": [{"company": "IBERDROLA", "ticker": "IBE.MC", "score": 0.91}],
  "latency_seconds": 1.8,
  "from_cache": false
}
```

### `POST /api/v1/agent` — Agente financiero

Combina el RAG con datos de mercado en tiempo real (yfinance). Más lento pero puede responder preguntas que mezclan fundamentales y precio actual.

```bash
curl -X POST http://localhost:8080/api/v1/agent \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Cómo compara el PER de SANTANDER con su beneficio neto reportado?"
  }'
```

### `POST /api/v1/ingest` — Reindexa los PDFs

```bash
curl -X POST http://localhost:8080/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{}'
```

### `GET /health` — Estado del sistema

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "healthy",
  "vector_store": "healthy",
  "redis": "healthy",
  "ollama": "healthy",
  "collection_count": 1240
}
```

## Desarrollo local

```bash
# Instalar dependencias
uv sync --dev

# Crear .env para desarrollo local
cp .env.example .env
# Editar .env: cambiar hostnames de servicios a localhost

# Lanzar servicios de soporte (sin la API)
docker compose up chroma redis mlflow postgres prometheus grafana -d

# Arrancar la API localmente
uv run uvicorn src.api.main:app --reload --port 8080

# Indexar PDFs
uv run python scripts/ingest.py

# Ejecutar tests
uv run pytest tests/ -v

# Linting
uv run ruff check src/ tests/
```

## Estructura del proyecto

```
src/
├── config.py                 # Settings centralizados (pydantic-settings)
├── logging_config.py         # Logging JSON estructurado (structlog)
├── ingestion/
│   ├── pdf_loader.py         # Carga PDFs con PyMuPDF, extrae metadata
│   └── pipeline.py           # Orquesta ETL: load → chunk → embed → store
├── vectorstore/
│   └── store.py              # Abstracción ChromaDB + LlamaIndex
├── rag/
│   ├── engine.py             # Query engine: retrieval + generación
│   └── prompts.py            # Templates de prompts en español/inglés
├── agents/
│   ├── tools.py              # Tools yfinance: precio, histórico, comparativa
│   └── financial_agent.py    # ReAct agent: RAG + tools en tiempo real
├── api/
│   ├── main.py               # FastAPI app, middlewares, Prometheus
│   ├── models.py             # Pydantic v2 request/response schemas
│   ├── dependencies.py       # DI: singletons de RAG, Redis, agent
│   ├── cache.py              # Redis cache con SHA-256 key
│   ├── rate_limiter.py       # Sliding window rate limit por IP
│   └── routes/
│       ├── query.py          # POST /query y /agent
│       ├── ingest.py         # POST /ingest
│       └── health.py         # GET /health, /health/live, /health/ready
└── monitoring/
    └── metrics.py            # Counters, Histograms y Gauges de Prometheus

infra/
├── docker/Dockerfile         # Multi-stage build, usuario no-root
├── prometheus/prometheus.yml # Configuración de scraping
└── grafana/
    ├── provisioning/         # Auto-provisioning de datasource y dashboards
    └── dashboards/           # Dashboard JSON con métricas RAG
```

## Empresas incluidas

29 empresas del IBEX35: ACCIONA, ACERINOX, ACS, AENA, AMADEUS, ARCELORMITTAL, BBVA, CELLNEX, ENAGAS, ENDESA, FERROVIAL, FLUIDRA, GRIFOLS, IAG, IBERDROLA, INDITEX, INDRA, LOGISTA, MAPFRE, MERLINPROPERTIES, NATURGY, PUIG, REDEIA, ROVI, SACYR, SANTANDER, SOLARIA, TELEFONICA, UNICAJA.

## Documentación técnica

Ver [ARCHITECTURE.md](ARCHITECTURE.md) para una explicación detallada de cada componente, las decisiones de diseño y el flujo completo de una petición.
