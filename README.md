# IBEX35 RAG

Sistema RAG para consultar resultados financieros de empresas del IBEX35 con documentos PDF, embeddings locales y un stack completo de observabilidad.

## Qué hace

El proyecto combina:

- búsqueda semántica sobre informes financieros PDF
- generación de respuestas con Ollama
- un agente opcional que mezcla RAG con datos de mercado en tiempo real
- observabilidad con métricas, logs y trazas

Flujo principal:

```text
PDFs -> ingesta -> chunking -> embeddings -> ChromaDB
Pregunta -> API -> retrieval -> Ollama -> respuesta
```

## Stack actual

| Componente | Tecnología |
|---|---|
| LLM | Ollama (`llama3.2:latest`) |
| Embeddings | Ollama (`nomic-embed-text:latest`) |
| Framework RAG | LlamaIndex |
| Vector store | ChromaDB |
| API | FastAPI + Pydantic v2 |
| Frontend | Chainlit |
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

La forma recomendada es usar el `Makefile`:

```bash
cp .env.example .env
make
# o, si prefieres ser explícito:
make start
```

`make` y `make start` hacen lo mismo: validan `.env`, levantan el stack, esperan a `ollama`/`chroma`/`redis`, validan modelos y ejecutan la ingesta inicial si la colección está vacía.

Si prefieres hacerlo a mano:

```bash
cp .env.example .env
docker compose up -d
docker compose exec -T api python scripts/ingest.py
curl http://localhost:8080/health
```

## URLs útiles

| Servicio | URL |
|---|---|
| Frontend | http://localhost:8501 |
| API | http://localhost:8080 |
| API docs | http://localhost:8080/docs |
| ChromaDB | http://localhost:8000 |
| Ollama | http://localhost:11434 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Loki | http://localhost:3100 |
| Tempo | http://localhost:3200 |
| Alertmanager | http://localhost:9093 |

Las credenciales de Grafana salen de `.env`.

## Endpoints principales

### `POST /api/v1/query`

Consulta RAG sobre los PDFs indexados.

```bash
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Cuál fue el EBITDA de IBERDROLA en 2024?",
    "company_filter": "IBERDROLA"
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

### `POST /api/v1/agent`

Modo agente para combinar la base documental con datos de mercado.

```bash
curl -X POST http://localhost:8080/api/v1/agent \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Compara el beneficio neto de BBVA con su cotización actual"
  }'
```

### `POST /api/v1/ingest`

Lanza la ingesta/indexación de PDFs.

```bash
curl -X POST http://localhost:8080/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{}'
```

Nota: la implementación actual ejecuta una nueva pasada de ingesta, pero no borra automáticamente la colección antes de insertar.

### Health y métricas

```bash
curl http://localhost:8080/health
curl http://localhost:8080/health/live
curl http://localhost:8080/health/ready
curl http://localhost:8080/metrics
```

## Operativa habitual

```bash
make
make start
make stop
make restart
make logs SVC=api
make logs SVC=ollama
make status
make ingest
make urls
```

Comandos que conviene recordar:

- `make` o `make start`: levanta el stack completo.
- `make stop`: apaga el stack.
- `make logs SVC=api`: sigue logs del servicio que te interese.
- `make ingest`: lanza la ingesta manualmente.

## Desarrollo local

```bash
uv sync --dev
cp .env.example .env
```

Para ejecutar la API fuera de Docker, ajusta `.env` para apuntar a `localhost` en lugar de a nombres de servicio Docker, luego:

```bash
docker compose up -d chroma redis ollama prometheus grafana loki tempo otel-collector alertmanager
uv run uvicorn src.api.main:app --reload --port 8080
```

Comandos útiles:

```bash
uv run pytest tests/ -v
uv run pytest tests/unit/ -v
uv run pytest tests/integration/ -v
uv run ruff check src/ tests/ scripts/
uv run ruff format src/ tests/ scripts/
uv run mypy src/
make logs SVC=api
```

## Estructura del proyecto

```text
src/
  api/            FastAPI, rutas, dependencias, caché y rate limiting
  agents/         agente financiero y herramientas de mercado
  ingestion/      carga de PDFs y chunking
  monitoring/     métricas Prometheus
  rag/            motor RAG y prompts
  vectorstore/    integración con ChromaDB

infra/
  alertmanager/
  docker/
  grafana/
  loki/
  otel/
  prometheus/
  promtail/
  tempo/

frontend/
  app.py          interfaz Chainlit

docs/
  documentación técnica y funcional
```

## Documentación

Toda la documentación, salvo este `README`, vive ahora en [docs](docs/):

- [docs/infra.md](docs/infra.md)
- [docs/RAG.md](docs/RAG.md)
- [docs/observability.md](docs/observability.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/DEVELOPER_MANUAL.md](docs/DEVELOPER_MANUAL.md)
- [docs/USER_MANUAL.md](docs/USER_MANUAL.md)

## Empresas incluidas

ACCIONA, ACERINOX, ACS, AENA, AMADEUS, ARCELORMITTAL, BBVA, CELLNEX, ENAGAS, ENDESA, FERROVIAL, FLUIDRA, GRIFOLS, IAG, IBERDROLA, INDITEX, INDRA, LOGISTA, MAPFRE, MERLINPROPERTIES, NATURGY, PUIG, REDEIA, ROVI, SACYR, SANTANDER, SOLARIA, TELEFONICA y UNICAJA.
