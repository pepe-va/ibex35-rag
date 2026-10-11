# Configuration

All runtime settings are defined in `Settings` (`src/core/config.py`) using pydantic-settings. Values are read, in order of precedence, from environment variables and then from the `.env` file in the working directory. Names are case-insensitive, and unknown variables are ignored.

Start from the template:

```bash
cp .env.example .env
```

Inside Docker Compose, the `api` service loads `.env` through `env_file` and overrides `OTEL_EXPORTER_OTLP_ENDPOINT`, `PDF_DIR` and `RAG_DATA_DIR` with container paths. `.env.example` uses Compose service names as hosts (`ollama`, `qdrant`, `redis`); when running the API or the ingestion scripts on the host, use `localhost` instead. The `make extract` and `make ingest` targets already do this for you.

## Contents

- [Models](#models)
- [Vector store](#vector-store)
- [Redis cache and rate limiting](#redis-cache-and-rate-limiting)
- [Semantic cache](#semantic-cache)
- [Retrieval and chunking](#retrieval-and-chunking)
- [Data paths](#data-paths)
- [Observability](#observability)
- [LangSmith](#langsmith)
- [Grafana](#grafana)
- [Application](#application)
- [Test tooling](#test-tooling)
- [Engine constants](#engine-constants)

## Models

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint. Use `http://ollama:11434` inside Compose. |
| `OLLAMA_LLM_MODEL` | `qwen2.5:7b` | Model used for RAG generation, the agent and RAGAS evaluation. |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text:latest` | Dense embedding model. Changing it requires re-creating the collection and re-ingesting. |
| `OLLAMA_ROUTER_MODEL` | *(unset → `OLLAMA_LLM_MODEL`)* | Small model for `/ask` routing. `make models` pulls `qwen2.5:1.5b` when it is not set. |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder. The image runs with Hugging Face in offline mode, so a different model must be added to `infra/docker/backend.Dockerfile` and the image rebuilt. |
| `LLM_TEMPERATURE` | `0.1` | 0.0–2.0. |
| `LLM_REQUEST_TIMEOUT` | `300` | Seconds before an LLM request times out. |

`make models` pulls the models named in `.env` (`OLLAMA_LLM_MODEL`, `OLLAMA_EMBED_MODEL`, `OLLAMA_ROUTER_MODEL`) if they are missing.

## Vector store

| Variable | Default | Description |
|---|---|---|
| `QDRANT_HOST` | `localhost` | Use `qdrant` inside Compose. |
| `QDRANT_PORT` | `6333` | HTTP port. |
| `QDRANT_COLLECTION` | `ibex35` | Collection holding the report chunks. |

## Redis cache and rate limiting

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Use `redis://redis:6379/0` inside Compose. |
| `CACHE_TTL_SECONDS` | `3600` | TTL for exact and semantic cache entries. |
| `RATE_LIMIT_REQUESTS` | `20` | Requests allowed per client IP per window. |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Sliding window length. |

## Semantic cache

| Variable | Default | Description |
|---|---|---|
| `SEMANTIC_CACHE_COLLECTION` | `ibex35_query_cache` | Qdrant collection for cached answers. |
| `SEMANTIC_CACHE_THRESHOLD` | `0.92` | Minimum cosine similarity (0–1) for a cache hit. |

The semantic cache embeds questions with FastEmbed `BAAI/bge-small-en-v1.5` (384 dimensions) in-process. If the model cannot be loaded, the cache disables itself and logs `semantic_cache_disabled`.

## Retrieval and chunking

| Variable | Default | Range | Description |
|---|---|---|---|
| `CHUNK_SIZE` | `1200` | 100–10000 | Characters per chunk. Re-ingest after changing it. |
| `CHUNK_OVERLAP` | `150` | 0–3000 | Overlap between chunks (capped at `CHUNK_SIZE - 1`). |
| `SIMILARITY_TOP_K` | `12` | 1–20 | Candidate pool is `max(2 × SIMILARITY_TOP_K, 10)` for single-company queries. |
| `RERANK_TOP_N` | `8` | 1–10 | Chunks kept after reranking and sent to the LLM. |

## Data paths

| Variable | Default (host) | In Compose | Description |
|---|---|---|---|
| `PDF_DIR` | `./data/raw/pdfs` | `/app/data/raw/pdfs` | Source PDFs. |
| `RAG_DATA_DIR` | `./data/processed/rag_data` | `/app/data/processed/rag_data` | Docling output (Markdown, tables, images). |

`./data` is mounted into the API container, so both the host scripts and the API see the same files.

## Observability

| Variable | Default | Description |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://otel-collector:4317` | OTLP gRPC endpoint for traces. Set to `http://localhost:4317` when running the API on the host. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR`. |
| `RAG_EVAL_INPUT_COST_PER_1K` | `0.0` | Cost per 1000 prompt tokens, for the `rag_eval_cost_per_inference` metric. |
| `RAG_EVAL_OUTPUT_COST_PER_1K` | `0.0` | Cost per 1000 completion tokens. |

## LangSmith

Optional. When enabled, LangChain sends traces to LangSmith and a background task copies LLM token and latency statistics into Prometheus `llm_*` metrics.

| Variable | Default | Description |
|---|---|---|
| `LANGSMITH_TRACING` | `false` | Set to `true` to enable. |
| `LANGSMITH_API_KEY` | — | Required when tracing is enabled. Create one at <https://smith.langchain.com> → Settings → API Keys. |
| `LANGSMITH_PROJECT` | `ibex35-rag` | Project name. |
| `LANGSMITH_ENDPOINT` | `https://api.smith.langchain.com` | API endpoint (change for the EU region or self-hosted). |
| `LANGSMITH_COLLECTOR_INTERVAL_SECONDS` | `60` | How often the collector polls LangSmith. |

Check connectivity with `make langsmith-check`. Note that traces include questions and retrieved context, which leave your machine.

## Grafana

| Variable | Default | Description |
|---|---|---|
| `GRAFANA_ADMIN_USER` | **required** | Grafana admin user. |
| `GRAFANA_ADMIN_PASSWORD` | **required** | Grafana admin password. Change the template value. |

Both are required by the `Settings` model too, so the API and the tests fail to start without them.

## Application

| Variable | Default | Description |
|---|---|---|
| `ENVIRONMENT` | `development` | Reported in the startup log. |
| `CORS_ALLOW_ORIGINS` | `[]` | Browser origins allowed to call the API, as a JSON list (e.g. `["http://localhost:3001"]`). The Chainlit frontend calls the API server-side and needs none. |
| `APP_NAME` | `ibex35-rag` | Reported in logs. |
| `APP_VERSION` | `0.1.0` | Reported by `/health` and the OpenAPI schema. |

## Test tooling

Used only by the end-to-end scripts in `tests/e2e/`.

| Variable | Default | Description |
|---|---|---|
| `API_URL` | `http://localhost:8080` | API under test. |
| `TEST_ENDPOINT` | `/api/v1/ask` | Endpoint used by `make agent_test`. |
| `AGENT_TEST_GENERATOR_BACKEND` | `auto` | Question generator for `make agent_test`: `ollama`, `openai` or `auto`. |
| `AGENT_TEST_GENERATOR_API_KEY` | — | API key for an OpenAI-compatible generator. |
| `OPENAI_API_KEY` | — | Alternative key for the OpenAI generator. |

## Engine constants

These are not environment variables; they are module constants in `src/rag/engine.py`. See [rag-pipeline.md](rag-pipeline.md#tuning-reference) for their effect.

| Constant | Value |
|---|---|
| `PER_COMPANY_K` | 10 |
| `RERANK_MIN_PER_COMPANY` | 3 |
| `MIN_RERANKER_SCORE` | 0.05 |
| `MIN_CHUNKS_FOR_COVERAGE` | 2 |

Other fixed values: the agent's model call limit is 8 per request (`src/agents/financial_agent.py`), the RAGAS evaluation pool has 2 threads (`src/rag/evaluation.py`), and ingestion uses 4 worker threads (`src/rag/vector_store/store.py`).
