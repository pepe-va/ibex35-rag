# API reference

Base URL: `http://localhost:8080`. The OpenAPI schema is served at `/openapi.json`, with interactive docs at `/docs` (Swagger UI) and `/redoc`.

There is no authentication. Do not expose the API publicly without putting an authenticating proxy in front of it.

## Contents

- [Endpoints](#endpoints)
- [Request body](#request-body)
- [POST /api/v1/ask](#post-apiv1ask)
- [POST /api/v1/ask/stream](#post-apiv1askstream)
- [POST /api/v1/query](#post-apiv1query)
- [POST /api/v1/agent](#post-apiv1agent)
- [POST /api/v1/ingest](#post-apiv1ingest)
- [Health endpoints](#health-endpoints)
- [GET /metrics](#get-metrics)
- [Caching](#caching)
- [Rate limiting](#rate-limiting)
- [Errors](#errors)

## Endpoints

| Method | Path | Rate limited | Cached | Purpose |
|---|---|---|---|---|
| `POST` | `/api/v1/ask` | Yes | RAG route only | Auto-routes between RAG and the agent. |
| `POST` | `/api/v1/ask/stream` | Yes | No | Streaming version of `/ask` (Server-Sent Events). |
| `POST` | `/api/v1/query` | Yes | Yes | RAG only. |
| `POST` | `/api/v1/agent` | Yes | No | Agent only. |
| `POST` | `/api/v1/ingest` | No | — | Runs the ingestion pipeline. |
| `GET` | `/health` | No | — | Dependency health. |
| `GET` | `/health/live` | No | — | Liveness probe. |
| `GET` | `/health/ready` | No | — | Readiness probe. |
| `GET` | `/metrics` | No | — | Prometheus metrics. |

## Request body

`/ask`, `/ask/stream`, `/query` and `/agent` accept the same JSON body (`QueryRequest`):

| Field | Type | Required | Description |
|---|---|---|---|
| `question` | string | Yes | 5 to 1000 characters. Spanish or English. |
| `company_filter` | string \| null | No | Restricts retrieval to one company, e.g. `"IBERDROLA"`. When omitted, companies are detected from the question. Part of the cache key. |
| `use_agent` | bool | No | `/ask` and `/ask/stream` only: forces the agent route. Default `false`. |
| `thread_id` | string \| null | No | Agent session ID. The same value continues a conversation. Without it, each request gets a fresh memory that is not shared. |

## POST /api/v1/ask

Recommended entry point. The route is chosen as follows:

1. `use_agent: true` → agent.
2. Market keywords ("cotización", "precio actual", "en tiempo real", "ticker", "stock price", ...) → agent, also when combined with fundamental keywords.
3. Fundamental keywords only ("EBITDA", "beneficio neto", "deuda", "dividendo", "2024", ...) → RAG.
4. Otherwise a one-word classification by `OLLAMA_ROUTER_MODEL`; RAG if the call fails.

```bash
curl -X POST http://localhost:8080/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Compara el beneficio neto de BBVA y SANTANDER en 2024"}'
```

Response (`AskResponse`):

| Field | Type | Description |
|---|---|---|
| `answer` | string | Generated answer or abstention message. |
| `route` | `"rag"` \| `"agent"` | Route taken. |
| `latency_seconds` | float | Server-side latency. |
| `query` | string | The original question. |
| `sources` | list | RAG only: `{company, ticker, score}` per chunk used. |
| `from_cache` | bool | RAG only: served from a cache. |
| `tools_used` | list[string] | Agent only: tools called, deduplicated. |
| `steps_taken` | int | Agent only: number of tool calls. |
| `steps_detail` | list | Agent only: `{tool, duration_seconds}` per call. |

## POST /api/v1/ask/stream

Same routing as `/ask`, returned as `text/event-stream`. Each event is a line `data: <json>` followed by a blank line.

```bash
curl -N -X POST http://localhost:8080/api/v1/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuál fue el EBITDA de IBERDROLA en 2024?"}'
```

Events:

| `type` | Route | Fields | Meaning |
|---|---|---|---|
| `route` | both | `route` | First event: `"rag"` or `"agent"`. |
| `step_start` | RAG | `step`, `label` | Stage started: `retrieve`, `rerank` or `generate`. |
| `step_end` | RAG | `step`, `duration_seconds`, `sources` (after `rerank`, same fields as `/query` sources plus `content_type` and `page`) | Stage finished. |
| `tool_start` | agent | `tool` | The agent called a tool. |
| `tool_end` | agent | `tool`, `duration_seconds` | The tool returned. |
| `token` | both | `content` | A chunk of the answer text. |
| `done` | both | RAG: `answer`, `latency_seconds`, `sources`, `query`. Agent: `latency_seconds`, `steps_taken`, `tools_used`, `steps_detail` | Final event. |
| `error` | both | `error` | The run failed; the stream ends. |

The streaming RAG path does not use the caches, the out-of-domain check or the coverage gate. Use `/ask` or `/query` when you need those guarantees.

## POST /api/v1/query

RAG only, never calls the agent.

```bash
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was IBERDROLA EBITDA in 2024?"}'
```

```json
{
  "answer": "...",
  "sources": [{ "company": "IBERDROLA", "ticker": "IBE.MC", "score": 0.91 }],
  "latency_seconds": 1.8,
  "from_cache": false,
  "query": "What was IBERDROLA EBITDA in 2024?"
}
```

## POST /api/v1/agent

Agent only. Combines the report knowledge base with live market data through `yfinance` (needs internet access).

```bash
curl -X POST http://localhost:8080/api/v1/agent \
  -H "Content-Type: application/json" \
  -d '{"question": "Compare BBVA net profit with its current share price", "thread_id": "session-123"}'
```

```json
{
  "answer": "...",
  "latency_seconds": 12.4,
  "steps_taken": 2,
  "tools_used": ["ibex35_financial_reports", "get_stock_price"],
  "steps_detail": [
    { "tool": "ibex35_financial_reports", "duration_seconds": 6.1 },
    { "tool": "get_stock_price", "duration_seconds": 0.8 }
  ],
  "query": "Compare BBVA net profit with its current share price"
}
```

## POST /api/v1/ingest

Runs extraction and indexing (see [rag-pipeline.md](rag-pipeline.md#ingestion)). The request blocks until the pipeline finishes and returns `202`.

```bash
curl -X POST http://localhost:8080/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{}'
```

| Body field | Type | Description |
|---|---|---|
| `pdf_dir` | string \| null | Subfolder of `PDF_DIR` to ingest, relative to it (for example `"2024"`). Omit it to ingest `PDF_DIR` itself (`/app/data/raw/pdfs` in the container, `./data/raw/pdfs` on the host). Paths outside `PDF_DIR` return `400`. |

Response (`IngestionResponse`):

```json
{
  "success": true,
  "total_documents": 30,
  "total_nodes": 30,
  "total_companies": 30,
  "duration_seconds": 412.7,
  "companies": ["ACCIONA", "ACERINOX", "..."],
  "errors": []
}
```

`total_documents` is the number of Markdown files produced by extraction and `total_nodes` the number of files newly indexed (already-indexed files are skipped). `success` is `false` when nothing new was indexed. A second request while one is running returns `409 Conflict`.

## Health endpoints

`GET /health` checks Qdrant, Redis and Ollama:

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "vector_store": "healthy",
  "redis": "healthy",
  "ollama": "healthy",
  "collection_count": 15234
}
```

Returns `200` when everything is healthy and `207` with `"status": "degraded"` otherwise.

| Endpoint | Success | Failure |
|---|---|---|
| `GET /health/live` | `200 {"status": "ok"}` | — (process is down) |
| `GET /health/ready` | `200 {"status": "ready"}` | `503 {"status": "not_ready", "reason": "vector_store" \| "vector_store_error"}` when Qdrant is unreachable |

## GET /metrics

Prometheus exposition format. Includes the default HTTP metrics from `prometheus-fastapi-instrumentator` and the application metrics listed in [observability.md](observability.md#metrics).

## Caching

Only RAG answers are cached, in two layers checked in order:

1. **Exact match** in Redis, key `ibex35:query:<first 16 hex chars of sha256(lowercase question | company_filter)>`.
2. **Semantic** in the Qdrant collection `SEMANTIC_CACHE_COLLECTION` (`ibex35_query_cache`): the question is embedded with FastEmbed `BAAI/bge-small-en-v1.5`, and a cached answer is returned if cosine similarity ≥ `SEMANTIC_CACHE_THRESHOLD` (0.92) with the same `company_filter`. Expired entries are deleted on read.

Both use `CACHE_TTL_SECONDS` (default 3600). Cached responses have `from_cache: true`. To clear them, run `docker compose exec redis redis-cli FLUSHDB` and delete the `ibex35_query_cache` collection in Qdrant.

## Rate limiting

A sliding window per client IP, stored in the Redis sorted set `ibex35:ratelimit:<ip>`: at most `RATE_LIMIT_REQUESTS` (20) requests per `RATE_LIMIT_WINDOW_SECONDS` (60) across the query endpoints. When the limit is exceeded the API returns:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 60

{"detail": {"error": "Rate limit exceeded", "limit": 20, "window_seconds": 60}}
```

If Redis is unavailable the limiter lets requests through.

## Errors

| Status | When |
|---|---|
| `422` | Validation error (e.g. `question` shorter than 5 or longer than 1000 characters). |
| `409` | `/ingest` while another ingestion is running. |
| `429` | Rate limit exceeded. |
| `500` | Unhandled error: `{"detail": "Internal server error"}`. The cause is only logged; check `make logs SVC=api` or the trace in Tempo. |
