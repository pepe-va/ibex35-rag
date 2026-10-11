# Observability

The stack exports metrics, logs and traces, and correlates them through the trace ID. Everything is provisioned automatically when you run `make start`.

## Contents

- [Overview](#overview)
- [Grafana](#grafana)
- [Metrics](#metrics)
  - [Requests and latency](#requests-and-latency)
  - [Cache](#cache)
  - [Retrieval](#retrieval)
  - [Ingestion and index](#ingestion-and-index)
  - [RAG quality](#rag-quality)
  - [LLM (via LangSmith)](#llm-via-langsmith)
  - [Infrastructure](#infrastructure)
- [Logs](#logs)
- [Traces](#traces)
- [Alerts](#alerts)
- [Useful queries](#useful-queries)

## Overview

| Signal | Producer | Pipeline | Where to look |
|---|---|---|---|
| Metrics | API `/metrics`, node-exporter, cAdvisor, redis-exporter, OTel Collector | Prometheus scrapes every 15 s | Grafana dashboards, <http://localhost:9090> |
| Logs | structlog JSON on stdout of every container | Promtail (Docker service discovery) → Loki | Grafana → Explore → Loki |
| Traces | OpenTelemetry SDK in the API (FastAPI, Redis, HTTPX instrumentation + manual RAG spans) | OTLP gRPC → OTel Collector → Tempo | Grafana → Explore → Tempo |
| Alerts | Prometheus rules | Alertmanager | <http://localhost:9093> |
| LLM usage | LangChain tracing | LangSmith → background collector → Prometheus `llm_*` | Grafana, LangSmith UI |

Configuration lives in `infra/`:

```text
infra/
  prometheus/prometheus.yml        scrape targets
  prometheus/rules/rag_alerts.yml  alert rules
  alertmanager/alertmanager.yml    routing (null receiver)
  grafana/provisioning/            datasources (Prometheus, Loki, Tempo, Alertmanager) and dashboard provider
  grafana/dashboards/              dashboard JSON
  loki/loki.yml, promtail/promtail.yml
  tempo/tempo.yml, otel/otel_collector.yml
```

## Grafana

<http://localhost:3000>, credentials from `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD`.

Provisioned dashboards:

- **IBEX35 RAG** (`ibex35_rag_todo.json`): all-in-one view with health cards, API backend, vector store, Redis cache, LLM and RAG quality panels.
- **IBEX35 RAG — v2** (`rag_monitor_v2.json`): a reorganized version with an overview row.

Datasources are linked both ways: from a Loki log line, the `trace_id` field opens the trace in Tempo; from a Tempo span, you can jump to the logs with the same trace ID.

## Metrics

All application metrics are defined in `src/observability/metrics.py`. HTTP metrics (`http_requests_total`, `http_request_duration_seconds`, ...) come from `prometheus-fastapi-instrumentator`; `/health/live`, `/health/ready` and `/metrics` are excluded.

### Requests and latency

| Metric | Type | Labels | Description |
|---|---|---|---|
| `rag_query_requests_total` | Counter | `company`, `cached` | RAG requests. `company` is the `company_filter` or `all`. |
| `rag_query_latency_seconds` | Histogram | `company` | End-to-end RAG latency (uncached). |
| `rag_agent_requests_total` | Counter | — | Agent requests. |
| `rag_agent_latency_seconds` | Histogram | — | Agent latency. |
| `rag_eval_stage_latency_seconds` | Histogram | `stage` | Per-stage latency: `extract_filters`, `retrieve`, `rerank`, `generate`. |

### Cache

| Metric | Type | Description |
|---|---|---|
| `rag_cache_hits_total` | Counter | Exact or semantic cache hits. |
| `rag_cache_misses_total` | Counter | Exact cache misses. |
| `rag_cache_keys_current` | Gauge | Keys in Redis. |
| `rag_cache_bytes_used` / `rag_cache_bytes_limit` | Gauge | Redis memory used and `maxmemory`. |
| `rag_cache_evictions_current` | Gauge | Keys evicted by Redis since it started. |

Cache and vector-store gauges are refreshed every 30 seconds by a background task in the API. Target hit rate is above 40% in steady state; evictions should stay at 0.

### Retrieval

| Metric | Type | Description |
|---|---|---|
| `rag_retrieved_chunks` | Histogram | Chunks kept after reranking, per query. |
| `rag_retrieval_score` | Histogram | Normalized (0–1) reranker score of each chunk. |
| `rag_empty_retrieval_total` | Counter | Queries with no chunks. |
| `rag_context_bytes` | Histogram | Size of the context sent to the LLM. |
| `rag_top_k_current` | Gauge | Current `SIMILARITY_TOP_K`. |

### Ingestion and index

| Metric | Type | Description |
|---|---|---|
| `rag_ingestion_runs_total` | Counter | Ingestion runs, labelled `status` = `success` / `error`. |
| `rag_ingestion_duration_seconds` | Histogram | Ingestion duration. |
| `rag_ingestion_last_duration_seconds` | Gauge | Duration of the last run. |
| `rag_vector_store_documents_current` | Gauge | Points in the Qdrant collection. |
| `rag_companies_indexed_current` | Gauge | Distinct companies indexed. |
| `rag_pages_indexed_current` | Gauge | Distinct (file, page) pairs indexed. |
| `rag_chunks_created_current` | Gauge | Chunks created in the last run. |
| `rag_chunk_size_bytes` | Histogram | Size of each chunk. |
| `rag_chunking_duration_seconds` | Histogram | Time to chunk one file. |

Ingestion metrics are only recorded by the process that runs the ingestion. `make ingest` runs on the host, so its counters are not visible in Prometheus; use `POST /api/v1/ingest` if you want them on the dashboards.

### RAG quality

Recorded after each uncached RAG query; see [rag-pipeline.md](rag-pipeline.md#7-online-evaluation) for how each value is computed. All are 0–1 histograms unless noted.

| Metric | Good value | Meaning |
|---|---|---|
| `rag_eval_faithfulness` | > 0.7 (< 0.4 suggests hallucination) | RAGAS: share of answer claims supported by the context. |
| `rag_eval_answer_relevancy` | > 0.7 | RAGAS: how well the answer addresses the question. |
| `rag_eval_contextual_precision` | > 0.6 | Relevant chunks ranked first. |
| `rag_eval_contextual_recall` | > 0.6 | Question terms covered by the context. |
| `rag_eval_chunk_utilization_ratio` | — | Share of relevant chunks in the context. |
| `rag_eval_chunk_attribution_ratio` | — | Share of chunks that contributed to the answer. |
| `rag_eval_answer_similarity` / `rag_eval_answer_correctness` | — | Lexical agreement with the top chunk. |
| `rag_eval_context_cliff_score` | low | Degradation risk with large contexts. |
| `rag_eval_abstention_ratio` | < 0.15 | 1 per abstaining answer, 0 otherwise. If it rises, data is probably missing. |
| `rag_eval_cost_per_inference` | — | Estimated cost (0 for local models unless costs are configured). |

RAGAS runs in the background with the local LLM, so faithfulness and relevancy appear some seconds after the answer and add GPU load.

### LLM (via LangSmith)

Only populated when `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are set. The collector polls LangSmith every `LANGSMITH_COLLECTOR_INTERVAL_SECONDS`.

| Metric | Type | Reference (qwen2.5:7b) |
|---|---|---|
| `llm_runs_total` | Counter | Grows with query and agent requests. |
| `llm_errors_total` | Counter | < 1% of runs; alert above 5%. |
| `llm_prompt_tokens` | Histogram | p50 800–2500, p95 < 6000. |
| `llm_completion_tokens` | Histogram | p50 100–600. A p50 below 20 suggests empty answers or mass abstention. |
| `llm_total_tokens` | Histogram | p50 < 3500. |
| `llm_request_duration_seconds` | Histogram | GPU p50 1–4 s; CPU p50 4–10 s. |
| `llm_tokens_per_second` | Histogram | GPU p50 25–60; CPU p50 5–15. A sustained drop indicates GPU/CPU saturation. |

### Infrastructure

| Job | Source | Examples |
|---|---|---|
| `node-exporter` | Host (scraped at `172.17.0.1:9100`) | `node_cpu_seconds_total`, `node_memory_MemAvailable_bytes` |
| `cadvisor` | Containers | `container_cpu_usage_seconds_total`, `container_memory_usage_bytes` |
| `redis-exporter` | Redis | `redis_connected_clients`, `redis_memory_used_bytes` |
| `otel-collector` | Collector internals | `otelcol_receiver_accepted_spans` |

## Logs

The API logs JSON through structlog, one event per line:

```json
{"event": "hybrid_search", "level": "info", "logger": "src.rag.engine",
 "timestamp": "2026-10-11T09:12:03.123Z", "path": "/api/v1/ask", "method": "POST",
 "client_ip": "172.18.0.1", "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
 "query": "¿Cuál fue el EBITDA de IBERDROLA en 2024?", "filters": {"company": "IBERDROLA", "fiscal_year": "2024"}, "results": 24}
```

Promtail labels each stream with `container` and `service` (the Compose service name), extracts `level` as a label, and drops `/health` access lines.

Example LogQL queries in Grafana → Explore → Loki:

```logql
{service="api"} | json | level="error"
{service="api"} | json | event="coverage_gate_abstention"
{service="api"} | json | event="ask_route" | line_format "{{.route}}  {{.question}}"
{service="api"} | json | trace_id="4bf92f3577b34da6a3ce929d0e0e4736"
```

Useful events: `ask_route`, `llm_route`, `hybrid_search`, `hybrid_search_relaxed_temporal_filters`, `coverage_gate_abstention`, `query_complete`, `cache_hit`, `semantic_cache_hit`, `rate_limit_exceeded`, `agent_complete`, `ragas_evaluation`, `ingestion_finished`.

## Traces

Service name: `ibex35-rag-api`. Each request produces a FastAPI server span with child spans for Redis and HTTP calls (Ollama, Qdrant via HTTPX), plus manual RAG spans:

| Span | Attributes |
|---|---|
| `rag.query` | `question`, `company_filter`, `abstention`, `abstention_reason`, `latency_seconds`, `num_sources` |
| `rag.retrieve` | `candidates` |
| `rag.rerank` | `top_k` |
| `rag.generate` | `answer_len` |

The engine runs in worker threads; the API propagates the OpenTelemetry context into them so these spans stay attached to the request trace.

## Alerts

Rules in `infra/prometheus/rules/rag_alerts.yml`:

| Alert | Condition | Severity |
|---|---|---|
| `RAGAPIDown` | API target down for 1 min | critical |
| `RAGAPIHighLatency` | RAG p95 latency > 20 s for 5 min | warning |
| `RAGCacheHitRateLow` | Cache hit rate < 10% for 10 min | warning |
| `IngestionFailed` | Any failed ingestion run in the last hour | critical |
| `VectorStoreEmpty` | Collection empty for 5 min | critical |
| `RAGEmptyRetrievalHigh` | > 20% of queries retrieve nothing for 5 min | warning |
| `HighMemoryUsage` | Host RAM > 90% for 5 min | critical |
| `HighCPUUsage` | Host CPU > 85% for 10 min | warning |
| `NodeExporterDown` | node-exporter down for 2 min | warning |
| `RedisDown` | redis-exporter down for 2 min | critical |

Alertmanager routes everything to a `null` receiver, so alerts are visible at <http://localhost:9093> and in Grafana but not delivered anywhere. To get notifications, add a Slack, email or webhook receiver in `infra/alertmanager/alertmanager.yml` and restart the `alertmanager` service.

## Useful queries

```promql
# RAG p95 latency
histogram_quantile(0.95, sum by (le) (rate(rag_query_latency_seconds_bucket[5m])))

# Slowest stage (p95)
histogram_quantile(0.95, sum by (le, stage) (rate(rag_eval_stage_latency_seconds_bucket[5m])))

# Cache hit rate
rate(rag_cache_hits_total[10m]) / (rate(rag_cache_hits_total[10m]) + rate(rag_cache_misses_total[10m]))

# Abstention rate
rate(rag_eval_abstention_ratio_sum[1h]) / rate(rag_eval_abstention_ratio_count[1h])

# Average faithfulness over the last hour
rate(rag_eval_faithfulness_sum[1h]) / rate(rag_eval_faithfulness_count[1h])

# Requests by status code
sum by (status) (rate(http_requests_total{job="ibex35-rag-api"}[5m]))
```
