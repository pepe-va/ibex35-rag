# Documentation

Detailed documentation for IBEX35 RAG. For installation and a quickstart, see the [main README](../README.md).

## Contents

- [Where to start](#where-to-start)
- [Documents](#documents)
- [Topic index](#topic-index)

## Where to start

| If you want to... | Read |
|---|---|
| Run the project for the first time | [README → Quickstart](../README.md#quickstart) |
| Understand how the system fits together | [architecture.md](architecture.md) |
| Understand or tune answer quality | [rag-pipeline.md](rag-pipeline.md) |
| Call the API from a client | [api.md](api.md) |
| Change models, limits or paths | [configuration.md](configuration.md) |
| Monitor the system or debug a slow or wrong answer | [observability.md](observability.md) |
| See how the chunks are distributed in the vector space | [qdrant-visualize.md](qdrant-visualize.md) |
| Contribute code | [CONTRIBUTING.md](../CONTRIBUTING.md) |

## Documents

| Document | Contents |
|---|---|
| [architecture.md](architecture.md) | Components, Docker Compose services, request and ingestion lifecycle, application layout, design decisions, known limitations |
| [rag-pipeline.md](rag-pipeline.md) | Docling extraction, metadata, chunking, deduplication, intent extraction, hybrid search, reranking, coverage gate, generation, online evaluation, agent, tuning reference |
| [api.md](api.md) | Endpoints, request body, SSE events, ingestion, health checks, caching, rate limiting, errors |
| [configuration.md](configuration.md) | Every environment variable grouped by area, plus engine constants |
| [observability.md](observability.md) | Grafana dashboards, metrics catalogue, logs, traces, alert rules, example PromQL and LogQL queries |
| [qdrant-visualize.md](qdrant-visualize.md) | Qdrant dashboard queries that explain the point distribution: by company, sector, content type, period, topic, language, and metadata quality checks |

## Topic index

| Topic | Section |
|---|---|
| Abstention / coverage gate | [rag-pipeline.md → Coverage gate](rag-pipeline.md#5-coverage-gate) |
| Adding a company | [CONTRIBUTING.md → Adding a company](../CONTRIBUTING.md#adding-a-company) |
| Agent tools and memory | [rag-pipeline.md → Agent](rag-pipeline.md#agent) |
| Alerts | [observability.md → Alerts](observability.md#alerts) |
| Auto-routing (`/ask`) | [api.md → POST /api/v1/ask](api.md#post-apiv1ask) |
| Caching | [api.md → Caching](api.md#caching) |
| Chunking | [rag-pipeline.md → Phase 2](rag-pipeline.md#phase-2-chunking-and-indexing) |
| Docker services and ports | [architecture.md → Docker Compose services](architecture.md#docker-compose-services) |
| Environment variables | [configuration.md](configuration.md) |
| Ingestion | [rag-pipeline.md → Ingestion](rag-pipeline.md#ingestion) |
| Logs | [observability.md → Logs](observability.md#logs) |
| Metrics | [observability.md → Metrics](observability.md#metrics) |
| Rate limiting | [api.md → Rate limiting](api.md#rate-limiting) |
| Re-ingesting from scratch | [rag-pipeline.md → Deduplication and re-ingestion](rag-pipeline.md#deduplication-and-re-ingestion) |
| Reranking | [rag-pipeline.md → Reranking](rag-pipeline.md#4-reranking) |
| Streaming (SSE) | [api.md → POST /api/v1/ask/stream](api.md#post-apiv1askstream) |
| Traces | [observability.md → Traces](observability.md#traces) |
| Vector space visualization | [qdrant-visualize.md](qdrant-visualize.md) |
| Tuning retrieval | [rag-pipeline.md → Tuning reference](rag-pipeline.md#tuning-reference) |
