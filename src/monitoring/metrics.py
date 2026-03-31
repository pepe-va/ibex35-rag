"""Prometheus metrics for the IBEX35 RAG API."""

from prometheus_client import Counter, Gauge, Histogram

# ── Query metrics ─────────────────────────────────────────────────────────────

QUERY_REQUESTS = Counter(
    "ibex35_rag_query_requests_total",
    "Total number of RAG query requests",
    labelnames=["company", "cached"],
)

QUERY_LATENCY = Histogram(
    "ibex35_rag_query_latency_seconds",
    "RAG query end-to-end latency",
    labelnames=["company"],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0),
)

# ── Agent metrics ─────────────────────────────────────────────────────────────

AGENT_REQUESTS = Counter(
    "ibex35_rag_agent_requests_total",
    "Total number of agent requests",
)

AGENT_LATENCY = Histogram(
    "ibex35_rag_agent_latency_seconds",
    "Agent end-to-end latency",
    buckets=(1.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0),
)

# ── Cache metrics ─────────────────────────────────────────────────────────────

CACHE_HITS = Counter(
    "ibex35_rag_cache_hits_total",
    "Total Redis cache hits",
)

CACHE_MISSES = Counter(
    "ibex35_rag_cache_misses_total",
    "Total Redis cache misses",
)

# ── Ingestion metrics ─────────────────────────────────────────────────────────

INGESTION_RUNS = Counter(
    "ibex35_rag_ingestion_runs_total",
    "Total ingestion pipeline runs",
    labelnames=["status"],
)

INGESTION_DURATION = Histogram(
    "ibex35_rag_ingestion_duration_seconds",
    "Ingestion pipeline duration",
    buckets=(10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
)

VECTOR_STORE_COUNT = Gauge(
    "ibex35_rag_vector_store_documents_total",
    "Number of documents in the vector store",
)
