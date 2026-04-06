"""Prometheus metrics for the IBEX35 RAG API."""

from prometheus_client import Counter, Gauge, Histogram

# ── Query metrics ─────────────────────────────────────────────────────────────

QUERY_REQUESTS = Counter(
    "rag_query_requests_total",
    "Total number of RAG query requests",
    labelnames=["company", "cached"],
)

QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "RAG query end-to-end latency",
    labelnames=["company"],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0),
)

# ── Agent metrics ─────────────────────────────────────────────────────────────

AGENT_REQUESTS = Counter(
    "rag_agent_requests_total",
    "Total number of agent requests",
)

AGENT_LATENCY = Histogram(
    "rag_agent_latency_seconds",
    "Agent end-to-end latency",
    buckets=(1.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0),
)

# ── Cache metrics ─────────────────────────────────────────────────────────────

CACHE_HITS = Counter(
    "rag_cache_hits_total",
    "Total Redis cache hits",
)

CACHE_MISSES = Counter(
    "rag_cache_misses_total",
    "Total Redis cache misses",
)

CACHE_KEYS_CURRENT = Gauge(
    "rag_cache_keys_current",
    "Current number of query results cached in Redis",
)

CACHE_BYTES_USED = Gauge(
    "rag_cache_bytes_used",
    "Redis memory used in bytes",
)

CACHE_BYTES_LIMIT = Gauge(
    "rag_cache_bytes_limit",
    "Redis maxmemory limit in bytes (0 = unlimited)",
)

CACHE_EVICTIONS = Gauge(
    "rag_cache_evictions_current",
    "Total keys evicted by Redis since start (mirrored from Redis INFO)",
)

# ── Ingestion metrics ─────────────────────────────────────────────────────────

INGESTION_RUNS = Counter(
    "rag_ingestion_runs_total",
    "Total ingestion pipeline runs",
    labelnames=["status"],
)

INGESTION_DURATION = Histogram(
    "rag_ingestion_duration_seconds",
    "Ingestion pipeline duration",
    buckets=(10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
)

VECTOR_STORE_COUNT = Gauge(
    "rag_vector_store_documents_current",
    "Number of documents in the vector store",
)

# ── Chunking metrics ──────────────────────────────────────────────────────────

CHUNKING_DURATION = Histogram(
    "rag_chunking_duration_seconds",
    "Time to chunk documents into nodes",
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)

CHUNKS_CREATED = Gauge(
    "rag_chunks_created_current",
    "Total chunks (nodes) created in last ingestion run",
)

CHUNK_SIZE_BYTES = Histogram(
    "rag_chunk_size_bytes",
    "Size of each chunk in bytes",
    buckets=(100, 250, 500, 1000, 2000, 4000, 8000),
)

# ── Retrieval metrics ─────────────────────────────────────────────────────────

RETRIEVED_CHUNKS = Histogram(
    "rag_retrieved_chunks",
    "Number of chunks retrieved per query",
    buckets=(1, 2, 3, 5, 8, 10, 15, 20),
)

RETRIEVAL_SCORE = Histogram(
    "rag_retrieval_score",
    "Similarity score of each retrieved chunk",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

EMPTY_RETRIEVAL = Counter(
    "rag_empty_retrieval_total",
    "Queries that returned zero chunks from the vector store",
)

CONTEXT_BYTES = Histogram(
    "rag_context_bytes",
    "Total context bytes sent to LLM per query",
    buckets=(500, 1000, 2000, 4000, 8000, 16000, 32000),
)

TOP_K = Gauge(
    "rag_top_k_current",
    "Current similarity_top_k setting",
)

# ── LLM / Ollama metrics (parsed from API response) ──────────────────────────

LLM_PROMPT_TOKENS = Histogram(
    "rag_llm_prompt_tokens",
    "Prompt tokens (prompt_eval_count) per Ollama request",
    buckets=(64, 128, 256, 512, 1024, 2048, 4096, 8192),
)

LLM_COMPLETION_TOKENS = Histogram(
    "rag_llm_completion_tokens",
    "Completion tokens (eval_count) per Ollama request",
    buckets=(32, 64, 128, 256, 512, 1024, 2048),
)

LLM_TOTAL_DURATION = Histogram(
    "rag_llm_total_duration_seconds",
    "Total Ollama request duration (total_duration field), in seconds",
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0),
)

LLM_LOAD_DURATION = Histogram(
    "rag_llm_load_duration_seconds",
    "Ollama model load duration (load_duration field), in seconds",
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
)

# ── Index content metrics ─────────────────────────────────────────────────────

COMPANIES_INDEXED = Gauge(
    "rag_companies_indexed_current",
    "Number of distinct companies indexed in the vector store",
)

PAGES_INDEXED = Gauge(
    "rag_pages_indexed_current",
    "Number of PDF pages indexed in the vector store",
)

INGESTION_LAST_DURATION = Gauge(
    "rag_ingestion_last_duration_seconds",
    "Duration in seconds of the last ingestion run",
)

# ── RAG eval metrics (4.2 Generation Quality) ────────────────────────────────

EVAL_FAITHFULNESS = Histogram(
    "rag_eval_faithfulness",
    "RAGAS Faithfulness: % of answer claims supported by retrieved context (0–1)",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

EVAL_ANSWER_RELEVANCY = Histogram(
    "rag_eval_answer_relevancy",
    "RAGAS Answer Relevancy: semantic pertinence of the answer to the question (0–1)",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

EVAL_ABSTENTION = Counter(
    "rag_eval_abstention_total",
    "Queries where the LLM abstained (answered 'I don't know' / no data available)",
)

EVAL_CHUNK_UTILIZATION = Histogram(
    "rag_eval_chunk_utilization_ratio",
    "Fraction of retrieved chunks with positive reranker score (proxy for utilization)",
    buckets=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
)

EVAL_TOKEN_IOU = Histogram(
    "rag_eval_token_iou",
    "Token-wise overlap between query tokens and retrieved context tokens (0–1)",
    buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0),
)

# ── RAG eval metrics (4.4 Stage latency) ─────────────────────────────────────

EVAL_STAGE_LATENCY = Histogram(
    "rag_eval_stage_latency_seconds",
    "Latency per RAG sub-stage",
    labelnames=["stage"],
    buckets=(0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
)
