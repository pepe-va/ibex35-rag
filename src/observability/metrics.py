"""Prometheus metrics for the IBEX35 RAG API.

Reference values per section:

LLM (qwen2.5:7b served locally by Ollama):
  - prompt_tokens p50: 800–2500 (RAG context + financial question)
  - completion_tokens p50: 100–600 (structured answer)
  - latency p50: 4–10 s (CPU), 1–4 s (GPU)
  - tokens/s p50: 5–15 (CPU), 25–60 (GPU)
  - error rate: < 1 % normal, alert above 5 %

RAG evaluation (RAGAS, 0–1 scale, production targets):
  - faithfulness: > 0.7 good, < 0.4 suggests hallucination
  - answer_relevancy: > 0.7 good
  - contextual_precision: > 0.6 good
  - contextual_recall: > 0.6 good
  - abstention_ratio: < 0.15 normal (if it rises, data is missing)

Cache (Redis):
  - target hit rate: > 40 % in steady state
  - evictions: ideally 0; growth means memory pressure
"""

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

# ── LLM metrics (collected from LangSmith) ────────────────────────────────────
# Filled by the background collector (src/observability/langsmith_collector.py),
# which polls the LangSmith API and records the recent LLM runs.

LLM_RUNS = Counter(
    "llm_runs_total",
    "LLM calls recorded in LangSmith. "
    "Reference: grows with rag_query_requests_total + agent requests.",
)

LLM_ERRORS = Counter(
    "llm_errors_total",
    "LLM calls that ended in an error (timeout, OOM, rejected by Ollama). "
    "Reference: < 1 % of runs. Alert above 5 %.",
)

LLM_PROMPT_TOKENS = Histogram(
    "llm_prompt_tokens",
    "Prompt tokens sent to the LLM per request, including RAG context and question. "
    "Reference qwen2.5:7b: p50 800–2500, p95 < 6000. "
    "Alert if p95 > 8000 (truncation risk with a 32k window).",
    buckets=(128, 256, 512, 1024, 2048, 4096, 6144, 8192, 12288),
)

LLM_COMPLETION_TOKENS = Histogram(
    "llm_completion_tokens",
    "Tokens generated in the LLM answer per request. "
    "Reference qwen2.5:7b: p50 100–600, p95 < 1500. "
    "A p50 below 20 suggests empty answers or mass abstention.",
    buckets=(16, 32, 64, 128, 256, 512, 1024, 2048),
)

LLM_TOTAL_TOKENS = Histogram(
    "llm_total_tokens",
    "Prompt + completion tokens per request. "
    "Reference: p50 < 3500, p95 < 8000. "
    "Useful to estimate cost if moving to a paid LLM.",
    buckets=(256, 512, 1024, 2048, 4096, 8192, 16384),
)

LLM_REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds",
    "LLM call latency from request to complete answer, in seconds. "
    "Reference qwen2.5:7b CPU: p50 4–10 s, p95 < 25 s. "
    "Reference qwen2.5:7b GPU: p50 1–4 s, p95 < 10 s. "
    "Alert if p95 > 30 s (users wait too long).",
    buckets=(0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 25.0, 40.0, 60.0),
)

LLM_TOKENS_PER_SECOND = Histogram(
    "llm_tokens_per_second",
    "Generation throughput: completion_tokens / LLM latency. "
    "Reference qwen2.5:7b CPU: p50 5–15 t/s. GPU: p50 25–60 t/s. "
    "A sustained drop means CPU/GPU saturation or memory contention.",
    buckets=(1.0, 3.0, 5.0, 10.0, 20.0, 35.0, 50.0, 80.0),
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

# ── RAG evaluation metrics ────────────────────────────────────────────────────

# Generation and retrieval quality

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

EVAL_CHUNK_UTILIZATION = Histogram(
    "rag_eval_chunk_utilization_ratio",
    "Fraction of retrieved chunks with positive reranker score (proxy for utilization)",
    buckets=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
)

EVAL_CONTEXTUAL_PRECISION = Histogram(
    "rag_eval_contextual_precision",
    "Contextual precision proxy for retrieval ranking quality (0–1)",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

EVAL_CONTEXTUAL_RECALL = Histogram(
    "rag_eval_contextual_recall",
    "Contextual recall proxy: query coverage by retrieved context (0–1)",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

EVAL_CHUNK_ATTRIBUTION = Histogram(
    "rag_eval_chunk_attribution_ratio",
    "Proxy ratio of retrieved chunks that contribute to final answer (0–1)",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# Lexical / semantic comparison (proxies, no ground truth)

EVAL_ANSWER_CORRECTNESS = Histogram(
    "rag_eval_answer_correctness",
    "Answer correctness proxy combining lexical/semantic agreement (0–1)",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

EVAL_ANSWER_SIMILARITY = Histogram(
    "rag_eval_answer_similarity",
    "Answer similarity proxy against reference text (0–1)",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# Production engineering

EVAL_ABSTENTION_RATIO = Histogram(
    "rag_eval_abstention_ratio",
    "Abstention ratio as binary observation per query (0 or 1)",
    buckets=(0.0, 0.5, 1.0),
)

EVAL_CONTEXT_CLIFF = Histogram(
    "rag_eval_context_cliff_score",
    "Context cliff proxy: quality degradation when context grows (0–1)",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

EVAL_COST_PER_INFERENCE = Histogram(
    "rag_eval_cost_per_inference",
    "Estimated cost per inference in configured currency / request",
    buckets=(0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
)

# ── RAG stage latency ─────────────────────────────────────────────────────────

EVAL_STAGE_LATENCY = Histogram(
    "rag_eval_stage_latency_seconds",
    "Latency per RAG sub-stage",
    labelnames=["stage"],
    buckets=(0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
)
