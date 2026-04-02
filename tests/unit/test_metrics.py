"""Unit tests for Prometheus metrics definition.

Verifies:
- All custom metrics use the rag_ prefix (not the legacy ibex35_rag_ prefix).
- New metrics (LLM, retrieval quality) are registered.
- Metric types and label names are correct.
"""

import pytest
from prometheus_client import REGISTRY, Counter, Gauge, Histogram

import src.monitoring.metrics as m


# ── Helpers ───────────────────────────────────────────────────────────────────


def _registered_names() -> set[str]:
    """Return all metric family names currently in the registry."""
    return {metric.name for metric in REGISTRY.collect()}


def _rag_names() -> set[str]:
    """Return only the rag_* metric families."""
    return {n for n in _registered_names() if n.startswith("rag_")}


# ── Prefix ────────────────────────────────────────────────────────────────────


def test_no_legacy_ibex35_prefix():
    """No metric should use the old ibex35_rag_ prefix."""
    legacy = {n for n in _registered_names() if n.startswith("ibex35_rag_")}
    assert legacy == set(), f"Legacy metrics still present: {legacy}"


def test_all_custom_metrics_use_rag_prefix():
    """Every custom metric object in the module must have a rag_ name."""
    for attr, obj in vars(m).items():
        if isinstance(obj, (Counter, Gauge, Histogram)):
            assert obj._name.startswith("rag_"), (
                f"Metric '{attr}' has name '{obj._name}', expected rag_ prefix"
            )


# ── Query metrics ─────────────────────────────────────────────────────────────


def test_query_requests_counter_exists():
    # prometheus_client stores Counter._name without the _total suffix;
    # the suffix is appended during serialisation.
    assert isinstance(m.QUERY_REQUESTS, Counter)
    assert m.QUERY_REQUESTS._name == "rag_query_requests"


def test_query_latency_histogram_exists():
    assert isinstance(m.QUERY_LATENCY, Histogram)
    assert m.QUERY_LATENCY._name == "rag_query_latency_seconds"
    assert "company" in m.QUERY_LATENCY._labelnames


def test_agent_metrics_exist():
    assert isinstance(m.AGENT_REQUESTS, Counter)
    assert isinstance(m.AGENT_LATENCY, Histogram)
    assert m.AGENT_REQUESTS._name == "rag_agent_requests"
    assert m.AGENT_LATENCY._name == "rag_agent_latency_seconds"


# ── Cache metrics ─────────────────────────────────────────────────────────────


def test_cache_counters_exist():
    assert isinstance(m.CACHE_HITS, Counter)
    assert isinstance(m.CACHE_MISSES, Counter)
    assert m.CACHE_HITS._name == "rag_cache_hits"
    assert m.CACHE_MISSES._name == "rag_cache_misses"


def test_cache_gauges_exist():
    assert isinstance(m.CACHE_KEYS_CURRENT, Gauge)
    assert isinstance(m.CACHE_BYTES_USED, Gauge)
    assert isinstance(m.CACHE_BYTES_LIMIT, Gauge)


# ── Retrieval quality metrics (new) ──────────────────────────────────────────


def test_retrieval_score_histogram_exists():
    assert isinstance(m.RETRIEVAL_SCORE, Histogram)
    assert m.RETRIEVAL_SCORE._name == "rag_retrieval_score"


def test_retrieval_score_buckets_cover_0_to_1():
    """Buckets should cover the full similarity score range [0, 1]."""
    upper_bounds = [b for b in m.RETRIEVAL_SCORE._upper_bounds if b != float("inf")]
    assert min(upper_bounds) <= 0.1
    assert max(upper_bounds) >= 0.9


def test_empty_retrieval_counter_exists():
    assert isinstance(m.EMPTY_RETRIEVAL, Counter)
    assert m.EMPTY_RETRIEVAL._name == "rag_empty_retrieval"


# ── LLM / Ollama metrics (new) ────────────────────────────────────────────────


def test_llm_prompt_tokens_histogram_exists():
    assert isinstance(m.LLM_PROMPT_TOKENS, Histogram)
    assert m.LLM_PROMPT_TOKENS._name == "rag_llm_prompt_tokens"


def test_llm_completion_tokens_histogram_exists():
    assert isinstance(m.LLM_COMPLETION_TOKENS, Histogram)
    assert m.LLM_COMPLETION_TOKENS._name == "rag_llm_completion_tokens"


def test_llm_total_duration_histogram_exists():
    assert isinstance(m.LLM_TOTAL_DURATION, Histogram)
    assert m.LLM_TOTAL_DURATION._name == "rag_llm_total_duration_seconds"


def test_llm_load_duration_histogram_exists():
    assert isinstance(m.LLM_LOAD_DURATION, Histogram)
    assert m.LLM_LOAD_DURATION._name == "rag_llm_load_duration_seconds"


def test_llm_token_buckets_cover_reasonable_range():
    """Prompt token buckets should cover at least up to 4096 tokens."""
    upper_bounds = [b for b in m.LLM_PROMPT_TOKENS._upper_bounds if b != float("inf")]
    assert max(upper_bounds) >= 4096


# ── Ingestion metrics ─────────────────────────────────────────────────────────


def test_ingestion_metrics_exist():
    assert isinstance(m.INGESTION_RUNS, Counter)
    assert isinstance(m.INGESTION_DURATION, Histogram)
    assert isinstance(m.INGESTION_LAST_DURATION, Gauge)
    assert isinstance(m.VECTOR_STORE_COUNT, Gauge)
    assert m.INGESTION_RUNS._name == "rag_ingestion_runs"
    assert m.VECTOR_STORE_COUNT._name == "rag_vector_store_documents_current"


def test_chunking_metrics_exist():
    assert isinstance(m.CHUNKING_DURATION, Histogram)
    assert isinstance(m.CHUNKS_CREATED, Gauge)
    assert isinstance(m.CHUNK_SIZE_BYTES, Histogram)
