"""Unit tests for _OllamaMetricsHandler.

Tests that the LlamaIndex callback handler correctly parses the Ollama
response dict and feeds the Prometheus histograms — without needing a
live Ollama instance.
"""

import pytest
from unittest.mock import MagicMock

from prometheus_client import REGISTRY

from src.rag.engine import _OllamaMetricsHandler


# ── Helpers ───────────────────────────────────────────────────────────────────


def _count(metric_name: str) -> float:
    """Return the _count sample of a histogram (total observations)."""
    return REGISTRY.get_sample_value(f"{metric_name}_count") or 0.0


def _sum(metric_name: str) -> float:
    """Return the _sum sample of a histogram (total sum of observations)."""
    return REGISTRY.get_sample_value(f"{metric_name}_sum") or 0.0


def _make_response(raw: dict) -> MagicMock:
    """Build a mock LlamaIndex response with a .raw dict."""
    response = MagicMock()
    response.raw = raw
    return response


# Import the CBEventType after the handler import (llama_index is a dep)
from llama_index.core.callbacks import CBEventType


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def handler() -> _OllamaMetricsHandler:
    return _OllamaMetricsHandler()


@pytest.fixture
def full_ollama_raw() -> dict:
    """Realistic Ollama API response fields."""
    return {
        "prompt_eval_count": 256,
        "eval_count": 128,
        "total_duration": 5_000_000_000,  # 5 seconds in ns
        "load_duration": 200_000_000,     # 0.2 seconds in ns
    }


# ── Token parsing ─────────────────────────────────────────────────────────────


def test_prompt_tokens_observed(handler, full_ollama_raw):
    before = _count("rag_llm_prompt_tokens")
    before_sum = _sum("rag_llm_prompt_tokens")

    handler.on_event_end(CBEventType.LLM, payload={"response": _make_response(full_ollama_raw)})

    assert _count("rag_llm_prompt_tokens") == before + 1
    assert _sum("rag_llm_prompt_tokens") == before_sum + 256


def test_completion_tokens_observed(handler, full_ollama_raw):
    before = _count("rag_llm_completion_tokens")
    before_sum = _sum("rag_llm_completion_tokens")

    handler.on_event_end(CBEventType.LLM, payload={"response": _make_response(full_ollama_raw)})

    assert _count("rag_llm_completion_tokens") == before + 1
    assert _sum("rag_llm_completion_tokens") == before_sum + 128


# ── Duration parsing (ns → s) ─────────────────────────────────────────────────


def test_total_duration_converted_from_nanoseconds(handler, full_ollama_raw):
    before = _count("rag_llm_total_duration_seconds")
    before_sum = _sum("rag_llm_total_duration_seconds")

    handler.on_event_end(CBEventType.LLM, payload={"response": _make_response(full_ollama_raw)})

    assert _count("rag_llm_total_duration_seconds") == before + 1
    # 5_000_000_000 ns = 5.0 s
    assert abs(_sum("rag_llm_total_duration_seconds") - (before_sum + 5.0)) < 1e-6


def test_load_duration_converted_from_nanoseconds(handler, full_ollama_raw):
    before = _count("rag_llm_load_duration_seconds")
    before_sum = _sum("rag_llm_load_duration_seconds")

    handler.on_event_end(CBEventType.LLM, payload={"response": _make_response(full_ollama_raw)})

    assert _count("rag_llm_load_duration_seconds") == before + 1
    # 200_000_000 ns = 0.2 s
    assert abs(_sum("rag_llm_load_duration_seconds") - (before_sum + 0.2)) < 1e-6


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_ignores_non_llm_events(handler):
    before_prompt = _count("rag_llm_prompt_tokens")
    before_completion = _count("rag_llm_completion_tokens")

    # Fire a different event type — should be ignored
    handler.on_event_end(CBEventType.QUERY, payload={"response": _make_response({"eval_count": 99})})
    handler.on_event_end(CBEventType.RETRIEVE, payload=None)

    assert _count("rag_llm_prompt_tokens") == before_prompt
    assert _count("rag_llm_completion_tokens") == before_completion


def test_handles_none_payload(handler):
    before = _count("rag_llm_prompt_tokens")
    handler.on_event_end(CBEventType.LLM, payload=None)
    assert _count("rag_llm_prompt_tokens") == before


def test_handles_response_without_raw(handler):
    """Response with no .raw attribute should not crash."""
    before = _count("rag_llm_prompt_tokens")
    response = MagicMock(spec=[])  # no attributes
    handler.on_event_end(CBEventType.LLM, payload={"response": response})
    assert _count("rag_llm_prompt_tokens") == before


def test_handles_empty_raw(handler):
    """Empty raw dict (e.g., streaming mode) should not observe anything."""
    before_prompt = _count("rag_llm_prompt_tokens")
    before_completion = _count("rag_llm_completion_tokens")

    handler.on_event_end(CBEventType.LLM, payload={"response": _make_response({})})

    assert _count("rag_llm_prompt_tokens") == before_prompt
    assert _count("rag_llm_completion_tokens") == before_completion


def test_handles_zero_eval_count(handler):
    """Zero values should not be observed (falsy guard in handler)."""
    before = _count("rag_llm_completion_tokens")
    raw = {"prompt_eval_count": 100, "eval_count": 0}  # eval_count is 0 → falsy
    handler.on_event_end(CBEventType.LLM, payload={"response": _make_response(raw)})

    # prompt tokens observed, completion tokens NOT (0 is falsy)
    assert _count("rag_llm_completion_tokens") == before


def test_partial_raw_only_observes_present_fields(handler):
    """If only some fields are present, only those metrics should be observed."""
    before_prompt = _count("rag_llm_prompt_tokens")
    before_total = _count("rag_llm_total_duration_seconds")

    # Only prompt_eval_count present
    raw = {"prompt_eval_count": 512}
    handler.on_event_end(CBEventType.LLM, payload={"response": _make_response(raw)})

    assert _count("rag_llm_prompt_tokens") == before_prompt + 1
    assert _count("rag_llm_total_duration_seconds") == before_total  # unchanged


# ── start_trace / end_trace do not crash ─────────────────────────────────────


def test_start_trace_is_no_op(handler):
    handler.start_trace(trace_id="abc")
    handler.start_trace()  # no args


def test_end_trace_is_no_op(handler):
    handler.end_trace(trace_id="abc", trace_map={})
    handler.end_trace()
