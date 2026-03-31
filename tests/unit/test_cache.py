"""Unit tests for Redis cache layer."""

import json
import pytest

from src.api.cache import get_cached_response, set_cached_response, _make_cache_key


def test_cache_key_deterministic():
    """Same inputs should always produce the same key."""
    k1 = _make_cache_key("¿Cuáles son los ingresos de BBVA?", "BBVA")
    k2 = _make_cache_key("¿Cuáles son los ingresos de BBVA?", "BBVA")
    assert k1 == k2


def test_cache_key_different_inputs():
    """Different questions should produce different keys."""
    k1 = _make_cache_key("¿Ingresos de BBVA?", None)
    k2 = _make_cache_key("¿Ingresos de Santander?", None)
    assert k1 != k2


def test_cache_key_company_filter_matters():
    """Same question with different company filter → different key."""
    k1 = _make_cache_key("¿Cuál es el EBITDA?", "BBVA")
    k2 = _make_cache_key("¿Cuál es el EBITDA?", "SANTANDER")
    assert k1 != k2


@pytest.mark.asyncio
async def test_cache_miss_returns_none(fake_redis):
    result = await get_cached_response(fake_redis, "unknown question", None)
    assert result is None


@pytest.mark.asyncio
async def test_cache_set_and_get(fake_redis):
    payload = {"answer": "Revenue was €10B", "latency_seconds": 1.2}
    await set_cached_response(fake_redis, "question", payload, ttl=60, company_filter=None)
    cached = await get_cached_response(fake_redis, "question", None)
    assert cached == payload


@pytest.mark.asyncio
async def test_cache_ttl_respected(fake_redis):
    """Verify TTL is set (fakeredis supports TTL)."""
    import asyncio
    payload = {"answer": "test"}
    await set_cached_response(fake_redis, "q", payload, ttl=1, company_filter=None)
    cached = await get_cached_response(fake_redis, "q", None)
    assert cached is not None
    await asyncio.sleep(1.1)
    expired = await get_cached_response(fake_redis, "q", None)
    assert expired is None
