"""Unit tests for sliding window rate limiter."""

import pytest
from fastapi import HTTPException
from unittest.mock import MagicMock

from src.api.rate_limiter import check_rate_limit
from src.config import Settings


def _make_request(ip: str = "127.0.0.1") -> MagicMock:
    request = MagicMock()
    request.client.host = ip
    return request


@pytest.fixture
def strict_settings(settings) -> Settings:
    settings.rate_limit_requests = 3
    settings.rate_limit_window_seconds = 60
    return settings


@pytest.mark.asyncio
async def test_rate_limit_allows_under_threshold(fake_redis, strict_settings):
    request = _make_request()
    for _ in range(3):
        await check_rate_limit(request, fake_redis, strict_settings)  # Should not raise


@pytest.mark.asyncio
async def test_rate_limit_blocks_over_threshold(fake_redis, strict_settings):
    request = _make_request()
    for _ in range(3):
        await check_rate_limit(request, fake_redis, strict_settings)

    with pytest.raises(HTTPException) as exc_info:
        await check_rate_limit(request, fake_redis, strict_settings)

    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_rate_limit_different_ips_independent(fake_redis, strict_settings):
    """Different IPs should have independent rate limit counters."""
    req1 = _make_request("192.168.1.1")
    req2 = _make_request("192.168.1.2")

    for _ in range(3):
        await check_rate_limit(req1, fake_redis, strict_settings)

    # req2 should still be allowed
    await check_rate_limit(req2, fake_redis, strict_settings)
