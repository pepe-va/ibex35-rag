"""Redis cache layer for RAG query results."""

import hashlib
import json

import redis.asyncio as aioredis

from src.logging_config import get_logger

logger = get_logger(__name__)

CACHE_PREFIX = "ibex35:query:"


def _make_cache_key(question: str, company_filter: str | None) -> str:
    raw = f"{question.lower().strip()}|{company_filter or ''}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{CACHE_PREFIX}{digest}"


async def get_cached_response(
    redis: aioredis.Redis,
    question: str,
    company_filter: str | None = None,
) -> dict | None:
    key = _make_cache_key(question, company_filter)
    try:
        value = await redis.get(key)
        if value:
            logger.info("cache_hit", key=key)
            return json.loads(value)
    except Exception as exc:
        logger.warning("cache_get_error", error=str(exc))
    return None


async def set_cached_response(
    redis: aioredis.Redis,
    question: str,
    response: dict,
    ttl: int,
    company_filter: str | None = None,
) -> None:
    key = _make_cache_key(question, company_filter)
    try:
        await redis.setex(key, ttl, json.dumps(response))
        logger.info("cache_set", key=key, ttl=ttl)
    except Exception as exc:
        logger.warning("cache_set_error", error=str(exc))
