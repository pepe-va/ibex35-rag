"""FastAPI dependency injection: shared instances across requests."""

from functools import lru_cache

import redis.asyncio as aioredis

from src.config import Settings, get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────


@lru_cache
def get_vector_store():
    from src.vectorstore.store import VectorStoreManager
    settings = get_settings()
    return VectorStoreManager(settings)


@lru_cache
def get_rag_engine():
    from src.rag.engine import RAGEngine
    settings = get_settings()
    vs = get_vector_store()
    return RAGEngine(settings=settings, index=vs.get_index())


@lru_cache
def get_financial_agent():
    from src.agents.financial_agent import FinancialAgent
    settings = get_settings()
    rag = get_rag_engine()
    return FinancialAgent(settings=settings, rag_engine=rag)


_redis_pool: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis_pool
    if _redis_pool is None:
        settings = get_settings()
        _redis_pool = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis_pool


async def close_redis() -> None:
    global _redis_pool
    if _redis_pool:
        await _redis_pool.aclose()
        _redis_pool = None


def get_settings_dep() -> Settings:
    return get_settings()
