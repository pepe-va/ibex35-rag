"""FastAPI dependencies: process-wide singletons shared across requests.

Heavy modules (LangChain, the reranker, FastEmbed) are imported inside the
factories so that importing the API stays fast and tests can patch them.
"""

from functools import lru_cache
from typing import TYPE_CHECKING, Annotated

import redis.asyncio as aioredis
from fastapi import Depends

from src.core.config import Settings, get_settings

if TYPE_CHECKING:
    from src.agents.financial_agent import FinancialAgent
    from src.api.semantic_cache import SemanticCache
    from src.rag.engine import RAGEngine
    from src.rag.vector_store.store import VectorStoreManager


@lru_cache
def get_vector_store() -> "VectorStoreManager":
    from src.rag.vector_store.store import VectorStoreManager

    return VectorStoreManager(get_settings())


@lru_cache
def get_rag_engine() -> "RAGEngine":
    from src.rag.engine import RAGEngine

    return RAGEngine(settings=get_settings(), vector_store=get_vector_store())


@lru_cache
def get_semantic_cache() -> "SemanticCache":
    from src.api.semantic_cache import SemanticCache

    return SemanticCache(get_settings())


@lru_cache
def get_financial_agent() -> "FinancialAgent":
    from src.agents.financial_agent import FinancialAgent

    return FinancialAgent(settings=get_settings(), rag_engine=get_rag_engine())


_redis_pool: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.from_url(
            get_settings().redis_url,
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


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
