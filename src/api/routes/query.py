"""Query endpoints: RAG and Agent."""

import asyncio
import time

from fastapi import APIRouter, Depends, Request

from src.api.cache import get_cached_response, set_cached_response
from src.api.dependencies import (
    get_financial_agent,
    get_rag_engine,
    get_redis,
    get_settings_dep,
)
from src.api.models import AgentResponse, QueryRequest, QueryResponse, SourceDoc
from src.api.rate_limiter import check_rate_limit
from src.config import Settings
from src.logging_config import get_logger
from src.monitoring.metrics import (
    AGENT_LATENCY,
    AGENT_REQUESTS,
    CACHE_HITS,
    CACHE_MISSES,
    QUERY_LATENCY,
    QUERY_REQUESTS,
)

router = APIRouter(tags=["query"])
logger = get_logger(__name__)


@router.post("/query", response_model=QueryResponse)
async def query_rag(
    request: Request,
    body: QueryRequest,
    settings: Settings = Depends(get_settings_dep),
) -> QueryResponse:
    """
    Query the RAG system with a financial question.

    - Checks Redis cache first (SHA-256 keyed by question + filter).
    - Falls back to LlamaIndex retrieval + Ollama generation.
    - Rate limited per IP (sliding window via Redis).
    """
    redis = await get_redis()
    await check_rate_limit(request, redis, settings)

    # Try cache
    cached = await get_cached_response(redis, body.question, body.company_filter)
    if cached:
        CACHE_HITS.inc()
        QUERY_REQUESTS.labels(company=body.company_filter or "all", cached="true").inc()
        return QueryResponse(**{**cached, "from_cache": True})

    CACHE_MISSES.inc()

    # RAG query — runs sync LlamaIndex in a thread to avoid blocking the event loop
    rag_engine = get_rag_engine()
    start = time.perf_counter()
    result = await asyncio.to_thread(
        rag_engine.query,
        body.question,
        body.company_filter,
    )
    QUERY_LATENCY.labels(company=body.company_filter or "all").observe(
        time.perf_counter() - start
    )
    QUERY_REQUESTS.labels(company=body.company_filter or "all", cached="false").inc()

    response = QueryResponse(
        answer=result.answer,
        sources=[SourceDoc(**s) for s in result.sources],
        latency_seconds=result.latency_seconds,
        from_cache=False,
        query=result.query,
    )

    # Store in cache
    await set_cached_response(
        redis,
        body.question,
        response.model_dump(),
        settings.cache_ttl_seconds,
        body.company_filter,
    )

    return response


@router.post("/agent", response_model=AgentResponse)
async def query_agent(
    request: Request,
    body: QueryRequest,
    settings: Settings = Depends(get_settings_dep),
) -> AgentResponse:
    """
    Query the ReAct financial agent.

    Combines RAG knowledge base with real-time market tools (yfinance).
    Best for complex, multi-step questions that require both fundamentals and market data.
    """
    redis = await get_redis()
    await check_rate_limit(request, redis, settings)

    agent = get_financial_agent()

    AGENT_REQUESTS.inc()
    start = time.perf_counter()
    result = await asyncio.to_thread(agent.run, body.question)
    AGENT_LATENCY.observe(time.perf_counter() - start)

    return AgentResponse(
        answer=result.answer,
        latency_seconds=result.latency_seconds,
        steps_taken=result.steps_taken,
        tools_used=result.tools_used,
        query=result.query,
    )
