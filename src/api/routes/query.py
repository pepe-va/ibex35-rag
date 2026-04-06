"""Query endpoints: RAG and Agent."""

import asyncio
import time

from fastapi import APIRouter, Depends, Request
from opentelemetry import context as otel_context

from src.api.cache import get_cached_response, set_cached_response
from src.api.dependencies import (
    get_financial_agent,
    get_rag_engine,
    get_redis,
    get_settings_dep,
)
from src.api.models import AgentResponse, AskResponse, QueryRequest, QueryResponse, SourceDoc
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

_AGENT_KEYWORDS = {
    # Spanish
    "precio", "cotización", "cotiza", "cotizacion", "bolsa", "mercado",
    "rentabilidad bursátil", "subida", "bajada", "variación", "variacion",
    "máximo", "minimo", "mínimo", "maximo", "52 semanas", "histórico",
    "historico", "acción", "accion", "acciones",
    # English
    "price", "stock price", "market", "trading", "high", "low",
    "52 week", "52-week", "return", "performance", "chart",
}


def _should_use_agent(question: str) -> bool:
    """Route to agent if the question is about market prices; otherwise use RAG."""
    q = question.lower()
    return any(kw in q for kw in _AGENT_KEYWORDS)


@router.post("/query", response_model=QueryResponse)
async def query_rag(
    request: Request,
    body: QueryRequest,
    settings: Settings = Depends(get_settings_dep),
) -> QueryResponse:
    """
    Query the RAG system with a financial question.

    - Checks Redis cache first (SHA-256 keyed by question + filter).
    - Falls back to hybrid retrieval + Ollama generation.
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

    # RAG query — runs sync engine in a thread to avoid blocking the event loop
    rag_engine = get_rag_engine()
    start = time.perf_counter()
    _ctx = otel_context.get_current()

    def _run_query():
        token = otel_context.attach(_ctx)
        try:
            return rag_engine.query(body.question, body.company_filter)
        finally:
            otel_context.detach(token)

    result = await asyncio.to_thread(_run_query)
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
    _ctx = otel_context.get_current()

    def _run_agent():
        token = otel_context.attach(_ctx)
        try:
            return agent.run(body.question, body.thread_id)
        finally:
            otel_context.detach(token)

    result = await asyncio.to_thread(_run_agent)
    AGENT_LATENCY.observe(time.perf_counter() - start)

    return AgentResponse(
        answer=result.answer,
        latency_seconds=result.latency_seconds,
        steps_taken=result.steps_taken,
        tools_used=result.tools_used,
        query=result.query,
    )


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: Request,
    body: QueryRequest,
    settings: Settings = Depends(get_settings_dep),
) -> AskResponse:
    """
    Auto-routing endpoint: uses RAG for fundamentals, Agent for market/price questions.

    The route is determined by keywords in the question. The caller can override
    by setting `use_agent=true` in the request body.
    """
    redis = await get_redis()
    await check_rate_limit(request, redis, settings)

    use_agent = body.use_agent or _should_use_agent(body.question)
    route = "agent" if use_agent else "rag"
    logger.info("ask_route", question=body.question[:80], route=route)

    if use_agent:
        agent = get_financial_agent()
        AGENT_REQUESTS.inc()
        start = time.perf_counter()
        _ctx = otel_context.get_current()

        def _run_agent():
            token = otel_context.attach(_ctx)
            try:
                return agent.run(body.question, body.thread_id)
            finally:
                otel_context.detach(token)

        result = await asyncio.to_thread(_run_agent)
        AGENT_LATENCY.observe(time.perf_counter() - start)

        return AskResponse(
            answer=result.answer,
            route=route,
            latency_seconds=result.latency_seconds,
            tools_used=result.tools_used,
            steps_taken=result.steps_taken,
            query=result.query,
        )

    # RAG path — check cache first
    cached = await get_cached_response(redis, body.question, body.company_filter)
    if cached:
        CACHE_HITS.inc()
        QUERY_REQUESTS.labels(company=body.company_filter or "all", cached="true").inc()
        return AskResponse(**{**cached, "route": route, "from_cache": True})

    CACHE_MISSES.inc()
    rag_engine = get_rag_engine()
    start = time.perf_counter()
    _ctx = otel_context.get_current()

    def _run_query():
        token = otel_context.attach(_ctx)
        try:
            return rag_engine.query(body.question, body.company_filter)
        finally:
            otel_context.detach(token)

    result = await asyncio.to_thread(_run_query)
    QUERY_LATENCY.labels(company=body.company_filter or "all").observe(
        time.perf_counter() - start
    )
    QUERY_REQUESTS.labels(company=body.company_filter or "all", cached="false").inc()

    response = AskResponse(
        answer=result.answer,
        route=route,
        sources=[SourceDoc(**s) for s in result.sources],
        latency_seconds=result.latency_seconds,
        from_cache=False,
        query=result.query,
    )

    await set_cached_response(
        redis,
        body.question,
        response.model_dump(),
        settings.cache_ttl_seconds,
        body.company_filter,
    )

    return response
