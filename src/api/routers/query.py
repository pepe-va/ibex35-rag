"""Query endpoints: RAG, agent, auto-routing and streaming."""

import asyncio
import json
import re
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar

import redis.asyncio as aioredis
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from opentelemetry import context as otel_context

from src.api import dependencies
from src.api.cache import get_cached_response, set_cached_response
from src.api.dependencies import SettingsDep
from src.api.rate_limiter import check_rate_limit
from src.api.schemas import AgentResponse, AskResponse, QueryRequest, QueryResponse, SourceDoc
from src.core.config import Settings
from src.core.logging import get_logger
from src.observability.metrics import (
    AGENT_LATENCY,
    AGENT_REQUESTS,
    CACHE_HITS,
    CACHE_MISSES,
    QUERY_LATENCY,
    QUERY_REQUESTS,
)

if TYPE_CHECKING:
    from langchain_ollama import ChatOllama

    from src.agents.financial_agent import AgentResult

router = APIRouter(tags=["query"])
logger = get_logger(__name__)

Route = Literal["rag", "agent"]
P = ParamSpec("P")
T = TypeVar("T")

# ── Routing ───────────────────────────────────────────────────────────────────

_ROUTE_SYSTEM = """\
Eres un clasificador de intención. Responde ÚNICAMENTE con una palabra.
- Responde "agente" si la pregunta requiere datos bursátiles en tiempo real: \
cotización actual, precio de la acción, ticker, valor en bolsa hoy.
- Responde "rag" si la pregunta es sobre informes financieros: resultados, \
EBITDA, beneficio, deuda, estrategia, dividendos, balances, cualquier dato histórico.
Ante la duda responde "rag"."""

_MARKET_HINTS = (
    "cotización", "cotiza", "precio actual", "precio hoy", "en tiempo real",
    "precio de mercado", "cotización ahora", "precio ahora",
    "intradia", "intradía", "52 semanas", "52-week", "volumen", "ticker",
    "variación diaria", "stock price", "market move", "trading",
)  # fmt: skip

_MARKET_PATTERN = re.compile(
    r"\b(precio|cotizaci[oó]n|market\s*price)\b.*\b(ahora|hoy|tiempo real)\b"
    r"|\b(ahora|hoy)\b.*\b(precio|cotizaci[oó]n)\b",
    re.I,
)

_FUNDAMENTAL_HINTS = (
    "ebitda", "beneficio neto", "beneficio operativo", "ingresos", "revenue",
    "deuda", "endeudamiento", "dividendo", "guidance", "margen", "balance",
    "resultado", "resultados", "informe", "reporte", "trimestre", "anual",
    "2023", "2024", "estrategia", "capex", "flujo de caja",
    "free cash flow", "roe", "roce", "book value", "per ", "p/e",
)  # fmt: skip


def _looks_like_market_question(question: str) -> bool:
    q = question.lower()
    return any(hint in q for hint in _MARKET_HINTS) or bool(_MARKET_PATTERN.search(question))


def _looks_like_fundamental_question(question: str) -> bool:
    q = question.lower()
    return any(hint in q for hint in _FUNDAMENTAL_HINTS)


@lru_cache
def _router_llm(model: str, base_url: str) -> "ChatOllama":
    # LangChain is imported lazily, after main.py has configured LangSmith.
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, base_url=base_url, temperature=0, num_predict=3)


async def _classify_with_llm(question: str, settings: Settings) -> bool:
    """Ask the router model whether the question needs the agent."""
    llm = _router_llm(settings.effective_router_model, settings.ollama_base_url)
    try:
        response = await llm.ainvoke([("system", _ROUTE_SYSTEM), ("human", question)])
    except Exception as exc:
        logger.warning("llm_route_failed", error=str(exc))
        return False
    route = str(response.content).strip().lower()
    logger.info("llm_route", question=question[:80], route=route)
    return "agent" in route  # matches "agente" and "agent"


async def _should_use_agent(question: str, settings: Settings) -> bool:
    """Market questions go to the agent, fundamentals to RAG; the LLM decides the rest."""
    if _looks_like_market_question(question):
        return True
    if _looks_like_fundamental_question(question):
        return False
    return await _classify_with_llm(question, settings)


async def _choose_route(body: QueryRequest, settings: Settings) -> Route:
    use_agent = body.use_agent or await _should_use_agent(body.question, settings)
    return "agent" if use_agent else "rag"


# ── Execution helpers ─────────────────────────────────────────────────────────


async def _run_in_thread(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run a sync callable in a worker thread, preserving the OpenTelemetry context."""
    ctx = otel_context.get_current()

    def with_context() -> T:
        token = otel_context.attach(ctx)
        try:
            return func(*args, **kwargs)
        finally:
            otel_context.detach(token)

    return await asyncio.to_thread(with_context)


async def _stream_in_thread(
    events: Callable[[], Iterator[dict[str, Any]]],
) -> AsyncIterator[dict[str, Any]]:
    """Consume a sync event iterator in a worker thread and re-yield its events."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    ctx = otel_context.get_current()

    def produce() -> None:
        token = otel_context.attach(ctx)
        try:
            for event in events():
                loop.call_soon_threadsafe(queue.put_nowait, event)
        finally:
            otel_context.detach(token)
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=produce, daemon=True).start()
    while (event := await queue.get()) is not None:
        yield event


async def _enforce_rate_limit(request: Request, settings: Settings) -> aioredis.Redis:
    redis = await dependencies.get_redis()
    await check_rate_limit(request, redis, settings)
    return redis


async def _answer_with_rag(
    question: str,
    company_filter: str | None,
    settings: Settings,
    redis: aioredis.Redis,
) -> QueryResponse:
    """Answer from the exact cache, the semantic cache or the RAG engine, in that order."""
    company_label = company_filter or "all"

    cached = await get_cached_response(redis, question, company_filter)
    if cached:
        CACHE_HITS.inc()
        QUERY_REQUESTS.labels(company=company_label, cached="true").inc()
        return QueryResponse(**{**cached, "from_cache": True})
    CACHE_MISSES.inc()

    semantic_cache = dependencies.get_semantic_cache()
    similar = await asyncio.to_thread(semantic_cache.get, question, company_filter)
    if similar:
        CACHE_HITS.inc()
        await set_cached_response(
            redis, question, similar, settings.cache_ttl_seconds, company_filter
        )
        return QueryResponse(**{**similar, "from_cache": True})

    # The outer timer is the source of truth for latency: it includes the
    # cache lookups and the thread hand-off.
    start = time.perf_counter()
    result = await _run_in_thread(dependencies.get_rag_engine().query, question, company_filter)
    latency = time.perf_counter() - start
    QUERY_LATENCY.labels(company=company_label).observe(latency)
    QUERY_REQUESTS.labels(company=company_label, cached="false").inc()

    response = QueryResponse(
        answer=result.answer,
        sources=[SourceDoc.model_validate(source) for source in result.sources],
        latency_seconds=round(latency, 3),
        from_cache=False,
        query=result.query,
    )
    payload = response.model_dump()
    await set_cached_response(redis, question, payload, settings.cache_ttl_seconds, company_filter)
    await asyncio.to_thread(semantic_cache.set, question, payload, company_filter)
    return response


async def _run_agent(question: str, thread_id: str | None) -> "AgentResult":
    agent = dependencies.get_financial_agent()
    AGENT_REQUESTS.inc()
    start = time.perf_counter()
    result = await _run_in_thread(agent.run, question, thread_id)
    AGENT_LATENCY.observe(time.perf_counter() - start)
    return result


def _agent_fields(result: "AgentResult") -> dict[str, Any]:
    return {
        "answer": result.answer,
        "latency_seconds": result.latency_seconds,
        "steps_taken": result.steps_taken,
        "tools_used": result.tools_used,
        "steps_detail": [
            {"tool": step.tool, "duration_seconds": step.duration_seconds}
            for step in result.steps_detail
        ],
        "query": result.query,
    }


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event)}\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: Request, body: QueryRequest, settings: SettingsDep) -> QueryResponse:
    """Answer a question from the financial reports (RAG only), with caching."""
    redis = await _enforce_rate_limit(request, settings)
    return await _answer_with_rag(body.question, body.company_filter, settings, redis)


@router.post("/agent", response_model=AgentResponse)
async def query_agent(request: Request, body: QueryRequest, settings: SettingsDep) -> AgentResponse:
    """Answer with the agent, which combines the reports with live market data."""
    await _enforce_rate_limit(request, settings)
    result = await _run_agent(body.question, body.thread_id)
    return AgentResponse(**_agent_fields(result))


@router.post("/ask", response_model=AskResponse)
async def ask(request: Request, body: QueryRequest, settings: SettingsDep) -> AskResponse:
    """Route fundamentals questions to RAG and market questions to the agent.

    `use_agent=true` in the body forces the agent route.
    """
    redis = await _enforce_rate_limit(request, settings)
    route = await _choose_route(body, settings)
    logger.info("ask_route", question=body.question[:80], route=route)

    if route == "agent":
        result = await _run_agent(body.question, body.thread_id)
        return AskResponse(route=route, **_agent_fields(result))

    response = await _answer_with_rag(body.question, body.company_filter, settings, redis)
    return AskResponse(route=route, **response.model_dump())


@router.post("/ask/stream")
async def ask_stream(
    request: Request, body: QueryRequest, settings: SettingsDep
) -> StreamingResponse:
    """Server-Sent Events: a route event, then step/tool and token events, then done."""
    await _enforce_rate_limit(request, settings)
    route = await _choose_route(body, settings)

    if route == "agent":
        agent = dependencies.get_financial_agent()
        events = partial(agent.stream_run, body.question, body.thread_id)
    else:
        engine = dependencies.get_rag_engine()
        events = partial(engine.stream_query, body.question, body.company_filter)

    async def event_stream() -> AsyncIterator[str]:
        yield _sse({"type": "route", "route": route})
        async for event in _stream_in_thread(events):
            yield _sse(event)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
