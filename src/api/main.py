"""FastAPI application entry point."""

import asyncio
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.base import RequestResponseEndpoint

from src.api import dependencies
from src.api.routers import health, ingest, query
from src.core.config import Settings, get_settings
from src.core.logging import get_logger, setup_logging
from src.observability.langsmith_collector import collect_langsmith_metrics
from src.observability.metrics import (
    CACHE_BYTES_LIMIT,
    CACHE_BYTES_USED,
    CACHE_EVICTIONS,
    CACHE_KEYS_CURRENT,
    COMPANIES_INDEXED,
    PAGES_INDEXED,
    TOP_K,
    VECTOR_STORE_COUNT,
)

SERVICE_NAME = "ibex35-rag-api"
METRICS_REFRESH_SECONDS = 30


def configure_langsmith(settings: Settings) -> str:
    """Export the LangSmith settings for LangChain and return a status for the startup log.

    LangChain reads these variables when it traces its first call. Every LangChain
    import in the app is lazy (see `src/api/dependencies.py`), so this runs first;
    `tests/unit/api/test_main.py` guards that.
    """
    if not (settings.langsmith_tracing and settings.langsmith_api_key):
        return "disabled (set LANGSMITH_TRACING=true and LANGSMITH_API_KEY)"
    os.environ.update(
        {
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_API_KEY": settings.langsmith_api_key.get_secret_value(),
            "LANGSMITH_PROJECT": settings.langsmith_project,
            "LANGSMITH_ENDPOINT": settings.langsmith_endpoint,
        }
    )
    return (
        f"enabled (project={settings.langsmith_project}, "
        f"endpoint={settings.langsmith_endpoint}, "
        f"interval={settings.langsmith_collector_interval_seconds}s)"
    )


def setup_tracing(settings: Settings) -> TracerProvider:
    """Send traces to the OTel Collector and instrument the Redis and HTTPX clients."""
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)
    RedisInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()
    return provider


settings = get_settings()
langsmith_status = configure_langsmith(settings)
tracer_provider = setup_tracing(settings)
setup_logging(settings.log_level)
logger = get_logger(__name__)


# ── Background metrics ────────────────────────────────────────────────────────


def _load_index_metrics() -> None:
    """Publish the index gauges and make sure the Qdrant payload indexes exist."""
    vector_store = dependencies.get_vector_store()
    count = vector_store.collection_count()
    VECTOR_STORE_COUNT.set(count)
    TOP_K.set(settings.similarity_top_k)
    logger.info("vector_store_count_loaded", count=count)

    stats = vector_store.get_metadata_stats()
    COMPANIES_INDEXED.set(stats["companies"])
    PAGES_INDEXED.set(stats["pages"])
    logger.info("metadata_stats_loaded", companies=stats["companies"], pages=stats["pages"])

    vector_store.ensure_payload_indexes()


async def _refresh_cache_metrics() -> None:
    redis = await dependencies.get_redis()
    memory = await redis.info("memory")
    CACHE_BYTES_USED.set(memory.get("used_memory", 0))
    CACHE_BYTES_LIMIT.set(memory.get("maxmemory", 0))
    stats = await redis.info("stats")
    CACHE_EVICTIONS.set(stats.get("evicted_keys", 0))
    CACHE_KEYS_CURRENT.set(await redis.dbsize())


async def _refresh_metrics_forever() -> None:
    """Refresh the Redis and vector store gauges periodically."""
    while True:
        await asyncio.sleep(METRICS_REFRESH_SECONDS)
        try:
            await _refresh_cache_metrics()
        except Exception as exc:
            logger.warning("cache_metrics_refresh_failed", error=str(exc))
        try:
            VECTOR_STORE_COUNT.set(dependencies.get_vector_store().collection_count())
        except Exception as exc:
            logger.warning("vector_store_metrics_refresh_failed", error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info(
        "app_starting",
        name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        langsmith=langsmith_status,
    )
    try:
        _load_index_metrics()
    except Exception as exc:
        logger.warning("vector_store_unavailable_at_startup", error=str(exc))

    background_tasks = [
        asyncio.create_task(_refresh_metrics_forever()),
        asyncio.create_task(collect_langsmith_metrics(settings)),
    ]
    yield
    for task in background_tasks:
        task.cancel()
    logger.info("app_stopping")
    await dependencies.close_redis()


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="IBEX35 RAG API",
    description=(
        "Production-ready Retrieval-Augmented Generation system for IBEX35 financial reports. "
        "Powered by LangChain + Ollama + Qdrant."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next: RequestResponseEndpoint) -> Response:
    """Bind request and trace context to every log line, then log the response."""
    start = time.perf_counter()
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        path=request.url.path,
        method=request.method,
        client_ip=request.client.host if request.client else "unknown",
    )
    span_context = otel_trace.get_current_span().get_span_context()
    if span_context.is_valid:
        structlog.contextvars.bind_contextvars(trace_id=format(span_context.trace_id, "032x"))
    response = await call_next(request)
    duration = time.perf_counter() - start
    logger.info("request_handled", status_code=response.status_code, duration=round(duration, 3))
    return response


Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    excluded_handlers=["/health/live", "/health/ready", "/metrics"],
).instrument(app).expose(app, endpoint="/metrics")

app.include_router(health.router)
app.include_router(query.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception", error=str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
