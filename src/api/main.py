"""FastAPI application entry point."""

import asyncio
import os
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.dependencies import close_redis, get_redis, get_vector_store
from src.api.routes import health, ingest, query
from src.config import get_settings
from src.logging_config import get_logger, setup_logging
from src.monitoring.metrics import (
    CACHE_BYTES_LIMIT,
    CACHE_BYTES_USED,
    CACHE_EVICTIONS,
    CACHE_KEYS_CURRENT,
    COMPANIES_INDEXED,
    PAGES_INDEXED,
    TOP_K,
    VECTOR_STORE_COUNT,
)

# ── OpenTelemetry ──────────────────────────────────────────────────────────────
_otel_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
_resource = Resource.create({"service.name": "ibex35-rag-api"})
_provider = TracerProvider(resource=_resource)
_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=_otel_endpoint, insecure=True)))
otel_trace.set_tracer_provider(_provider)
RedisInstrumentor().instrument()
HTTPXClientInstrumentor().instrument()

settings = get_settings()
setup_logging(settings.log_level)
logger = get_logger(__name__)


async def _refresh_redis_metrics() -> None:
    """Background task: update Redis memory/cache gauges every 30 s."""
    while True:
        await asyncio.sleep(30)
        try:
            r = await get_redis()
            mem = await r.info("memory")
            CACHE_BYTES_USED.set(mem.get("used_memory", 0))
            limit = mem.get("maxmemory", 0)
            CACHE_BYTES_LIMIT.set(limit)
            stats = await r.info("stats")
            CACHE_EVICTIONS.set(stats.get("evicted_keys", 0))
            keys = await r.dbsize()
            CACHE_KEYS_CURRENT.set(keys)
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "app_starting",
        name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )
    try:
        vs = get_vector_store()
        count = vs.collection_count()
        VECTOR_STORE_COUNT.set(count)
        TOP_K.set(settings.similarity_top_k)
        logger.info("vector_store_count_loaded", count=count)
        stats = vs.get_metadata_stats()
        COMPANIES_INDEXED.set(stats["companies"])
        PAGES_INDEXED.set(stats["pages"])
        logger.info("metadata_stats_loaded", companies=stats["companies"], pages=stats["pages"])
    except Exception:
        pass
    task = asyncio.create_task(_refresh_redis_metrics())
    yield
    task.cancel()
    logger.info("app_stopping")
    await close_redis()


app = FastAPI(
    title="IBEX35 RAG API",
    description=(
        "Production-ready Retrieval-Augmented Generation system for IBEX35 financial reports. "
        "Powered by LlamaIndex + Ollama + ChromaDB."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

FastAPIInstrumentor.instrument_app(app, tracer_provider=_provider)

# ── Middleware ─────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not settings.is_production else [],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        path=request.url.path,
        method=request.method,
        client_ip=request.client.host if request.client else "unknown",
    )
    span = otel_trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.is_valid:
        structlog.contextvars.bind_contextvars(trace_id=format(ctx.trace_id, "032x"))
    response = await call_next(request)
    duration = time.perf_counter() - start
    logger.info("request_handled", status_code=response.status_code, duration=round(duration, 3))
    return response


# ── Prometheus ─────────────────────────────────────────────────────────────────

Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    excluded_handlers=["/health/live", "/health/ready", "/metrics"],
).instrument(app).expose(app, endpoint="/metrics")

# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(query.router, prefix="/api/v1")
app.include_router(ingest.router, prefix="/api/v1")


# ── Exception handlers ─────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )
