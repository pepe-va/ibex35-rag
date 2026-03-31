"""FastAPI application entry point."""

import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.dependencies import close_redis
from src.api.routes import health, ingest, query
from src.config import get_settings
from src.logging_config import get_logger, setup_logging

settings = get_settings()
setup_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "app_starting",
        name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )
    yield
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
