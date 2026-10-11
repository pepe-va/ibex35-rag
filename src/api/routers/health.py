"""Health check endpoints."""

from typing import Any

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api import dependencies
from src.api.dependencies import SettingsDep
from src.api.schemas import HealthResponse
from src.core.config import Settings
from src.core.logging import get_logger

router = APIRouter(tags=["health"])
logger = get_logger(__name__)

HEALTHY = "healthy"
UNHEALTHY = "unhealthy"
OLLAMA_TIMEOUT_SECONDS = 5.0


def _vector_store_status() -> tuple[str, int]:
    """Return the vector store status and its number of points."""
    try:
        vector_store = dependencies.get_vector_store()
        if vector_store.health_check():
            return HEALTHY, vector_store.collection_count()
    except Exception as exc:
        logger.warning("health_vector_store_failed", error=str(exc))
    return UNHEALTHY, 0


async def _redis_status() -> str:
    try:
        redis = await dependencies.get_redis()
        await redis.ping()  # type: ignore[misc]  # redis-py types ping() as sync or async
    except Exception as exc:
        logger.warning("health_redis_failed", error=str(exc))
        return UNHEALTHY
    return HEALTHY


async def _ollama_status(settings: Settings) -> str:
    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{settings.ollama_base_url}/api/tags")
    except Exception as exc:
        logger.warning("health_ollama_failed", error=str(exc))
        return UNHEALTHY
    return HEALTHY if response.status_code == httpx.codes.OK else UNHEALTHY


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: SettingsDep) -> JSONResponse:
    """Full health check of the vector store, Redis and Ollama (200 healthy, 207 degraded)."""
    vector_store, collection_count = _vector_store_status()
    statuses = {
        "vector_store": vector_store,
        "redis": await _redis_status(),
        "ollama": await _ollama_status(settings),
    }
    overall = HEALTHY if all(s == HEALTHY for s in statuses.values()) else "degraded"
    health = HealthResponse(
        status=overall,
        version=settings.app_version,
        collection_count=collection_count,
        **statuses,
    )
    status_code = 200 if overall == HEALTHY else 207
    return JSONResponse(content=health.model_dump(), status_code=status_code)


@router.get("/health/live")
async def liveness() -> dict[str, Any]:
    """Liveness probe: the process is alive."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness() -> JSONResponse:
    """Readiness probe: the vector store is reachable."""
    try:
        ready = dependencies.get_vector_store().health_check()
    except Exception as exc:
        logger.warning("readiness_check_failed", error=str(exc))
        return JSONResponse({"status": "not_ready", "reason": "vector_store_error"}, 503)
    if not ready:
        return JSONResponse({"status": "not_ready", "reason": "vector_store"}, 503)
    return JSONResponse({"status": "ready"})
