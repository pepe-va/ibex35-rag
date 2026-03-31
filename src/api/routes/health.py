"""Health check endpoints."""

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.api.dependencies import get_redis, get_settings_dep, get_vector_store
from src.api.models import HealthResponse
from src.config import Settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    settings: Settings = Depends(get_settings_dep),
) -> HealthResponse:
    """Full health check: vector store, Redis, Ollama."""
    # Vector store
    vs_status = "unhealthy"
    collection_count = 0
    try:
        vs = get_vector_store()
        if vs.health_check():
            vs_status = "healthy"
            collection_count = vs.collection_count()
    except Exception:
        pass

    # Redis
    redis_status = "unhealthy"
    try:
        r = await get_redis()
        await r.ping()
        redis_status = "healthy"
    except Exception:
        pass

    # Ollama
    ollama_status = "unhealthy"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            if resp.status_code == 200:
                ollama_status = "healthy"
    except Exception:
        pass

    overall = (
        "healthy"
        if all(s == "healthy" for s in [vs_status, redis_status, ollama_status])
        else "degraded"
    )

    health = HealthResponse(
        status=overall,
        version=settings.app_version,
        vector_store=vs_status,
        redis=redis_status,
        ollama=ollama_status,
        collection_count=collection_count,
    )

    status_code = 200 if overall == "healthy" else 207
    return JSONResponse(content=health.model_dump(), status_code=status_code)


@router.get("/health/live")
async def liveness() -> dict:
    """Kubernetes liveness probe — just confirms the process is alive."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness() -> dict:
    """Kubernetes readiness probe — confirms the app can serve traffic."""
    try:
        vs = get_vector_store()
        if not vs.health_check():
            return JSONResponse({"status": "not_ready", "reason": "vector_store"}, status_code=503)
    except Exception as exc:
        return JSONResponse({"status": "not_ready", "reason": str(exc)}, status_code=503)
    return {"status": "ready"}
