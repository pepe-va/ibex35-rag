"""Sliding window rate limiter using Redis."""

import time

import redis.asyncio as aioredis
from fastapi import HTTPException, Request, status

from src.config import Settings
from src.logging_config import get_logger

logger = get_logger(__name__)


async def check_rate_limit(
    request: Request,
    redis: aioredis.Redis,
    settings: Settings,
) -> None:
    """Sliding window rate limiter keyed by client IP."""
    client_ip = request.client.host if request.client else "unknown"
    key = f"ibex35:ratelimit:{client_ip}"
    now = time.time()
    window_start = now - settings.rate_limit_window_seconds

    pipe = await redis.pipeline()
    # Remove old entries outside the window
    pipe.zremrangebyscore(key, 0, window_start)
    # Count requests in the current window
    pipe.zcard(key)
    # Add current request
    pipe.zadd(key, {str(now): now})
    # Set expiry so keys clean themselves up
    pipe.expire(key, settings.rate_limit_window_seconds * 2)

    try:
        results = await pipe.execute()
        request_count = results[1]

        if request_count >= settings.rate_limit_requests:
            logger.warning(
                "rate_limit_exceeded",
                client_ip=client_ip,
                count=request_count,
                limit=settings.rate_limit_requests,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "limit": settings.rate_limit_requests,
                    "window_seconds": settings.rate_limit_window_seconds,
                },
                headers={"Retry-After": str(settings.rate_limit_window_seconds)},
            )
    except HTTPException:
        raise
    except Exception as exc:
        # Don't block requests if Redis is down
        logger.warning("rate_limit_redis_error", error=str(exc))
