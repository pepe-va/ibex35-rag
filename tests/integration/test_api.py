"""Integration tests for the FastAPI endpoints (mocked dependencies)."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def client():
    """TestClient with all heavy dependencies mocked."""
    with (
        patch("src.api.dependencies.get_vector_store") as mock_vs,
        patch("src.api.dependencies.get_rag_engine") as mock_rag,
        patch("src.api.dependencies.get_financial_agent") as mock_agent,
        patch("src.api.dependencies.get_redis", new_callable=AsyncMock) as mock_redis,
    ):
        # Mock Redis
        redis_mock = AsyncMock()
        redis_mock.get = AsyncMock(return_value=None)
        redis_mock.setex = AsyncMock(return_value=True)
        redis_mock.zremrangebyscore = AsyncMock()
        redis_mock.zcard = AsyncMock(return_value=0)
        redis_mock.zadd = AsyncMock()
        redis_mock.expire = AsyncMock()
        redis_mock.pipeline.return_value.__aenter__ = AsyncMock(return_value=redis_mock)
        redis_mock.pipeline.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_redis.return_value = redis_mock

        # Fake pipeline execute for rate limiter
        pipe_mock = MagicMock()
        pipe_mock.execute = AsyncMock(return_value=[None, 0, None, None])
        pipe_mock.zremrangebyscore = MagicMock()
        pipe_mock.zcard = MagicMock()
        pipe_mock.zadd = MagicMock()
        pipe_mock.expire = MagicMock()
        redis_mock.pipeline.return_value = pipe_mock

        # Mock RAG engine
        from src.rag.engine import QueryResult
        rag_mock = MagicMock()
        rag_mock.query.return_value = QueryResult(
            answer="IBERDROLA reportó ingresos de €47.000M en 2024.",
            sources=[{"company": "IBERDROLA", "ticker": "IBE.MC", "score": 0.91}],
            latency_seconds=1.2,
            num_retrieved=1,
            query="¿Cuáles fueron los ingresos de IBERDROLA?",
        )
        mock_rag.return_value = rag_mock

        # Mock vector store
        vs_mock = MagicMock()
        vs_mock.health_check.return_value = True
        vs_mock.collection_count.return_value = 150
        mock_vs.return_value = vs_mock

        from src.api.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def test_liveness(client):
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_query_returns_answer(client):
    resp = client.post(
        "/api/v1/query",
        json={"question": "¿Cuáles fueron los ingresos de IBERDROLA en 2024?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "IBERDROLA" in data["answer"]
    assert data["from_cache"] is False
    assert data["latency_seconds"] > 0


def test_query_validates_min_length(client):
    resp = client.post("/api/v1/query", json={"question": "hi"})
    assert resp.status_code == 422


def test_query_validates_max_length(client):
    resp = client.post("/api/v1/query", json={"question": "x" * 1001})
    assert resp.status_code == 422


def test_metrics_endpoint_available(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"ibex35_rag" in resp.content or b"http_requests" in resp.content
