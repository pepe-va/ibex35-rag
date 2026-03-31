"""Shared fixtures for unit and integration tests."""

import pytest
import fakeredis.aioredis as fakeredis
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from src.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        ollama_base_url="http://localhost:11434",
        ollama_llm_model="llama3.2:latest",
        ollama_embed_model="nomic-embed-text:latest",
        chroma_host="localhost",
        chroma_port=8000,
        chroma_collection="test_ibex35",
        redis_url="redis://localhost:6379/0",
        mlflow_tracking_uri="http://localhost:5000",
        mlflow_experiment_name="test-ibex35-rag",
        pdf_dir="./ibex35/Resultados financieros",
        environment="test",
        rate_limit_requests=100,
    )


@pytest.fixture
async def fake_redis():
    """In-memory Redis using fakeredis."""
    r = fakeredis.FakeRedis()
    yield r
    await r.aclose()


@pytest.fixture
def mock_rag_result():
    from src.rag.engine import QueryResult
    return QueryResult(
        answer="IBERDROLA reportó ingresos de 47.000M€ en 2024.",
        sources=[{"company": "IBERDROLA", "ticker": "IBE.MC", "score": 0.92}],
        latency_seconds=1.5,
        num_retrieved=3,
        query="¿Cuáles fueron los ingresos de Iberdrola?",
    )


@pytest.fixture
def mock_agent_result():
    from src.agents.financial_agent import AgentResult
    return AgentResult(
        answer="IBERDROLA cotiza a €12.50 con un retorno YTD del +8.2%.",
        latency_seconds=4.2,
        steps_taken=3,
        tools_used=["get_stock_price", "ibex35_financial_reports"],
        query="¿Cómo cotiza Iberdrola y cuáles son sus fundamentales?",
    )
