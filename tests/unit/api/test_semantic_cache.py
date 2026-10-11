"""Unit tests for the Qdrant-backed semantic cache."""

import json
import time
from unittest.mock import create_autospec, patch

from qdrant_client import QdrantClient
from qdrant_client.http.models import QueryResponse, ScoredPoint

from src.api.semantic_cache import SemanticCache


def _cache_with_points(settings, points: list[ScoredPoint]):
    # autospec: calling a method the installed qdrant-client lacks fails the test.
    client = create_autospec(QdrantClient, instance=True)
    client.get_collections.return_value.collections = []
    client.query_points.return_value = QueryResponse(points=points)
    with (
        patch("src.api.semantic_cache.QdrantClient", return_value=client),
        patch("src.api.semantic_cache.TextEmbedding") as embedding,
    ):
        embedding.return_value.embed.side_effect = lambda texts: iter([[0.1, 0.2, 0.3]])
        return SemanticCache(settings), client


def _point(created_at: float, answer: str = "cached") -> ScoredPoint:
    payload = {"created_at": created_at, "result_json": json.dumps({"answer": answer})}
    return ScoredPoint(id=1, version=1, score=0.97, payload=payload)


def test_hit_returns_cached_result(settings):
    cache, _ = _cache_with_points(settings, [_point(time.time())])
    assert cache.get("¿EBITDA de BBVA?") == {"answer": "cached"}


def test_miss_returns_none(settings):
    cache, _ = _cache_with_points(settings, [])
    assert cache.get("¿EBITDA de BBVA?") is None


def test_expired_entry_is_deleted(settings):
    cache, client = _cache_with_points(settings, [_point(time.time() - 10 * 86_400)])
    assert cache.get("¿EBITDA de BBVA?") is None
    client.delete.assert_called_once()
