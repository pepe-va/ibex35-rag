"""Semantic cache backed by a dedicated Qdrant collection.

Stores query embeddings alongside their RAG results. On lookup, performs
ANN search and returns the cached result if similarity >= threshold.
"""

import json
import time
import uuid
from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.core.config import Settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# FastEmbed runs in-process (no network), model cached after first load.
# BAAI/bge-small-en-v1.5 → 384d, fast, good multilingual support.
_FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
_FASTEMBED_DIM = 384


class SemanticCache:
    """Cache RAG responses by semantic similarity of the query vector.

    Uses FastEmbed (in-process, no Ollama call) to avoid adding latency
    to every cache miss. If the model can't be loaded, operates as a no-op cache.
    """

    def __init__(self, settings: Settings) -> None:
        self._client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self._collection = settings.semantic_cache_collection
        self._threshold = settings.semantic_cache_threshold
        self._ttl = settings.cache_ttl_seconds
        self._enabled = False
        try:
            self._embed_model = TextEmbedding(model_name=_FASTEMBED_MODEL)
            self._ensure_collection(_FASTEMBED_DIM)
            self._enabled = True
        except Exception as exc:
            logger.warning(
                "semantic_cache_disabled",
                reason=str(exc),
                hint="Pre-download model or mount fastembed cache to enable semantic cache",
            )

    def _ensure_collection(self, vector_size: int) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info("semantic_cache_collection_created", collection=self._collection)

    def _embed(self, text: str) -> list[float]:
        return list(next(iter(self._embed_model.embed([text]))))

    def get(self, question: str, company_filter: str | None = None) -> dict[str, Any] | None:
        """Return cached result if a semantically similar query exists, else None."""
        if not self._enabled:
            return None
        try:
            vector = self._embed(question)
            cf = company_filter or ""
            response = self._client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=1,
                score_threshold=self._threshold,
                query_filter=Filter(
                    must=[FieldCondition(key="company_filter", match=MatchValue(value=cf))]
                ),
                with_payload=True,
            )
            if not response.points:
                logger.debug("semantic_cache_miss", question=question[:60])
                return None

            hit = response.points[0]
            payload = hit.payload or {}
            created_at = payload.get("created_at", 0.0)
            if time.time() - created_at > self._ttl:
                self._client.delete(collection_name=self._collection, points_selector=[hit.id])
                logger.debug("semantic_cache_expired", question=question[:60])
                return None

            logger.info("semantic_cache_hit", question=question[:60], score=round(hit.score, 3))
            cached: dict[str, Any] = json.loads(payload["result_json"])
            return cached

        except Exception as exc:
            logger.warning("semantic_cache_get_error", error=str(exc))
            return None

    def set(self, question: str, result: dict[str, Any], company_filter: str | None = None) -> None:
        """Store query embedding + result in the cache collection."""
        if not self._enabled:
            return
        try:
            vector = self._embed(question)
            self._client.upsert(
                collection_name=self._collection,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "question": question,
                            "company_filter": company_filter or "",
                            "result_json": json.dumps(result),
                            "created_at": time.time(),
                        },
                    )
                ],
            )
            logger.info("semantic_cache_set", question=question[:60])
        except Exception as exc:
            logger.warning("semantic_cache_set_error", error=str(exc))
