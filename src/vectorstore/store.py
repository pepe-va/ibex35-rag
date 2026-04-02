"""ChromaDB vector store manager using LlamaIndex abstractions."""

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import Settings
from src.logging_config import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """Manages ChromaDB vector store lifecycle and operations."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: chromadb.HttpClient | None = None
        self._collection: chromadb.Collection | None = None
        self._index: VectorStoreIndex | None = None
        self._embed_model = OllamaEmbedding(
            model_name=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
        )

    def _get_client(self) -> chromadb.HttpClient:
        if self._client is None:
            self._client = chromadb.HttpClient(
                host=self.settings.chroma_host,
                port=self.settings.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info(
                "chroma_connected",
                host=self.settings.chroma_host,
                port=self.settings.chroma_port,
            )
        return self._client

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("collection_ready", name=self.settings.chroma_collection)
        return self._collection

    def get_index(self) -> VectorStoreIndex:
        """Return (or build) the LlamaIndex VectorStoreIndex."""
        if self._index is None:
            collection = self._get_collection()
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
                embed_model=self._embed_model,
            )
            logger.info("index_ready", collection=self.settings.chroma_collection)
        return self._index

    # nomic-embed-text context limit ≈ 512 tokens ~ 2000 chars
    _MAX_CHARS = 2000

    def add_nodes(self, nodes: list[BaseNode], batch_size: int = 32) -> None:
        """Embed nodes and add them to the vector store in batches."""
        index = self.get_index()
        truncated = 0
        for node in nodes:
            if len(node.text) > self._MAX_CHARS:
                node.text = node.text[: self._MAX_CHARS]
                truncated += 1
        if truncated:
            logger.warning("nodes_truncated", count=truncated, max_chars=self._MAX_CHARS)
        logger.info("adding_nodes", count=len(nodes), batch_size=batch_size)
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            index.insert_nodes(batch)
            logger.info("batch_added", batch=f"{i // batch_size + 1}", nodes=len(batch))
        logger.info("nodes_added", count=len(nodes))

    def collection_count(self) -> int:
        """Return number of vectors stored."""
        try:
            return self._get_collection().count()
        except Exception:
            return 0

    def get_metadata_stats(self) -> dict:
        """Return stats about indexed content: companies and pages."""
        try:
            results = self._get_collection().get(include=["metadatas"])
            companies: set = set()
            pages: set = set()
            for m in results.get("metadatas") or []:
                if m:
                    companies.add(m.get("company"))
                    pages.add((m.get("source"), m.get("page")))
            return {
                "companies": len(companies - {None}),
                "pages": len(pages - {(None, None)}),
            }
        except Exception:
            return {"companies": 0, "pages": 0}

    def health_check(self) -> bool:
        """Ping ChromaDB."""
        try:
            self._get_client().heartbeat()
            return True
        except Exception:
            return False
