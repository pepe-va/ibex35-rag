"""Ingestion pipeline: PDF → chunks → embeddings → vector store."""

import time
from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from src.config import Settings
from src.logging_config import get_logger
from src.monitoring.metrics import (
    CHUNK_SIZE_BYTES,
    CHUNKS_CREATED,
    CHUNKING_DURATION,
    COMPANIES_INDEXED,
    INGESTION_DURATION,
    INGESTION_LAST_DURATION,
    INGESTION_RUNS,
    PAGES_INDEXED,
    VECTOR_STORE_COUNT,
)

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    total_documents: int = 0
    total_nodes: int = 0
    total_companies: int = 0
    duration_seconds: float = 0.0
    companies: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and self.total_nodes > 0


class IngestionPipeline:
    """Orchestrates PDF ingestion: load → chunk → embed → store."""

    def __init__(self, settings: Settings, vector_store: "VectorStoreManager") -> None:  # type: ignore[name-defined]  # noqa: F821
        self.settings = settings
        self.vector_store = vector_store
        self.splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def _chunk_documents(self, documents: list[Document]) -> list:
        """Split documents into nodes (chunks)."""
        with CHUNKING_DURATION.time():
            nodes = self.splitter.get_nodes_from_documents(documents)
        for node in nodes:
            CHUNK_SIZE_BYTES.observe(len(node.text.encode("utf-8")))
        CHUNKS_CREATED.set(len(nodes))
        logger.info("documents_chunked", input_docs=len(documents), output_nodes=len(nodes))
        return nodes

    def run(self, pdf_dir: str | Path | None = None) -> IngestionResult:
        """Run the full ingestion pipeline."""
        from src.ingestion.pdf_loader import load_all_pdfs

        pdf_dir = pdf_dir or self.settings.pdf_dir
        result = IngestionResult()
        start_time = time.perf_counter()

        logger.info(
            "ingestion_started",
            pdf_dir=str(pdf_dir),
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            embed_model=self.settings.ollama_embed_model,
        )

        try:
            # 1. Load PDFs
            documents = load_all_pdfs(pdf_dir)
            result.total_documents = len(documents)
            result.companies = sorted({d.metadata["company"] for d in documents})
            result.total_companies = len(result.companies)
            logger.info("pdfs_loaded", total_documents=result.total_documents, total_companies=result.total_companies)

            # 2. Chunk
            nodes = self._chunk_documents(documents)
            result.total_nodes = len(nodes)

            # 3. Embed + store
            self.vector_store.add_nodes(nodes)
            logger.info(
                "ingestion_complete",
                companies=result.total_companies,
                documents=result.total_documents,
                nodes=result.total_nodes,
            )

        except Exception as exc:
            error_msg = str(exc)
            result.errors.append(error_msg)
            logger.exception("ingestion_failed", error=error_msg)
            raise

        finally:
            result.duration_seconds = time.perf_counter() - start_time
            status = "success" if result.success else "error"
            INGESTION_RUNS.labels(status=status).inc()
            INGESTION_DURATION.observe(result.duration_seconds)
            INGESTION_LAST_DURATION.set(result.duration_seconds)
            VECTOR_STORE_COUNT.set(result.total_nodes)
            COMPANIES_INDEXED.set(result.total_companies)
            PAGES_INDEXED.set(result.total_documents)
            logger.info(
                "ingestion_finished",
                duration_seconds=round(result.duration_seconds, 2),
                success=result.success,
            )

        return result
