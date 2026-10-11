"""Ingestion pipeline: PDF → Markdown (Docling) → Qdrant hybrid index."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.config import Settings
from src.core.logging import get_logger
from src.observability.metrics import (
    COMPANIES_INDEXED,
    INGESTION_DURATION,
    INGESTION_LAST_DURATION,
    INGESTION_RUNS,
    PAGES_INDEXED,
    VECTOR_STORE_COUNT,
)

if TYPE_CHECKING:
    from src.rag.vector_store.store import VectorStoreManager

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    total_pdfs: int = 0
    total_ingested: int = 0
    total_companies: int = 0
    duration_seconds: float = 0.0
    companies: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.errors and self.total_ingested > 0


class IngestionPipeline:
    """Two phases: extract the PDFs to Markdown, then index the Markdown in Qdrant."""

    def __init__(self, settings: Settings, vector_store: "VectorStoreManager") -> None:
        self.settings = settings
        self.vector_store = vector_store

    def run(self, pdf_dir: str | Path | None = None) -> IngestionResult:
        """Run extraction, then indexing. Re-raises any failure after recording it."""
        # Docling is heavy; load it only when an ingestion actually runs.
        from src.ingestion.pdf_loader import extract_all_pdfs

        pdf_dir = Path(pdf_dir or self.settings.pdf_dir)
        rag_data_dir = Path(self.settings.rag_data_dir)
        result = IngestionResult()
        started_at = time.perf_counter()
        logger.info("ingestion_started", pdf_dir=str(pdf_dir), rag_data_dir=str(rag_data_dir))

        try:
            md_files = extract_all_pdfs(pdf_dir, rag_data_dir)
            result.total_pdfs = len(md_files)
            # Markdown lives at .../markdown/<COMPANY>/<stem>.md
            result.companies = sorted({md.parent.name for md in md_files})
            result.total_companies = len(result.companies)
            logger.info(
                "extraction_complete", pdfs=result.total_pdfs, companies=result.total_companies
            )

            result.total_ingested = self.vector_store.ingest_all_files(rag_data_dir)
            logger.info("ingestion_complete", new_files=result.total_ingested)
        except Exception as exc:
            result.errors.append(str(exc))
            logger.exception("ingestion_failed", error=str(exc))
            raise
        finally:
            result.duration_seconds = time.perf_counter() - started_at
            self._record_metrics(result)

        return result

    def _record_metrics(self, result: IngestionResult) -> None:
        INGESTION_RUNS.labels(status="success" if result.success else "error").inc()
        INGESTION_DURATION.observe(result.duration_seconds)
        INGESTION_LAST_DURATION.set(result.duration_seconds)
        VECTOR_STORE_COUNT.set(self.vector_store.collection_count())
        COMPANIES_INDEXED.set(result.total_companies)
        PAGES_INDEXED.set(result.total_pdfs)
        logger.info(
            "ingestion_finished",
            duration_seconds=round(result.duration_seconds, 2),
            success=result.success,
        )
