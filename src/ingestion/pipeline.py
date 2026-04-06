"""Ingestion pipeline: PDF → extract (Docling) → ingest (Qdrant hybrid)."""

import time
from dataclasses import dataclass, field
from pathlib import Path

from src.config import Settings
from src.logging_config import get_logger
from src.monitoring.metrics import (
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
    total_pdfs: int = 0
    total_ingested: int = 0
    total_companies: int = 0
    duration_seconds: float = 0.0
    companies: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and self.total_ingested > 0


class IngestionPipeline:
    """
    Two-phase pipeline:
      1. Extract PDFs → markdown + images + tables (Docling, notebook 06-01)
      2. Ingest .md files → Qdrant hybrid store (notebook 06-03)
    """

    def __init__(self, settings: Settings, vector_store: "VectorStoreManager") -> None:  # type: ignore[name-defined]  # noqa: F821
        self.settings = settings
        self.vector_store = vector_store

    def run(self, pdf_dir: str | Path | None = None) -> IngestionResult:
        """Run extraction then ingestion."""
        from src.ingestion.pdf_loader import extract_all_pdfs

        pdf_dir = Path(pdf_dir or self.settings.pdf_dir)
        rag_data_dir = Path(self.settings.rag_data_dir)
        result = IngestionResult()
        start = time.perf_counter()

        logger.info("ingestion_started", pdf_dir=str(pdf_dir), rag_data_dir=str(rag_data_dir))

        try:
            # Phase 1 — Extract PDFs to markdown/images/tables on disk
            md_files = extract_all_pdfs(pdf_dir, rag_data_dir)
            result.total_pdfs = len(md_files)

            # Collect unique companies from extracted markdown files
            companies: set[str] = set()
            for md in md_files:
                # path: .../markdown/{COMPANY}/{stem}.md
                companies.add(md.parent.name)
            result.companies = sorted(companies)
            result.total_companies = len(result.companies)

            logger.info(
                "extraction_complete",
                pdfs=result.total_pdfs,
                companies=result.total_companies,
            )

            # Phase 2 — Ingest all .md files from rag_data_dir into Qdrant
            result.total_ingested = self.vector_store.ingest_all_files(rag_data_dir)
            logger.info("ingestion_complete", new_files=result.total_ingested)

        except Exception as exc:
            error_msg = str(exc)
            result.errors.append(error_msg)
            logger.exception("ingestion_failed", error=error_msg)
            raise

        finally:
            result.duration_seconds = time.perf_counter() - start
            status = "success" if result.success else "error"
            INGESTION_RUNS.labels(status=status).inc()
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

        return result
