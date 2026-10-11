"""Ingestion endpoint: index PDFs through the API."""

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from src.api import dependencies
from src.api.dependencies import SettingsDep
from src.api.schemas import IngestionRequest, IngestionResponse
from src.core.logging import get_logger
from src.ingestion import pipeline
from src.observability.metrics import VECTOR_STORE_COUNT

router = APIRouter(tags=["ingestion"])
logger = get_logger(__name__)

_ingestion_lock = asyncio.Lock()


def _resolve_pdf_dir(requested: str | None, base_dir: str) -> Path:
    """Resolve the requested PDF directory, confined to the configured base dir."""
    base = Path(base_dir).resolve()
    if not requested:
        return base
    candidate = (base / requested).resolve()
    if not candidate.is_relative_to(base):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="pdf_dir must be inside the configured PDF directory.",
        )
    return candidate


@router.post("/ingest", response_model=IngestionResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_ingestion(body: IngestionRequest, settings: SettingsDep) -> IngestionResponse:
    """Extract the PDFs with Docling and index them in Qdrant.

    The request returns when the pipeline finishes. Only one ingestion runs at a
    time; a concurrent request gets 409.
    """
    if _ingestion_lock.locked():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ingestion already in progress. Try again later.",
        )

    pdf_dir = _resolve_pdf_dir(body.pdf_dir, settings.pdf_dir)
    vector_store = dependencies.get_vector_store()
    ingestion = pipeline.IngestionPipeline(settings=settings, vector_store=vector_store)

    async with _ingestion_lock:
        # The pipeline is synchronous; keep it off the event loop.
        result = await asyncio.to_thread(ingestion.run, pdf_dir)
    VECTOR_STORE_COUNT.set(vector_store.collection_count())

    return IngestionResponse(
        success=result.success,
        total_documents=result.total_pdfs,
        total_nodes=result.total_ingested,
        total_companies=result.total_companies,
        duration_seconds=round(result.duration_seconds, 2),
        companies=result.companies,
        errors=result.errors,
    )
