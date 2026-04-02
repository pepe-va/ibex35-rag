"""Ingestion endpoint: trigger PDF indexing via API."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_settings_dep, get_vector_store
from src.api.models import IngestionRequest, IngestionResponse
from src.config import Settings
from src.logging_config import get_logger
from src.monitoring.metrics import VECTOR_STORE_COUNT

router = APIRouter(tags=["ingestion"])
logger = get_logger(__name__)

_ingestion_running = False


@router.post("/ingest", response_model=IngestionResponse, status_code=status.HTTP_202_ACCEPTED)
async def trigger_ingestion(
    body: IngestionRequest,
    settings: Settings = Depends(get_settings_dep),
) -> IngestionResponse:
    """
    Trigger PDF ingestion in the background.

    Loads all PDFs from the configured directory, chunks them, computes embeddings
    with Ollama, and stores vectors in ChromaDB.
    """
    global _ingestion_running
    if _ingestion_running:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ingestion already in progress. Try again later.",
        )

    from src.ingestion.pipeline import IngestionPipeline

    vs = get_vector_store()
    pipeline = IngestionPipeline(settings=settings, vector_store=vs)

    try:
        _ingestion_running = True
        # Run sync pipeline in a thread to avoid blocking the event loop
        result = await asyncio.to_thread(pipeline.run, body.pdf_dir)
    finally:
        _ingestion_running = False
        VECTOR_STORE_COUNT.set(vs.collection_count())

    return IngestionResponse(
        success=result.success,
        total_documents=result.total_documents,
        total_nodes=result.total_nodes,
        total_companies=result.total_companies,
        duration_seconds=round(result.duration_seconds, 2),
        companies=result.companies,
        errors=result.errors,
    )
