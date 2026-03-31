"""Ingestion pipeline: PDF → chunks → embeddings → vector store."""

import time
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from src.config import Settings
from src.logging_config import get_logger

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
        nodes = self.splitter.get_nodes_from_documents(documents)
        logger.info("documents_chunked", input_docs=len(documents), output_nodes=len(nodes))
        return nodes

    def run(self, pdf_dir: str | Path | None = None) -> IngestionResult:
        """Run the full ingestion pipeline with MLflow tracking."""
        from src.ingestion.pdf_loader import load_all_pdfs

        pdf_dir = pdf_dir or self.settings.pdf_dir
        result = IngestionResult()
        start_time = time.perf_counter()

        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.settings.mlflow_experiment_name)

        with mlflow.start_run(run_name="ingestion"):
            mlflow.set_tag("pipeline", "ingestion")
            mlflow.log_param("pdf_dir", str(pdf_dir))
            mlflow.log_param("chunk_size", self.settings.chunk_size)
            mlflow.log_param("chunk_overlap", self.settings.chunk_overlap)
            mlflow.log_param("embed_model", self.settings.ollama_embed_model)

            try:
                # 1. Load PDFs
                logger.info("ingestion_started", pdf_dir=str(pdf_dir))
                documents = load_all_pdfs(pdf_dir)
                result.total_documents = len(documents)
                result.companies = sorted({d.metadata["company"] for d in documents})
                result.total_companies = len(result.companies)
                mlflow.log_metric("total_documents", result.total_documents)
                mlflow.log_metric("total_companies", result.total_companies)

                # 2. Chunk
                nodes = self._chunk_documents(documents)
                result.total_nodes = len(nodes)
                mlflow.log_metric("total_nodes", result.total_nodes)
                mlflow.log_metric(
                    "avg_nodes_per_doc",
                    result.total_nodes / max(result.total_documents, 1),
                )

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
                mlflow.log_param("error", error_msg)
                logger.exception("ingestion_failed", error=error_msg)
                raise

            finally:
                result.duration_seconds = time.perf_counter() - start_time
                mlflow.log_metric("duration_seconds", result.duration_seconds)
                mlflow.log_metric("success", int(result.success))

        return result
