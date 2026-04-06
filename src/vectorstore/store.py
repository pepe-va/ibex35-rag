"""Qdrant vector store — hybrid search (dense + sparse BM25) via LangChain."""

import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode

from src.config import Settings
from src.ingestion.pdf_loader import COMPANY_TICKER_MAP, extract_metadata_from_filename
from src.logging_config import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """Manages Qdrant hybrid vector store and document ingestion."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._store: QdrantVectorStore | None = None

    def _get_store(self) -> QdrantVectorStore:
        if self._store is None:
            url = f"http://{self.settings.qdrant_host}:{self.settings.qdrant_port}"
            embeddings = OllamaEmbeddings(
                model=self.settings.ollama_embed_model,
                base_url=self.settings.ollama_base_url,
            )
            sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
            # from_documents with an empty list creates the collection if it doesn't exist
            # and connects to an existing one if it does (force_recreate=False)
            self._store = QdrantVectorStore.from_documents(
                documents=[],
                embedding=embeddings,
                sparse_embedding=sparse_embeddings,
                collection_name=self.settings.qdrant_collection,
                url=url,
                retrieval_mode=RetrievalMode.HYBRID,
                force_recreate=False,
            )
            logger.info(
                "qdrant_connected",
                url=url,
                collection=self.settings.qdrant_collection,
            )
        return self._store

    # ── 06-03: deduplication helpers ─────────────────────────────────────────

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Compute SHA-256 hash of a file for deduplication."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def get_processed_hashes(self) -> set[str]:
        """Return the set of file hashes already ingested in Qdrant."""
        processed: set[str] = set()
        offset = None
        store = self._get_store()

        while True:
            points, offset = store.client.scroll(
                collection_name=self.settings.qdrant_collection,
                limit=10_000,
                with_payload=True,
                offset=offset,
            )
            if not points:
                break
            processed.update(
                point.payload.get("metadata", {}).get("file_hash")
                for point in points
                if point.payload.get("metadata", {}).get("file_hash") is not None
            )
            if offset is None:
                break

        return processed

    # ── 06-03: ingestion function ─────────────────────────────────────────────

    @staticmethod
    def _extract_page_number(file_path: Path) -> int | None:
        """Extract page number from filenames like 'page_28.md' or 'table_1_page_5.md'."""
        match = re.search(r"page_(\d+)", file_path.stem)
        return int(match.group(1)) if match else None

    def ingest_file_in_db(self, file_path: Path, processed_hashes: set[str]) -> None:
        """
        Ingest a single .md file into Qdrant with deduplication.

        Content types detected from path:
          - markdown/  → text, split by page breaks
          - tables/    → table, one document per file
          - images_desc/ → image, one document per file
        """
        file_hash = self.compute_file_hash(file_path)
        if file_hash in processed_hashes:
            logger.info("already_ingested", file=str(file_path))
            return

        path_str = str(file_path)
        if "markdown" in path_str:
            content_type = "text"
            doc_name = file_path.stem
        elif "tables" in path_str:
            content_type = "table"
            doc_name = file_path.parent.name
        elif "images_desc" in path_str:
            content_type = "image"
            doc_name = file_path.parent.name
        else:
            content_type = "unknown"
            doc_name = file_path.stem

        content = file_path.read_text(encoding="utf-8")
        base_metadata = extract_metadata_from_filename(doc_name)
        base_metadata.update(
            {
                "content_type": content_type,
                "file_hash": file_hash,
                "source_file": doc_name,
                "ticker": COMPANY_TICKER_MAP.get(base_metadata["company"], "UNKNOWN"),
            }
        )

        store = self._get_store()

        if content_type == "text":
            pages = content.split("<!-- page break -->")
            documents = []
            for idx, page in enumerate(pages, start=1):
                if page.strip():
                    meta = {**base_metadata, "page": idx}
                    documents.append(Document(page_content=page, metadata=meta))
            store.add_documents(documents)
            logger.info("text_ingested", file=file_path.name, pages=len(documents))
        else:
            page_num = self._extract_page_number(file_path)
            meta = {**base_metadata, "page": page_num}
            store.add_documents([Document(page_content=content, metadata=meta)])
            logger.info("doc_ingested", file=file_path.name, type=content_type)

        processed_hashes.add(file_hash)

    def ingest_all_files(self, rag_data_dir: Path) -> int:
        """Ingest all .md files under rag_data_dir (markdown + tables + images_desc)."""
        all_files = list(Path(rag_data_dir).rglob("*.md"))
        if not all_files:
            raise ValueError(f"No .md files found in {rag_data_dir}")

        processed_hashes = self.get_processed_hashes()
        logger.info(
            "ingestion_started",
            total_files=len(all_files),
            already_processed=len(processed_hashes),
        )

        ingested = 0
        for md_file in all_files:
            try:
                before = len(processed_hashes)
                self.ingest_file_in_db(md_file, processed_hashes)
                if len(processed_hashes) > before:
                    ingested += 1
            except Exception:
                logger.exception("file_ingestion_error", path=str(md_file))

        logger.info("ingestion_finished", new_files=ingested)
        return ingested

    # ── Compatibility helpers (metrics + health) ──────────────────────────────

    def collection_count(self) -> int:
        """Return number of points stored in Qdrant."""
        try:
            store = self._get_store()
            info = store.client.get_collection(self.settings.qdrant_collection)
            return info.points_count or 0
        except Exception:
            return 0

    def get_metadata_stats(self) -> dict:
        """Return distinct companies and pages indexed."""
        try:
            store = self._get_store()
            companies: set = set()
            pages: set = set()
            offset = None
            while True:
                points, offset = store.client.scroll(
                    collection_name=self.settings.qdrant_collection,
                    limit=1000,
                    with_payload=True,
                    offset=offset,
                )
                if not points:
                    break
                for point in points:
                    meta = (point.payload or {}).get("metadata") or {}
                    companies.add(meta.get("company"))
                    pages.add((meta.get("source_file"), meta.get("page")))
                if offset is None:
                    break
            return {
                "companies": len(companies - {None}),
                "pages": len(pages - {(None, None)}),
            }
        except Exception:
            return {"companies": 0, "pages": 0}

    def health_check(self) -> bool:
        """Ping Qdrant."""
        try:
            self._get_store().client.get_collections()
            return True
        except Exception:
            return False
