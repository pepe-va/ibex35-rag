"""Qdrant vector store: hybrid (dense + BM25) search, chunking and deduplicated ingestion."""

import hashlib
import json
import re
import threading
import time
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import PayloadSchemaType

from src.core.config import Settings
from src.core.logging import get_logger
from src.ingestion.metadata import COMPANY_TICKER_MAP, PAGE_BREAK, extract_metadata_from_filename
from src.observability.metrics import CHUNK_SIZE_BYTES, CHUNKING_DURATION, CHUNKS_CREATED

logger = get_logger(__name__)

INGEST_WORKERS = 4
SCROLL_BATCH_SIZE = 1000
HASH_BLOCK_SIZE = 4096
FILTERABLE_FIELDS = ("metadata.company", "metadata.fiscal_year", "metadata.fiscal_quarter")

# Content type by the folder a file lives in (checked in order).
_CONTENT_TYPES = (("markdown", "text"), ("tables", "table"), ("images_desc", "image"))

# Period mentions inside a single chunk.
_CHUNK_YEAR_RE = re.compile(r"\b(20[12]\d)\b")
_CHUNK_QUARTER_RE = re.compile(
    r"\b(Q[1-4])\b"
    r"|\b([1-4]T)\b"
    r"|\b(9M|6M|3M)\s*(?:20[12]\d)?"
    r"|\b(primer|segundo|tercer|cuarto)\s+trimestre\b",
    re.I,
)
_PERIOD_KEYWORDS = (
    ("9m", "9M"),
    ("6m", "6M"),
    ("3m", "3M"),
    ("primer", "Q1"),
    ("segundo", "Q2"),
    ("tercer", "Q3"),
    ("cuarto", "Q4"),
)


# ── Chunk metadata ────────────────────────────────────────────────────────────


def _normalize_period(marker: str) -> str:
    """Normalize a period marker (`q3`, `3T`, `9M 2025`, `tercer trimestre`) to `Q3`, `9M`, ..."""
    marker = marker.lower()
    if re.match(r"q[1-4]", marker):
        return marker.upper()
    if re.match(r"[1-4]t", marker):
        return f"Q{marker[0]}"
    return next(
        (period for keyword, period in _PERIOD_KEYWORDS if keyword in marker), marker.upper()
    )


def _annotate_chunk_period(chunk_text: str, base_metadata: dict[str, Any]) -> dict[str, Any]:
    """Override the document's period with the one the chunk itself mentions.

    A multi-period report (e.g. 9M2025 with FY2024 comparatives) would otherwise
    give every chunk the same fiscal_year and fiscal_quarter as the file.
    """
    meta = dict(base_metadata)
    years = _CHUNK_YEAR_RE.findall(chunk_text)
    if years:
        meta["fiscal_year"] = Counter(years).most_common(1)[0][0]
    quarter = _CHUNK_QUARTER_RE.search(chunk_text)
    if quarter:
        meta["fiscal_quarter"] = _normalize_period(quarter.group(0))
    return meta


def _classify(file_path: Path) -> tuple[str, str]:
    """Return (content type, document name) for a file under rag_data_dir."""
    path = str(file_path)
    for folder, content_type in _CONTENT_TYPES:
        if folder in path:
            doc_name = file_path.stem if content_type == "text" else file_path.parent.name
            return content_type, doc_name
    return "unknown", file_path.stem


def _base_metadata(file_path: Path, content_type: str, doc_name: str) -> dict[str, Any]:
    """Read the sidecar written at extraction time; fall back to the filename."""
    if content_type == "text":
        sidecar = file_path.with_suffix(".meta.json")
    else:
        sidecar = file_path.parent / "meta.json"
    if sidecar.exists():
        logger.debug("metadata_from_sidecar", file=file_path.name)
        sidecar_metadata: dict[str, Any] = json.loads(sidecar.read_text(encoding="utf-8"))
        return sidecar_metadata

    logger.debug("metadata_from_filename", file=file_path.name)
    metadata = extract_metadata_from_filename(doc_name)
    metadata["ticker"] = COMPANY_TICKER_MAP.get(metadata["company"], "UNKNOWN")
    return metadata


def _extract_page_number(file_path: Path) -> int | None:
    """Page number from filenames such as `page_28.md` or `table_1_page_5.md`."""
    match = re.search(r"page_(\d+)", file_path.stem)
    return int(match.group(1)) if match else None


def compute_file_hash(file_path: Path) -> str:
    """SHA-256 of a file, used to skip files that are already indexed."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(HASH_BLOCK_SIZE), b""):
            sha256.update(block)
    return sha256.hexdigest()


# ── Store ─────────────────────────────────────────────────────────────────────


class VectorStoreManager:
    """Qdrant hybrid vector store and the ingestion of extracted Markdown into it."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        overlap = min(settings.chunk_overlap, max(settings.chunk_size - 1, 0))
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        url = f"http://{settings.qdrant_host}:{settings.qdrant_port}"
        self.store = QdrantVectorStore.from_documents(
            documents=[],
            embedding=OllamaEmbeddings(
                model=settings.ollama_embed_model, base_url=settings.ollama_base_url
            ),
            sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
            collection_name=settings.qdrant_collection,
            url=url,
            retrieval_mode=RetrievalMode.HYBRID,
            force_recreate=False,
        )
        logger.info("qdrant_connected", url=url, collection=settings.qdrant_collection)

    @property
    def _collection(self) -> str:
        return self.settings.qdrant_collection

    def _scroll_metadata(self) -> Iterator[dict[str, Any]]:
        """Yield the metadata payload of every point in the collection."""
        offset = None
        while True:
            points, offset = self.store.client.scroll(
                collection_name=self._collection,
                limit=SCROLL_BATCH_SIZE,
                with_payload=True,
                offset=offset,
            )
            for point in points:
                yield (point.payload or {}).get("metadata") or {}
            if not points or offset is None:
                return

    def get_processed_hashes(self) -> set[str]:
        """Hashes of the files already ingested."""
        return {meta["file_hash"] for meta in self._scroll_metadata() if meta.get("file_hash")}

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def _split_text(self, text: str) -> list[str]:
        if not text.strip():
            return []
        return [chunk for chunk in self._text_splitter.split_text(text) if chunk.strip()]

    def _text_documents(self, content: str, base_metadata: dict[str, Any]) -> list[Document]:
        """Split Markdown per page, chunk each page and annotate each chunk's period."""
        started_at = time.perf_counter()
        documents = []
        for page, page_text in enumerate(content.split(PAGE_BREAK), start=1):
            for chunk_index, chunk_text in enumerate(self._split_text(page_text), start=1):
                metadata = _annotate_chunk_period(
                    chunk_text, {**base_metadata, "page": page, "chunk_index": chunk_index}
                )
                documents.append(Document(page_content=chunk_text, metadata=metadata))
                CHUNK_SIZE_BYTES.observe(len(chunk_text.encode("utf-8")))
        CHUNKING_DURATION.observe(time.perf_counter() - started_at)
        return documents

    def _ingest_content(self, file_path: Path, file_hash: str) -> int:
        """Ingest one .md file without the deduplication check. Returns the chunks created.

        Text files are chunked; table and image files become one document each.
        """
        content_type, doc_name = _classify(file_path)
        content = file_path.read_text(encoding="utf-8")
        base_metadata = {
            **_base_metadata(file_path, content_type, doc_name),
            "content_type": content_type,
            "file_hash": file_hash,
            "source_file": doc_name,
        }

        if content_type == "text":
            documents = self._text_documents(content, base_metadata)
            if documents:
                self.store.add_documents(documents)
            logger.info(
                "text_ingested",
                file=file_path.name,
                pages=content.count(PAGE_BREAK) + 1,
                chunks=len(documents),
            )
            return len(documents)

        metadata = {**base_metadata, "page": _extract_page_number(file_path)}
        self.store.add_documents([Document(page_content=content, metadata=metadata)])
        CHUNK_SIZE_BYTES.observe(len(content.encode("utf-8")))
        logger.info("doc_ingested", file=file_path.name, type=content_type)
        return 1

    def ingest_all_files(self, rag_data_dir: Path) -> int:
        """Ingest every new .md file under rag_data_dir in parallel. Returns the files ingested."""
        all_files = list(Path(rag_data_dir).rglob("*.md"))
        if not all_files:
            raise ValueError(f"No .md files found in {rag_data_dir}")

        processed_hashes = self.get_processed_hashes()
        lock = threading.Lock()
        logger.info(
            "ingestion_started", total_files=len(all_files), already_processed=len(processed_hashes)
        )

        def ingest_if_new(md_file: Path) -> int:
            file_hash = compute_file_hash(md_file)
            with lock:
                if file_hash in processed_hashes:
                    return 0
                # Claim the file before releasing the lock so no other thread ingests it.
                processed_hashes.add(file_hash)
            try:
                return self._ingest_content(md_file, file_hash)
            except Exception:
                logger.exception("file_ingestion_error", path=str(md_file))
                return 0

        with ThreadPoolExecutor(max_workers=INGEST_WORKERS, thread_name_prefix="ingest") as pool:
            chunks_per_file = list(pool.map(ingest_if_new, all_files))

        chunks_created = sum(chunks_per_file)
        new_files = sum(1 for chunks in chunks_per_file if chunks > 0)
        CHUNKS_CREATED.set(chunks_created)
        logger.info("ingestion_finished", new_files=new_files, chunks_created=chunks_created)
        return new_files

    def ensure_payload_indexes(self) -> None:
        """Create keyword payload indexes on the filter fields for fast filtered search."""
        for field_name in FILTERABLE_FIELDS:
            try:
                self.store.client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.info("payload_index_created", field=field_name)
            except Exception as exc:
                # The index already exists, or the collection is empty.
                logger.debug("payload_index_skip", field=field_name, reason=str(exc))

    # ── Statistics and health ─────────────────────────────────────────────────

    def collection_count(self) -> int:
        """Number of points in the collection (0 if Qdrant is unreachable)."""
        try:
            info = self.store.client.get_collection(self._collection)
        except Exception as exc:
            logger.warning("collection_count_failed", error=str(exc))
            return 0
        return info.points_count or 0

    def get_metadata_stats(self) -> dict[str, Any]:
        """Distinct companies and (file, page) pairs indexed (zeros if Qdrant is unreachable)."""
        companies: set[str | None] = set()
        pages: set[tuple[str | None, int | None]] = set()
        try:
            for meta in self._scroll_metadata():
                companies.add(meta.get("company"))
                pages.add((meta.get("source_file"), meta.get("page")))
        except Exception as exc:
            logger.warning("metadata_stats_failed", error=str(exc))
            return {"companies": 0, "pages": 0}
        return {"companies": len(companies - {None}), "pages": len(pages - {(None, None)})}

    def health_check(self) -> bool:
        """True if Qdrant answers."""
        try:
            self.store.client.get_collections()
        except Exception as exc:
            logger.warning("qdrant_health_check_failed", error=str(exc))
            return False
        return True
