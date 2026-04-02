#!/usr/bin/env python3
"""CLI script to ingest IBEX35 PDFs into the vector store."""

import sys
import time
from pathlib import Path

# Allow running from repo root: python scripts/ingest.py
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from src.config import get_settings
from src.ingestion.pipeline import IngestionPipeline
from src.logging_config import setup_logging
from src.vectorstore.store import VectorStoreManager


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest IBEX35 financial PDFs into ChromaDB vector store."
    )
    parser.add_argument(
        "--pdf-dir",
        default=None,
        help="Path to PDF directory (default: from settings)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(args.log_level)

    print("\n" + "=" * 60)
    print("  IBEX35 RAG — Ingestion Pipeline")
    print("=" * 60)
    print(f"  PDF dir  : {args.pdf_dir or settings.pdf_dir}")
    print(f"  Embed    : {settings.ollama_embed_model}")
    print(f"  Chunk    : {settings.chunk_size} tokens (overlap {settings.chunk_overlap})")
    print(f"  Chroma   : {settings.chroma_host}:{settings.chroma_port}/{settings.chroma_collection}")
    print("=" * 60 + "\n")

    vs = VectorStoreManager(settings)
    pipeline = IngestionPipeline(settings=settings, vector_store=vs)

    start = time.perf_counter()
    try:
        result = pipeline.run(pdf_dir=args.pdf_dir)
    except Exception as exc:
        print(f"\n❌  Ingestion FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.perf_counter() - start

    print("\n" + "=" * 60)
    if result.success:
        print("  ✅  Ingestion complete!")
    else:
        print("  ⚠️  Ingestion finished with errors")
    print(f"  Companies  : {result.total_companies}")
    print(f"  Documents  : {result.total_documents} pages")
    print(f"  Nodes      : {result.total_nodes} chunks")
    print(f"  Duration   : {elapsed:.1f}s")
    if result.errors:
        print(f"  Errors     : {len(result.errors)}")
        for e in result.errors:
            print(f"    - {e}")
    print("=" * 60 + "\n")

    if not result.success:
        sys.exit(1)


if __name__ == "__main__":
    main()
