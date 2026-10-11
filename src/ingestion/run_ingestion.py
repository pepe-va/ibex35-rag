#!/usr/bin/env python3
"""CLI for the full ingestion pipeline: Docling extraction, chunking and Qdrant indexing.

Usage:
    uv run python src/ingestion/run_ingestion.py --pdf-dir ./data/raw/pdfs
    make ingest
"""

import argparse
import sys
import time

from src.core.config import get_settings
from src.core.logging import setup_logging
from src.ingestion.pipeline import IngestionPipeline
from src.rag.vector_store.store import VectorStoreManager

SEP = "=" * 60


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingestar Markdown IBEX35 en Qdrant (chunking + indexación híbrida)."
    )
    parser.add_argument(
        "--pdf-dir",
        default=None,
        help="Directorio raíz con los datos (por defecto: settings.pdf_dir)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(args.log_level)

    print("\n" + SEP)
    print("  IBEX35 RAG — Ingesta (chunking + Qdrant)")
    print(SEP)
    print(f"  RAG dir  : {settings.rag_data_dir}")
    print(f"  Embed    : {settings.ollama_embed_model}")
    print(f"  Chunk    : {settings.chunk_size} (overlap {settings.chunk_overlap})")
    print(
        f"  Qdrant   : {settings.qdrant_host}:{settings.qdrant_port}/{settings.qdrant_collection}"
    )
    print(SEP + "\n")

    start = time.perf_counter()

    vs = VectorStoreManager(settings)
    pipeline = IngestionPipeline(settings=settings, vector_store=vs)
    try:
        result = pipeline.run(pdf_dir=args.pdf_dir)
    except Exception as exc:
        print(f"\n❌  Ingesta FALLIDA: {exc}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.perf_counter() - start

    print("\n" + SEP)
    if not result.errors:
        print("  ✅  Ingesta completada")
    else:
        print("  ⚠️  Ingesta finalizada con errores")
    print(f"  PDFs       : {result.total_pdfs}")
    print(f"  Empresas   : {result.total_companies}")
    print(f"  Nuevos     : {result.total_ingested}")
    print(f"  Qdrant pts : {vs.collection_count()}")
    print(f"  Duración   : {elapsed:.1f}s")
    if result.errors:
        print(f"  Errores    : {len(result.errors)}")
        for e in result.errors:
            print(f"    - {e}")
    print(SEP + "\n")

    if result.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
