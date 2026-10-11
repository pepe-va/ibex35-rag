#!/usr/bin/env python3
"""CLI for ingestion phase 1: extract PDFs to Markdown, tables and images with Docling.

Usage:
    uv run python src/ingestion/preprocessor.py --pdf-dir ./data/raw/pdfs
    make extract
"""

import argparse
import sys
import time

from src.core.config import get_settings
from src.core.logging import setup_logging
from src.ingestion.pdf_loader import extract_all_pdfs

SEP = "=" * 60


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extraer PDFs IBEX35 a Markdown/tablas/imágenes con Docling."
    )
    parser.add_argument(
        "--pdf-dir",
        default=None,
        help="Directorio con los PDFs (por defecto: settings.pdf_dir)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(args.log_level)

    pdf_dir = args.pdf_dir or settings.pdf_dir

    print("\n" + SEP)
    print("  IBEX35 RAG — Preprocesado (Docling)")
    print(SEP)
    print(f"  PDF dir  : {pdf_dir}")
    print(f"  RAG dir  : {settings.rag_data_dir}")
    print(SEP + "\n")

    start = time.perf_counter()

    try:
        produced = extract_all_pdfs(pdf_dir, settings.rag_data_dir)
    except Exception as exc:
        print(f"\n❌  Extracción FALLIDA: {exc}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.perf_counter() - start
    companies = sorted({md.parent.name for md in produced})

    print("\n" + SEP)
    print("  ✅  Extracción completada")
    print(f"  PDFs/MDs   : {len(produced)}")
    print(f"  Empresas   : {len(companies)}")
    print(f"  Duración   : {elapsed:.1f}s")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
