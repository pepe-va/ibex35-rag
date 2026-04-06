"""Unit tests for PDF loader."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.pdf_loader import (
    COMPANY_TICKER_MAP,
    load_pdf_as_documents,
    load_all_pdfs,
)


def test_company_ticker_map_completeness():
    """All major IBEX35 companies should have a ticker mapping."""
    required = ["IBERDROLA", "SANTANDER", "INDITEX", "BBVA", "TELEFONICA"]
    for company in required:
        assert company in COMPANY_TICKER_MAP, f"{company} missing from ticker map"
        assert COMPANY_TICKER_MAP[company].endswith(".MC"), f"{company} ticker should end with .MC"


def test_load_all_pdfs_missing_dir():
    """Should raise FileNotFoundError for non-existent directory."""
    with pytest.raises(FileNotFoundError):
        load_all_pdfs("/non/existent/path")


def test_load_all_pdfs_empty_dir(tmp_path):
    """Should raise ValueError when no PDFs found."""
    with pytest.raises(ValueError, match="No PDF files found"):
        load_all_pdfs(tmp_path)


@patch("src.ingestion.pdf_loader.DocumentConverter")
def test_load_pdf_as_documents(mock_converter_cls, tmp_path):
    """Should return one Document per non-empty page."""
    mock_result = MagicMock()
    mock_result.document.export_to_markdown.return_value = (
        "Revenue: €10B EBITDA: €3B\n<!-- page break -->\nNet profit: €500M"
    )
    mock_converter_cls.return_value.convert.return_value = mock_result

    pdf_path = tmp_path / "IBERDROLA.pdf"
    pdf_path.touch()

    docs = load_pdf_as_documents(pdf_path)

    assert len(docs) == 2
    assert docs[0].metadata["company"] == "IBERDROLA"
    assert docs[0].metadata["ticker"] == "IBE.MC"
    assert docs[0].metadata["doc_type"] == "financial_results"
    assert docs[0].metadata["content_type"] == "text"


@patch("src.ingestion.pdf_loader.DocumentConverter")
def test_load_pdf_skips_empty_pages(mock_converter_cls, tmp_path):
    """Pages with only whitespace should be skipped."""
    mock_result = MagicMock()
    mock_result.document.export_to_markdown.return_value = (
        "   \n  \n<!-- page break -->\nNet profit: €500M"
    )
    mock_converter_cls.return_value.convert.return_value = mock_result

    pdf_path = tmp_path / "SANTANDER.pdf"
    pdf_path.touch()

    docs = load_pdf_as_documents(pdf_path)
    assert len(docs) == 1
