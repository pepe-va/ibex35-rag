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


@patch("src.ingestion.pdf_loader.fitz.open")
def test_load_pdf_as_documents(mock_fitz, tmp_path):
    """Should return one Document per non-empty page."""
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Revenue: €10B EBITDA: €3B"
    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page, mock_page]))
    mock_doc.__len__ = MagicMock(return_value=2)
    mock_doc.__enter__ = MagicMock(return_value=mock_doc)
    mock_doc.__exit__ = MagicMock(return_value=False)
    mock_fitz.return_value = mock_doc

    pdf_path = tmp_path / "IBERDROLA.pdf"
    pdf_path.touch()

    docs = load_pdf_as_documents(pdf_path)

    assert len(docs) == 2
    assert docs[0].metadata["company"] == "IBERDROLA"
    assert docs[0].metadata["ticker"] == "IBE.MC"
    assert docs[0].metadata["doc_type"] == "financial_results"


@patch("src.ingestion.pdf_loader.fitz.open")
def test_load_pdf_skips_empty_pages(mock_fitz, tmp_path):
    """Pages with only whitespace should be skipped."""
    mock_page_empty = MagicMock()
    mock_page_empty.get_text.return_value = "   \n  "
    mock_page_content = MagicMock()
    mock_page_content.get_text.return_value = "Net profit: €500M"

    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page_empty, mock_page_content]))
    mock_doc.__len__ = MagicMock(return_value=2)
    mock_doc.__enter__ = MagicMock(return_value=mock_doc)
    mock_doc.__exit__ = MagicMock(return_value=False)
    mock_fitz.return_value = mock_doc

    pdf_path = tmp_path / "SANTANDER.pdf"
    pdf_path.touch()

    docs = load_pdf_as_documents(pdf_path)
    assert len(docs) == 1
