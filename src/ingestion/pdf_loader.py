"""PDF loader with metadata extraction for IBEX35 financial reports."""

from pathlib import Path

import fitz  # PyMuPDF
from llama_index.core import Document

from src.logging_config import get_logger

logger = get_logger(__name__)

# Map known tickers for IBEX35 companies
COMPANY_TICKER_MAP: dict[str, str] = {
    "ACCIONA": "ANA.MC",
    "ACERINOX": "ACX.MC",
    "ACS": "ACS.MC",
    "AENA": "AENA.MC",
    "AMADEUS": "AMS.MC",
    "ARCELORMITTAL": "MTS.MC",
    "BBVA": "BBVA.MC",
    "CELLNEX": "CLNX.MC",
    "ENAGAS": "ENG.MC",
    "ENDESA": "ELE.MC",
    "FERROVIAL": "FER.MC",
    "FLUIDRA": "FLDR.MC",
    "GRIFOLS": "GRF.MC",
    "IAG": "IAG.MC",
    "IBERDROLA": "IBE.MC",
    "INDITEX": "ITX.MC",
    "INDRA": "IDR.MC",
    "LOGISTA": "LOG.MC",
    "MAPFRE": "MAP.MC",
    "MERLINPROPERTIES": "MRL.MC",
    "NATURGY": "NTGY.MC",
    "PUIG": "PUIG.MC",
    "REDEIA": "REE.MC",
    "ROVI": "ROVI.MC",
    "SACYR": "SCYR.MC",
    "SANTANDER": "SAN.MC",
    "SOLARIA": "SLR.MC",
    "TELEFONICA": "TEF.MC",
    "UNICAJA": "UNI.MC",
}


def _extract_text_by_page(pdf_path: Path) -> list[dict[str, str | int]]:
    """Extract text from each page preserving structure."""
    pages = []
    with fitz.open(str(pdf_path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append({"page": page_num, "text": text, "total_pages": len(doc)})
    return pages


def load_pdf_as_documents(pdf_path: Path) -> list[Document]:
    """Load a single PDF and return a list of LlamaIndex Documents (one per page)."""
    company_name = pdf_path.stem.upper()
    ticker = COMPANY_TICKER_MAP.get(company_name, "UNKNOWN")

    logger.info("loading_pdf", path=str(pdf_path), company=company_name, ticker=ticker)

    pages = _extract_text_by_page(pdf_path)
    documents = []

    for page_data in pages:
        doc = Document(
            text=str(page_data["text"]),
            metadata={
                "company": company_name,
                "ticker": ticker,
                "source": pdf_path.name,
                "page": page_data["page"],
                "total_pages": page_data["total_pages"],
                "doc_type": "financial_results",
                "index": "ibex35",
            },
            # Exclude from LLM context to save tokens, keep for filtering
            excluded_llm_metadata_keys=["page", "total_pages", "source"],
        )
        documents.append(doc)

    logger.info("pdf_loaded", company=company_name, pages=len(documents))
    return documents


def load_all_pdfs(pdf_dir: str | Path) -> list[Document]:
    """Load all PDFs from a directory."""
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = sorted(p for p in pdf_dir.glob("*.pdf") if not p.name.endswith(":Zone.Identifier"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")

    logger.info("loading_all_pdfs", directory=str(pdf_dir), count=len(pdf_files))

    all_documents: list[Document] = []
    for pdf_path in pdf_files:
        try:
            docs = load_pdf_as_documents(pdf_path)
            all_documents.extend(docs)
        except Exception:
            logger.exception("pdf_load_error", path=str(pdf_path))

    logger.info("all_pdfs_loaded", total_documents=len(all_documents), total_files=len(pdf_files))
    return all_documents
