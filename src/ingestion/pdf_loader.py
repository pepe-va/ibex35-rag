"""PDF extraction with Docling: markdown, images and tables for IBEX35 reports."""

from pathlib import Path
from typing import List, Tuple

from docling_core.types.doc import PictureItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from langchain_core.documents import Document

from src.logging_config import get_logger

logger = get_logger(__name__)

# ── IBEX35 company → ticker map ───────────────────────────────────────────────

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

_PAGE_BREAK = "<!-- page break -->"


# ── Metadata extraction ───────────────────────────────────────────────────────

def extract_metadata_from_filename(filename: str) -> dict:
    """
    Extract metadata from filename.

    Supported formats:
      - "IBERDROLA.pdf"               → single-word IBEX35 company
      - "IBERDROLA 10-K 2024.pdf"     → company + doc_type + year
      - "IBERDROLA 10-Q Q1 2024.pdf"  → company + doc_type + quarter + year
    """
    name = filename.replace(".pdf", "").replace(".md", "")
    parts = name.split()

    if len(parts) == 4:
        return {
            "company": parts[0].upper(),
            "doc_type": parts[1].lower(),
            "fiscal_quarter": parts[2].lower(),
            "fiscal_year": parts[3],
        }
    elif len(parts) == 3:
        return {
            "company": parts[0].upper(),
            "doc_type": parts[1].lower(),
            "fiscal_quarter": None,
            "fiscal_year": parts[2],
        }
    else:
        return {
            "company": parts[0].upper(),
            "doc_type": "financial_results",
            "fiscal_quarter": None,
            "fiscal_year": None,
        }


# ── Docling conversion ────────────────────────────────────────────────────────

def convert_pdf_to_docling(pdf_file: Path):
    """Convert PDF with Docling (text + image extraction enabled)."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_page_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    return doc_converter.convert(pdf_file)


# ── Image extraction ──────────────────────────────────────────────────────────

def save_page_images(doc_result, images_dir: Path) -> None:
    """Save pages that contain large images (>500×500 px) as PNG files."""
    pages_to_save: set = set()

    for item in doc_result.document.iterate_items():
        element = item[0]
        if isinstance(element, PictureItem):
            image = element.get_image(doc_result.document)
            if image and image.size[0] > 500 and image.size[1] > 500:
                page_no = element.prov[0].page_no if element.prov else None
                if page_no:
                    pages_to_save.add(page_no)

    for page_no in pages_to_save:
        page = doc_result.document.pages[page_no]
        if page.image and page.image.pil_image:
            page.image.pil_image.save(images_dir / f"page_{page_no}.png", "PNG")
            logger.info("image_saved", page=page_no, dir=str(images_dir))


# ── Table extraction ──────────────────────────────────────────────────────────

def extract_context_and_table(lines: List[str], table_index: int) -> Tuple[str, int]:
    """
    Extract a markdown table starting at table_index plus the 2 preceding lines as context.

    Returns:
        (combined_content, next_line_index)
    """
    table_lines = []
    i = table_index

    while i < len(lines) and lines[i].startswith("|"):
        table_lines.append(lines[i])
        i += 1

    start = max(0, table_index - 2)
    context_lines = lines[start:table_index]
    content = "\n".join(context_lines) + "\n\n" + "\n".join(table_lines)

    return content, i


def extract_tables_with_context(markdown_text: str) -> List[Tuple[str, str, int]]:
    """
    Find all tables in the markdown and return them with context and page number.

    Returns:
        List of (content, table_name, page_number)
    """
    lines = [line for line in markdown_text.split("\n") if line.strip()]
    tables = []
    current_page = 1
    table_num = 1
    i = 0

    while i < len(lines):
        if _PAGE_BREAK in lines[i]:
            current_page += 1
            i += 1
            continue

        if lines[i].startswith("|") and lines[i].count("|") > 1:
            content, next_i = extract_context_and_table(lines, i)
            tables.append((content, f"table_{table_num}", current_page))
            table_num += 1
            i = next_i
        else:
            i += 1

    return tables


def save_tables(markdown_text: str, tables_dir: Path) -> None:
    """Save each extracted table (with context and page number) to a separate .md file."""
    tables = extract_tables_with_context(markdown_text)
    for table_content, table_name, page_num in tables:
        content_with_page = f"**Page:** {page_num}\n\n{table_content}"
        (tables_dir / f"{table_name}_page_{page_num}.md").write_text(
            content_with_page, encoding="utf-8"
        )
    if tables:
        logger.info("tables_saved", count=len(tables), dir=str(tables_dir))


# ── Main extraction function ──────────────────────────────────────────────────

def extract_pdf_content(pdf_file: Path, rag_data_dir: Path) -> Path:
    """
    Extract a PDF into markdown, images and tables under rag_data_dir.

    Output structure:
        {rag_data_dir}/markdown/{company}/{stem}.md
        {rag_data_dir}/images/{company}/{stem}/page_{n}.png
        {rag_data_dir}/tables/{company}/{stem}/table_{n}_page_{m}.md

    Returns:
        Path to the saved markdown file.
    """
    metadata = extract_metadata_from_filename(pdf_file.stem)
    company = metadata["company"]

    md_dir = rag_data_dir / "markdown" / company
    images_dir = rag_data_dir / "images" / company / pdf_file.stem
    tables_dir = rag_data_dir / "tables" / company / pdf_file.stem

    for dir_path in [md_dir, images_dir, tables_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    logger.info("extracting_pdf", file=pdf_file.name, company=company)

    doc_result = convert_pdf_to_docling(pdf_file)
    markdown_text = doc_result.document.export_to_markdown(
        page_break_placeholder=_PAGE_BREAK
    )

    md_path = md_dir / f"{pdf_file.stem}.md"
    md_path.write_text(markdown_text, encoding="utf-8")

    save_page_images(doc_result, images_dir)
    save_tables(markdown_text, tables_dir)

    logger.info("pdf_extracted", file=pdf_file.name, markdown=str(md_path))
    return md_path


def load_pdf_as_documents(pdf_file: Path) -> list[Document]:
    """
    Convert a single PDF directly to LangChain Documents (one per non-empty page).

    Uses Docling for extraction. Each page becomes a Document with metadata
    extracted from the filename (company, doc_type, fiscal_year, fiscal_quarter).
    """
    pdf_file = Path(pdf_file)
    metadata = extract_metadata_from_filename(pdf_file.stem)
    metadata.update(
        {
            "content_type": "text",
            "ticker": COMPANY_TICKER_MAP.get(metadata["company"], "UNKNOWN"),
            "source_file": pdf_file.stem,
        }
    )

    doc_result = convert_pdf_to_docling(pdf_file)
    markdown_text = doc_result.document.export_to_markdown(
        page_break_placeholder=_PAGE_BREAK
    )

    documents = []
    for idx, page in enumerate(markdown_text.split(_PAGE_BREAK), start=1):
        if page.strip():
            documents.append(
                Document(page_content=page, metadata={**metadata, "page": idx})
            )

    return documents


def load_all_pdfs(pdf_dir: Path) -> list[Path]:
    """
    List all PDF files in pdf_dir.

    Raises FileNotFoundError if directory does not exist.
    Raises ValueError if no PDFs are found.
    """
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = sorted(
        p for p in pdf_dir.glob("*.pdf") if not p.name.endswith(":Zone.Identifier")
    )
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")

    return pdf_files


def extract_all_pdfs(pdf_dir: Path, rag_data_dir: Path) -> list[Path]:
    """Extract all PDFs in pdf_dir and return the list of markdown files produced."""
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = sorted(
        p for p in pdf_dir.glob("*.pdf") if not p.name.endswith(":Zone.Identifier")
    )
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")

    logger.info("extracting_all_pdfs", count=len(pdf_files), dir=str(pdf_dir))
    produced: list[Path] = []
    for pdf_path in pdf_files:
        try:
            md_path = extract_pdf_content(pdf_path, rag_data_dir)
            produced.append(md_path)
        except Exception:
            logger.exception("pdf_extraction_error", path=str(pdf_path))

    return produced
