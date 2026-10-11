"""PDF extraction with Docling: Markdown, tables and page images for IBEX35 reports."""

import json
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument, PictureItem
from langchain_core.documents import Document

from src.core.logging import get_logger
from src.ingestion.metadata import PAGE_BREAK, build_metadata, extract_metadata_from_filename

logger = get_logger(__name__)

IMAGE_SCALE = 2
# A page is saved as an image when it holds a picture larger than this on both sides.
MIN_IMAGE_SIDE_PX = 500
# Lines of text kept before each table as its context.
TABLE_CONTEXT_LINES = 2


# ── Docling conversion ────────────────────────────────────────────────────────


def convert_pdf_to_docling(pdf_file: Path) -> ConversionResult:
    """Convert a PDF with Docling, with page and picture images enabled."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)
    pipeline_options.images_scale = IMAGE_SCALE
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_page_images = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    return converter.convert(pdf_file)


def _export_markdown(doc_result: ConversionResult) -> str:
    return str(doc_result.document.export_to_markdown(page_break_placeholder=PAGE_BREAK))


# ── Images ────────────────────────────────────────────────────────────────────


def _large_picture_pages(document: DoclingDocument) -> set[int]:
    """Pages that contain at least one picture larger than MIN_IMAGE_SIDE_PX."""
    pages: set[int] = set()
    for element, _level in document.iterate_items():
        if not isinstance(element, PictureItem) or not element.prov:
            continue
        image = element.get_image(document)
        if image and min(image.size) > MIN_IMAGE_SIDE_PX:
            pages.add(element.prov[0].page_no)
    return pages


def save_page_images(doc_result: ConversionResult, images_dir: Path) -> None:
    """Save the pages that contain large pictures as PNG files."""
    for page_no in _large_picture_pages(doc_result.document):
        page = doc_result.document.pages[page_no]
        if page.image and page.image.pil_image:
            page.image.pil_image.save(images_dir / f"page_{page_no}.png", "PNG")
            logger.info("image_saved", page=page_no, dir=str(images_dir))


# ── Tables ────────────────────────────────────────────────────────────────────


def extract_context_and_table(lines: list[str], table_index: int) -> tuple[str, int]:
    """Return the table starting at `table_index` with its preceding context lines.

    Returns:
        (content, index of the first line after the table)
    """
    end = table_index
    while end < len(lines) and lines[end].startswith("|"):
        end += 1

    context = lines[max(0, table_index - TABLE_CONTEXT_LINES) : table_index]
    content = "\n".join(context) + "\n\n" + "\n".join(lines[table_index:end])
    return content, end


def extract_tables_with_context(markdown_text: str) -> list[tuple[str, str, int]]:
    """Find every Markdown table with its context and page number.

    Returns:
        A list of (content, table name, page number).
    """
    lines = [line for line in markdown_text.split("\n") if line.strip()]
    tables: list[tuple[str, str, int]] = []
    current_page = 1
    i = 0
    while i < len(lines):
        if PAGE_BREAK in lines[i]:
            current_page += 1
            i += 1
        elif lines[i].startswith("|") and lines[i].count("|") > 1:
            content, i = extract_context_and_table(lines, i)
            tables.append((content, f"table_{len(tables) + 1}", current_page))
        else:
            i += 1
    return tables


def save_tables(markdown_text: str, tables_dir: Path) -> None:
    """Save each table, with its context and page number, to its own .md file."""
    tables = extract_tables_with_context(markdown_text)
    for content, table_name, page in tables:
        (tables_dir / f"{table_name}_page_{page}.md").write_text(
            f"**Page:** {page}\n\n{content}", encoding="utf-8"
        )
    if tables:
        logger.info("tables_saved", count=len(tables), dir=str(tables_dir))


# ── Extraction ────────────────────────────────────────────────────────────────


def extract_pdf_content(pdf_file: Path, rag_data_dir: Path) -> Path:
    """Extract a PDF into Markdown, tables and page images under rag_data_dir.

    Output structure:
        {rag_data_dir}/markdown/{company}/{stem}.md   (+ {stem}.meta.json)
        {rag_data_dir}/images/{company}/{stem}/page_{n}.png   (+ meta.json)
        {rag_data_dir}/tables/{company}/{stem}/table_{n}_page_{m}.md   (+ meta.json)

    Returns:
        Path to the saved Markdown file.
    """
    company: str = extract_metadata_from_filename(pdf_file.stem)["company"]
    md_dir = rag_data_dir / "markdown" / company
    images_dir = rag_data_dir / "images" / company / pdf_file.stem
    tables_dir = rag_data_dir / "tables" / company / pdf_file.stem
    for directory in (md_dir, images_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    logger.info("extracting_pdf", file=pdf_file.name, company=company)
    doc_result = convert_pdf_to_docling(pdf_file)
    markdown_text = _export_markdown(doc_result)
    md_path = md_dir / f"{pdf_file.stem}.md"
    md_path.write_text(markdown_text, encoding="utf-8")

    # Sidecars: next to the Markdown for text ingestion, and in the table and
    # image folders for their ingestion.
    metadata = build_metadata(pdf_file.stem, markdown_text)
    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)
    for sidecar in (
        md_dir / f"{pdf_file.stem}.meta.json",
        tables_dir / "meta.json",
        images_dir / "meta.json",
    ):
        sidecar.write_text(metadata_json, encoding="utf-8")
    logger.info(
        "metadata_extracted",
        file=pdf_file.name,
        company=metadata.get("company"),
        year=metadata.get("fiscal_year"),
        doc_type=metadata.get("doc_type"),
    )

    save_page_images(doc_result, images_dir)
    save_tables(markdown_text, tables_dir)
    logger.info("pdf_extracted", file=pdf_file.name, markdown=str(md_path))
    return md_path


def _list_pdfs(pdf_dir: str | Path) -> list[Path]:
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    pdf_files = sorted(p for p in pdf_dir.glob("*.pdf") if not p.name.endswith(":Zone.Identifier"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_dir}")
    return pdf_files


def extract_all_pdfs(pdf_dir: str | Path, rag_data_dir: str | Path) -> list[Path]:
    """Extract every PDF in pdf_dir and return the Markdown files produced.

    A PDF that fails is logged and skipped.
    """
    pdf_files = _list_pdfs(pdf_dir)
    logger.info("extracting_all_pdfs", count=len(pdf_files), dir=str(pdf_dir))
    produced: list[Path] = []
    for pdf_path in pdf_files:
        try:
            produced.append(extract_pdf_content(pdf_path, Path(rag_data_dir)))
        except Exception:
            logger.exception("pdf_extraction_error", path=str(pdf_path))
    return produced


# ── In-memory loading (no files written) ──────────────────────────────────────


def load_pdf_as_documents(pdf_file: str | Path) -> list[Document]:
    """Convert a PDF with Docling and return one Document per non-empty page."""
    pdf_file = Path(pdf_file)
    markdown_text = _export_markdown(convert_pdf_to_docling(pdf_file))
    metadata = {**build_metadata(pdf_file.stem, markdown_text), "content_type": "text"}
    return [
        Document(page_content=page.strip(), metadata={**metadata, "page": number})
        for number, page in enumerate(markdown_text.split(PAGE_BREAK), start=1)
        if page.strip()
    ]


def load_all_pdfs(pdf_dir: str | Path) -> list[Document]:
    """Load every PDF in pdf_dir as page Documents. A PDF that fails is logged and skipped."""
    documents: list[Document] = []
    for pdf_path in _list_pdfs(pdf_dir):
        try:
            documents.extend(load_pdf_as_documents(pdf_path))
        except Exception:
            logger.exception("pdf_load_error", path=str(pdf_path))
    return documents
