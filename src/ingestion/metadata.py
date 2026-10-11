"""Report metadata: company tickers, and company/period/type detection from text."""

import re
from typing import Any

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
    "REPSOL": "REP.MC",
    "ROVI": "ROVI.MC",
    "SACYR": "SCYR.MC",
    "SANTANDER": "SAN.MC",
    "SOLARIA": "SLR.MC",
    "TELEFONICA": "TEF.MC",
    "UNICAJA": "UNI.MC",
}

# Docling page-break marker kept in the exported Markdown
PAGE_BREAK = "<!-- page break -->"


# ── Metadata extraction ───────────────────────────────────────────────────────


def extract_metadata_from_filename(filename: str) -> dict[str, Any]:
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
    if len(parts) == 3:
        return {
            "company": parts[0].upper(),
            "doc_type": parts[1].lower(),
            "fiscal_quarter": None,
            "fiscal_year": parts[2],
        }
    return {
        "company": parts[0].upper(),
        "doc_type": "financial_results",
        "fiscal_quarter": None,
        "fiscal_year": None,
    }


# ── Content-based metadata extraction ────────────────────────────────────────

# Patterns ordered by specificity (most specific first)
_DOC_TYPE_PATTERNS: list[tuple[str, str]] = [
    (r"10-K", "10-k"),
    (r"10-Q", "10-q"),
    (r"annual\s+report|informe\s+anual", "annual_report"),
    (r"resultados\s+(?:del\s+)?(?:ejercicio|anuales)", "annual_results"),
    (r"resultados\s+(?:del\s+)?(?:primer|segundo|tercer|cuarto)\s+trimestre", "quarterly_results"),
    (r"quarterly\s+results?|interim\s+report", "quarterly_results"),
    (r"resultados", "financial_results"),
]

_QUARTER_PATTERNS: list[tuple[str, str]] = [
    (r"\bQ([1-4])\b", r"q\1"),
    (r"\b([1-4])T\b", r"q\1"),
    (r"primer\s+trimestre|first\s+quarter|1H\b", "q1"),
    (r"segundo\s+trimestre|second\s+quarter", "q2"),
    (r"tercer\s+trimestre|third\s+quarter", "q3"),
    (r"cuarto\s+trimestre|fourth\s+quarter|2H\b", "q4"),
]


def extract_metadata_from_content(markdown_text: str) -> dict[str, Any]:
    """
    Extract metadata from the first 200 lines of a Docling-generated markdown.

    Returns a partial dict with only the keys that could be detected
    (company, fiscal_year, fiscal_quarter, doc_type). Missing keys are omitted
    so the caller can distinguish "not found" from an explicit None.
    """
    head = "\n".join(markdown_text.split("\n")[:200])
    head_upper = head.upper()
    result: dict[str, Any] = {}

    # Company — match against known IBEX35 ticker map
    for company in COMPANY_TICKER_MAP:
        if company in head_upper:
            result["company"] = company
            break

    # Fiscal year — prefer the first 4-digit year in 20XX range
    year_match = re.search(r"\b(20[12]\d)\b", head)
    if year_match:
        result["fiscal_year"] = year_match.group(1)

    # Quarter
    for pattern, replacement in _QUARTER_PATTERNS:
        m = re.search(pattern, head, re.IGNORECASE)
        if m:
            # replacement may contain back-references (e.g. r"q\1") or be a plain string
            result["fiscal_quarter"] = m.expand(replacement) if "\\" in replacement else replacement
            break

    # Doc type
    for pattern, doc_type in _DOC_TYPE_PATTERNS:
        if re.search(pattern, head, re.IGNORECASE):
            result["doc_type"] = doc_type
            break

    return result


def merge_metadata(from_filename: dict[str, Any], from_content: dict[str, Any]) -> dict[str, Any]:
    """
    Merge filename-derived and content-derived metadata dicts.

    Content wins for any key where it has a non-None value.
    Filename values serve as fallback.
    """
    return {
        **from_filename,
        **{key: value for key, value in from_content.items() if value is not None},
    }


def build_metadata(stem: str, markdown_text: str) -> dict[str, Any]:
    """Metadata from the document content, with the filename as fallback, plus the ticker."""
    metadata = merge_metadata(
        extract_metadata_from_filename(stem), extract_metadata_from_content(markdown_text)
    )
    metadata["ticker"] = COMPANY_TICKER_MAP.get(metadata.get("company", ""), "UNKNOWN")
    return metadata
