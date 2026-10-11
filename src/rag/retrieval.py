"""Retrieval helpers: company detection, query intent, filters and context ordering.

Pure functions and constants used by RAGEngine.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

# ── IBEX35 company alias map (regex → canonical name) ────────────────────────
# Keep specific patterns before generic ones to avoid false positives with
# short aliases (e.g. \bree\b).

_COMPANY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\biberdrola\b", re.I), "IBERDROLA"),
    (re.compile(r"\bsantander\b", re.I), "SANTANDER"),
    (re.compile(r"\binditex\b|\bzara\b", re.I), "INDITEX"),
    (re.compile(r"\bbbva\b", re.I), "BBVA"),
    (re.compile(r"\btelef[oó]nica\b", re.I), "TELEFONICA"),
    (re.compile(r"\bacciona\b", re.I), "ACCIONA"),
    (re.compile(r"\bacerinox\b", re.I), "ACERINOX"),
    (re.compile(r"\bacs\b", re.I), "ACS"),
    (re.compile(r"\baena\b", re.I), "AENA"),
    (re.compile(r"\bamadeus\b", re.I), "AMADEUS"),
    (re.compile(r"\barcelor\b|\barcelormittal\b", re.I), "ARCELORMITTAL"),
    (re.compile(r"\bcellnex\b", re.I), "CELLNEX"),
    (re.compile(r"\benag[aá]s\b", re.I), "ENAGAS"),
    (re.compile(r"\bendesa\b", re.I), "ENDESA"),
    (re.compile(r"\bferrovial\b", re.I), "FERROVIAL"),
    (re.compile(r"\bfluidra\b", re.I), "FLUIDRA"),
    (re.compile(r"\bgrifols\b", re.I), "GRIFOLS"),
    (re.compile(r"\biag\b|\biberia\b", re.I), "IAG"),
    (re.compile(r"\bindra\b", re.I), "INDRA"),
    (re.compile(r"\blogista\b", re.I), "LOGISTA"),
    (re.compile(r"\bmapfre\b", re.I), "MAPFRE"),
    (re.compile(r"\bmerlin\b", re.I), "MERLINPROPERTIES"),
    (re.compile(r"\bnaturgy\b", re.I), "NATURGY"),
    (re.compile(r"\bpuig\b", re.I), "PUIG"),
    (re.compile(r"\bredeia\b|\bree\b|\bred el[eé]ctrica\b", re.I), "REDEIA"),
    (re.compile(r"\brepsol\b", re.I), "REPSOL"),
    (re.compile(r"\brovi\b", re.I), "ROVI"),
    (re.compile(r"\bsacyr\b", re.I), "SACYR"),
    (re.compile(r"\bsolaria\b", re.I), "SOLARIA"),
    (re.compile(r"\bunicaja\b", re.I), "UNICAJA"),
]

# Non-IBEX35 companies whose mention would pollute retrieval
_OUT_OF_DOMAIN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\btesla\b", re.I), "TESLA"),
    (re.compile(r"\bapple\b|\baapl\b", re.I), "APPLE"),
    (re.compile(r"\bamazon\b|\bamzn\b", re.I), "AMAZON"),
    (re.compile(r"\bgoogle\b|\balphabet\b|\bgoog\b", re.I), "ALPHABET"),
    (re.compile(r"\bmicrosoft\b|\bmsft\b", re.I), "MICROSOFT"),
    (re.compile(r"\bmeta\b|\bfacebook\b", re.I), "META"),
    (re.compile(r"\bnvidia\b|\bnvda\b", re.I), "NVIDIA"),
    (re.compile(r"\bnetflix\b|\bnflx\b", re.I), "NETFLIX"),
]

_GENERAL_MARKET = re.compile(r"\bibex\b|\bmercado espa[nñ]ol\b|\btodas las empresas\b", re.I)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_QUARTER_RE = re.compile(r"\b(q[1-4]|trimestre\s*[1-4])\b", re.I)

# Signals that the user asks for the annual (not quarterly) report
_ANNUAL_RE = re.compile(
    r"\binforme\s+anual\b|\banuales?\b|\bfull\s+year\b|\bfy\s*20\d{2}\b"
    r"|\bcierre\s+(?:del\s+)?a[nñ]o\b|\bresultados\s+anuales?\b",
    re.I,
)

# Signals that the user asks for the latest report
_LATEST_RE = re.compile(
    r"\b[uú]ltim[ao]s?\b|\bm[aá]s\s+reciente\b|\bactual\b|\brecientemente\b"
    r"|\blatest\b|\bmost\s+recent\b|\blast\s+(?:quarter|report)\b",
    re.I,
)


# ── QueryIntent ───────────────────────────────────────────────────────────────


@dataclass
class QueryIntent:
    """Structured intent extracted from the user's question."""

    companies: list[str] = field(default_factory=list)
    fiscal_year: str | None = None
    fiscal_quarter: str | None = None
    wants_annual: bool = False  # explicitly asks for the annual report
    wants_latest: bool = False  # asks for the most recent report, no fixed year
    is_comparison: bool = False  # two or more companies
    requires_specific_period: bool = False  # explicit year or quarter
    out_of_domain_companies: list[str] = field(default_factory=list)  # non-IBEX35

    @property
    def has_out_of_domain(self) -> bool:
        return bool(self.out_of_domain_companies)


# ── Public API ────────────────────────────────────────────────────────────────


def extract_companies(user_query: str) -> list[str]:
    """Extract the IBEX35 companies mentioned in the query, in order of appearance."""
    matches = sorted(
        (
            (found.start(), canonical)
            for pattern, canonical in _COMPANY_PATTERNS
            for found in pattern.finditer(user_query)
        ),
        key=lambda match: match[0],
    )
    return list(dict.fromkeys(canonical for _, canonical in matches))


def extract_out_of_domain_companies(user_query: str) -> list[str]:
    """Detect non-IBEX35 companies whose mention would pollute retrieval."""
    return list(
        dict.fromkeys(
            canonical
            for pattern, canonical in _OUT_OF_DOMAIN_PATTERNS
            if pattern.search(user_query)
        )
    )


def _extract_quarter(user_query: str) -> str | None:
    """Return the quarter mentioned in the query as `Q1`–`Q4`, if any."""
    match = _QUARTER_RE.search(user_query)
    if not match:
        return None
    digit = re.search(r"[1-4]", match.group(1))
    return f"Q{digit.group()}" if digit else None


def extract_query_intent(user_query: str) -> QueryIntent:
    """Parse the query into a structured QueryIntent."""
    companies = extract_companies(user_query)
    year_match = _YEAR_RE.search(user_query)
    fiscal_year = year_match.group(1) if year_match else None
    fiscal_quarter = _extract_quarter(user_query)

    return QueryIntent(
        companies=companies,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        wants_annual=bool(_ANNUAL_RE.search(user_query)),
        wants_latest=bool(_LATEST_RE.search(user_query)),
        is_comparison=len(companies) >= 2,
        requires_specific_period=bool(fiscal_year or fiscal_quarter),
        out_of_domain_companies=extract_out_of_domain_companies(user_query),
    )


def extract_filters(user_query: str) -> dict[str, Any]:
    """Build Qdrant metadata filters from a natural-language query.

    Returns only the fields that were found, e.g.
    `{"company": "IBERDROLA", "fiscal_year": "2024"}`. Multi-company questions
    use the `companies` key; general-market questions get no filters.
    """
    intent = extract_query_intent(user_query)
    filters: dict[str, Any] = {}
    if len(intent.companies) == 1:
        filters["company"] = intent.companies[0]
    elif intent.companies:
        filters["companies"] = intent.companies
    elif _GENERAL_MARKET.search(user_query):
        return filters

    if intent.fiscal_year:
        filters["fiscal_year"] = intent.fiscal_year
    if intent.fiscal_quarter:
        filters["fiscal_quarter"] = intent.fiscal_quarter
    return filters


def reorder_for_llm(docs: list[Document]) -> list[Document]:
    """Mitigate lost-in-the-middle: best chunk first, second-best last, rest in middle."""
    if len(docs) <= 2:
        return docs
    return [docs[0]] + docs[2:] + [docs[1]]
