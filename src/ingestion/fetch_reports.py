#!/usr/bin/env python3
"""Download the latest results report of each IBEX35 company and verify it before keeping it.

A report is kept only when it is a readable PDF in which
  - the company is the IBEX35 company mentioned most often, and
  - the latest reporting period on its first pages is the one in report_sources.csv
    (so neither last year's report nor a newer one passes for it).
Anything else is discarded, so a wrong link never reaches the ingestion.

Usage:
    uv run python src/ingestion/fetch_reports.py --pdf-dir ./data/raw/pdfs
    make fetch
"""

import argparse
import csv
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pypdfium2 as pdfium  # installed with docling

from src.core.config import get_settings

SOURCES_CSV = Path(__file__).with_name("report_sources.csv")
MAX_PDF_BYTES = 100 * 1024 * 1024
# Pages searched for the reporting period: the cover and what follows it.
HEAD_PAGES = 5
TIMEOUT_SECONDS = 60
# Several investor-relations sites reject requests without a browser User-Agent.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0 Safari/537.36"
)

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

# Reporting periods: closing month and the names they go by (normalised text).
PERIODS: dict[str, tuple[int, str]] = {
    "Q1": (
        3,
        "1t|q1|1q|3m|primer trimestre|first quarter|tres meses|three months"
        "|enero-marzo|january-march",
    ),
    "H1": (
        6,
        "1s|h1|1h|6m|2t|q2|2q|primer semestre|1er semestre|first half|half-year"
        "|half year|seis meses|six months|segundo trimestre|second quarter"
        "|enero-junio|january-june",
    ),
    "9M": (
        9,
        "9m|3t|q3|3q|nueve meses|9 meses|nine months|9 months|tercer trimestre"
        "|third quarter|enero-septiembre|january-september",
    ),
}
MONTH_NAMES = {3: "marzo|march", 6: "junio|june", 9: "septiembre|september"}


# ── Sources ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReportSource:
    key: str
    company: str
    period: str
    year: str
    url: str
    results_page: str
    aliases: tuple[str, ...]
    key_figures: tuple[str, ...]

    @property
    def filename(self) -> str:
        # Company, doc_type, period and year, as extract_metadata_from_filename expects.
        return f"{self.key} financial_results {self.period.lower()} {self.year}.pdf"

    @property
    def period_end(self) -> tuple[int, int]:
        return int(self.year), PERIODS[self.period][0]


def load_sources(path: Path = SOURCES_CSV) -> list[ReportSource]:
    """Read the report manifest (one row per company, `;`-separated)."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    sources = []
    for row in rows:
        period, year = row["period"].split()
        if period not in PERIODS:
            raise ValueError(f"{row['company']}: unknown period {period!r}")
        sources.append(
            ReportSource(
                key=row["key"],
                company=row["company"],
                period=period,
                year=year,
                url=row["url"],
                results_page=row["results_page"],
                aliases=tuple(a for a in row["aliases"].split("|") if a),
                key_figures=tuple(n for n in row["key_figures"].split("|") if n),
            )
        )
    return sources


# ── Text matching ─────────────────────────────────────────────────────────────


def normalize(text: str) -> str:
    """Lowercase, strip accents, unify dashes, collapse spaces: "Enero – Junio" → "enero-junio"."""
    text = re.sub(r"\s*[-\u2010-\u2015]\s*", "-", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text).lower()


def company_mentions(text: str, sources: Sequence[ReportSource]) -> Counter[str]:
    """How often each company is named. Longer aliases go first and are blanked out,
    so "acciona energia" never also counts as "acciona"."""
    aliases = sorted(
        ((alias, s.key) for s in sources for alias in s.aliases), key=lambda a: -len(a[0])
    )
    counts: Counter[str] = Counter()
    for alias, key in aliases:
        text, found = re.subn(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", " ", text)
        counts[key] += found
    return counts


def period_mentions(text: str) -> Counter[tuple[int, int]]:
    """How often each (year, closing month) is named as a period ("1S 2026", "9M26"),
    a closing month ("septiembre 2026") or a closing date ("30/06/2026", "June 30, 2026")."""
    year = r"(?:[ -]?(?:de |del |of )?(20\d\d)|[-']?(\d\d))(?!\d)"
    found: Counter[tuple[int, int]] = Counter()
    for month, names in PERIODS.values():
        mm = f"{month:02d}"
        months = MONTH_NAMES[month]
        patterns = (
            rf"(?<![a-z0-9])(?:{names}){year}",
            # A month counts with no day before it, or with its closing day only, so
            # that dates in legal boilerplate ("14 de marzo de 2019") are ignored.
            rf"(?<![a-z])(?<!\d )(?<!\d de )(?:{months}),? (?:de |of )?(20\d\d)()",
            rf"(?<!\d)3[01] (?:de )?(?:{months}),? (?:de |of )?(20\d\d)()",
            rf"(?:{months}) 3[01],? (20\d\d)()",
            rf"(?<!\d)3[01][./-]{mm}[./-](20\d\d)()",
        )
        for pattern in patterns:
            for long_year, short_year in re.findall(pattern, text):
                found[int(long_year or f"20{short_year}"), month] += 1
    return found


def period_label(period_end: tuple[int, int]) -> str:
    year, month = period_end
    name = next(name for name, (m, _) in PERIODS.items() if m == month)
    return f"{name} {year}"


_NUMBER_RE = re.compile(r"\d{1,3}(?:[., ]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?")


def number_readings(token: str) -> set[float]:
    """Values a printed number can stand for: 3.574 is 3574 in Spanish and 3.574 in English."""
    token = token.replace(" ", "")
    readings = set()
    for spelled in (token.replace(".", "").replace(",", "."), token.replace(",", "")):
        try:
            readings.add(float(spelled))
        except ValueError:
            continue
    return readings


def contains_figure(text: str, figure: str) -> bool:
    """Whether a Spanish-formatted figure in M€ (e.g. 3.334,9) appears in the report.

    Matches it at its printed precision, so 180 accepts 180,0, and also expressed
    in thousands of euros (124,9 M€ printed as 124.939).
    """
    integer, _, decimals = figure.partition(",")
    target = float(f"{integer.replace('.', '')}.{decimals or 0}")
    tolerance = 0.5 * 10 ** -len(decimals)
    return any(
        abs(value - target) <= tolerance or abs(value / 1000 - target) <= tolerance
        for token in _NUMBER_RE.findall(text)
        for value in number_readings(token)
    )


# ── Verification ──────────────────────────────────────────────────────────────


@dataclass
class Verification:
    pages: int = 0
    problems: list[str] = field(default_factory=list)
    missing_figures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def read_pdf_pages(path: Path) -> list[str]:
    """Plain text of each page. Raises pdfium.PdfiumError when the file is not a valid PDF."""
    pdf = pdfium.PdfDocument(path)
    try:
        pages = []
        for index in range(len(pdf)):
            page = pdf[index]
            textpage = page.get_textpage()
            pages.append(textpage.get_text_range())
            textpage.close()
            page.close()
        return pages
    finally:
        pdf.close()


def check_company(text: str, source: ReportSource, sources: Sequence[ReportSource]) -> str:
    """Problem with the company the report is about, or "" when it is `source`."""
    counts = company_mentions(text, sources)
    if not counts[source.key]:
        return f"el texto no menciona a {source.company}"
    top_key, top_count = counts.most_common(1)[0]
    if counts[source.key] < top_count:
        top = next(s.company for s in sources if s.key == top_key)
        return f"parece de {top} ({top_count} menciones frente a {counts[source.key]})"
    return ""


def reported_period(pages: Sequence[str]) -> tuple[int, int] | None:
    """The period a report is about: the one named most often on the first pages that name
    any, so the cover outweighs comparative tables. A tie goes to the latest period."""
    for n in range(1, min(HEAD_PAGES, len(pages)) + 1):
        mentions = period_mentions(normalize(" ".join(pages[:n])))
        if mentions:
            top = max(mentions.values())
            return max(period for period, count in mentions.items() if count == top)
    return None


def check_period(pages: Sequence[str], source: ReportSource) -> str:
    """Problem with the reporting period, or "" when it is `source`'s."""
    expected = period_label(source.period_end)
    reported = reported_period(pages)
    if reported is None:
        return f"no se identifica el periodo (se esperaba {expected})"
    if reported != source.period_end:
        return f"es de {period_label(reported)}, no de {expected}"
    return ""


def verify_report(
    path: Path, source: ReportSource, sources: Sequence[ReportSource]
) -> Verification:
    """Check that the PDF at `path` is the report `source` describes."""
    try:
        pages = read_pdf_pages(path)
    except pdfium.PdfiumError as exc:
        return Verification(problems=[f"PDF ilegible ({exc})"])

    result = Verification(pages=len(pages))
    text = normalize(" ".join(pages))
    if not text.strip():
        result.problems.append("el PDF no tiene texto extraíble")
        return result
    # Covers often carry the name only in the logo, so the whole report is searched.
    result.problems = [
        p for p in (check_company(text, source, sources), check_period(pages, source)) if p
    ]
    # Soft check: the headline figure of the index should appear somewhere in the report.
    result.missing_figures = [f for f in source.key_figures if not contains_figure(text, f)]
    return result


# ── Download ──────────────────────────────────────────────────────────────────


class DownloadError(Exception):
    pass


def download(client: httpx.Client, url: str, dest: Path) -> None:
    """Stream `url` into `dest`, refusing anything that is not a PDF or is too large."""
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            size = 0
            with dest.open("wb") as out:
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > MAX_PDF_BYTES:
                        raise DownloadError(f"supera {MAX_PDF_BYTES // 2**20} MB")
                    out.write(chunk)
            content_type = response.headers.get("content-type", "?")
    except httpx.HTTPStatusError as exc:
        raise DownloadError(f"HTTP {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise DownloadError(f"{type(exc).__name__}: {exc}") from exc

    # The PDF header may be preceded by up to 1 KB of junk.
    with dest.open("rb") as f:
        if b"%PDF-" not in f.read(1024):
            raise DownloadError(f"el servidor no devolvió un PDF ({content_type.split(';')[0]})")


# ── Fetch one company ─────────────────────────────────────────────────────────


@dataclass
class Outcome:
    source: ReportSource
    verification: Verification | None = None
    error: str = ""
    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.verification is not None and self.verification.ok


def fetch_report(
    client: httpx.Client,
    source: ReportSource,
    sources: Sequence[ReportSource],
    pdf_dir: Path,
    force: bool = False,
) -> Outcome:
    """Download and verify one report. Only a verified PDF ends up in pdf_dir."""
    target = pdf_dir / source.filename
    if target.exists() and not force:
        verification = verify_report(target, source, sources)
        if verification.ok:
            return Outcome(source, verification, cached=True)
        target.unlink()

    # Not *.pdf, so a partial download is never picked up by the ingestion.
    partial = pdf_dir / f".{source.filename}.part"
    try:
        download(client, source.url, partial)
        verification = verify_report(partial, source, sources)
        if verification.ok:
            partial.replace(target)
        return Outcome(source, verification)
    except DownloadError as exc:
        return Outcome(source, error=str(exc))
    finally:
        partial.unlink(missing_ok=True)


# ── Terminal output ───────────────────────────────────────────────────────────


def describe(outcome: Outcome, path: Path) -> str:
    """One status line: green tick with details, or red cross with the reason."""
    v = outcome.verification
    if v is None:
        return f"{RED}✗ {outcome.error}{RESET}"
    if not v.ok:
        return f"{RED}✗ documento incorrecto: {'; '.join(v.problems)}{RESET}"
    source = outcome.source
    size_mb = path.stat().st_size / 2**20
    line = f"{GREEN}✓{RESET} {source.period} {source.year} · {v.pages} págs · {size_mb:.1f} MB"
    if outcome.cached:
        line += f" {DIM}(ya descargado){RESET}"
    if v.missing_figures:
        figures = ", ".join(v.missing_figures)
        line += f" {YELLOW}· cifra del índice no encontrada: {figures} M€{RESET}"
    return line


def print_summary(outcomes: list[Outcome], pdf_dir: Path) -> None:
    verified = sum(o.ok for o in outcomes)
    colour = GREEN if verified == len(outcomes) else YELLOW
    print(f"\n  {colour}{verified}/{len(outcomes)} informes verificados{RESET} en {pdf_dir}")
    failed = [o for o in outcomes if not o.ok]
    if failed:
        print("  Pendientes (descárgalos a mano y guárdalos con el nombre indicado):")
        for o in failed:
            print(f"    - {o.source.company}: {o.source.results_page}")
            print(f"      {DIM}→ {pdf_dir / o.source.filename}{RESET}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descargar y verificar los últimos informes de resultados del IBEX35."
    )
    parser.add_argument(
        "--pdf-dir", default=None, help="Destino de los PDF (por defecto: settings.pdf_dir)"
    )
    parser.add_argument("--force", action="store_true", help="Volver a descargar todo")
    parser.add_argument("--only", nargs="+", metavar="KEY", help="Solo estas empresas (ej. BBVA)")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir or get_settings().pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    sources = load_sources()
    selected = sources
    if args.only:
        wanted = {key.upper() for key in args.only}
        selected = [s for s in sources if s.key in wanted]

    print(f"► Últimos resultados del IBEX35 → {pdf_dir}")
    outcomes = []
    headers = {"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*;q=0.8"}
    with httpx.Client(headers=headers, follow_redirects=True, timeout=TIMEOUT_SECONDS) as client:
        for n, source in enumerate(selected, 1):
            print(f"  [{n:02d}/{len(selected)}] {source.company:<18} ", end="", flush=True)
            outcome = fetch_report(client, source, sources, pdf_dir, force=args.force)
            print(describe(outcome, pdf_dir / source.filename), flush=True)
            outcomes.append(outcome)

    print_summary(outcomes, pdf_dir)
    # Fail only when there is nothing to ingest at all (e.g. offline on a fresh clone).
    if not any(pdf_dir.glob("*.pdf")):
        sys.exit(1)


if __name__ == "__main__":
    main()
