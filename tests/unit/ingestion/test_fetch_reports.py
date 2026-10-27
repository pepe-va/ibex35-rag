"""Unit tests for the report downloader and its verification."""

from pathlib import Path

import httpx
import pytest

from src.ingestion import fetch_reports
from src.ingestion.fetch_reports import (
    ReportSource,
    company_mentions,
    contains_figure,
    fetch_report,
    load_sources,
    normalize,
    period_mentions,
    reported_period,
    verify_report,
)
from src.ingestion.metadata import COMPANY_TICKER_MAP

SOURCES = load_sources()
BY_KEY = {s.key: s for s in SOURCES}
BBVA_H1 = BY_KEY["BBVA"]
BBVA_PAGES = [
    "BBVA Informe de resultados Enero – Junio 2026",
    "Resultado atribuido 6.051 M€ frente a 1S 2025. BBVA mantiene su objetivo.",
]


def test_manifest_covers_the_ibex35_with_known_companies():
    assert len(SOURCES) == 35
    assert {s.key for s in SOURCES} <= set(COMPANY_TICKER_MAP)
    assert all(s.url.startswith("https://") for s in SOURCES)


def test_filename_follows_the_ingestion_convention():
    assert BBVA_H1.filename == "BBVA financial_results h1 2026.pdf"


def test_normalize_unifies_dashes_and_accents():
    assert normalize("Enero – Junio  Telefónica") == "enero-junio telefonica"


def test_company_mentions_counts_longest_alias_first():
    text = normalize("Acciona Energía, filial de Acciona. Acciona Energía crece.")
    counts = company_mentions(text, SOURCES)
    assert counts["ACCIONAENERGIA"] == 2
    assert counts["ACCIONA"] == 1


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Resultados 1S 2026", (2026, 6)),
        ("Rdos-1S26", (2026, 6)),
        ("H1 2026 financial results", (2026, 6)),
        ("first half of 2026", (2026, 6)),
        ("Información Consolidada 3T Septiembre 2026", (2026, 9)),
        ("Nota CNMV Resultados 9M2026", (2026, 9)),
        ("a 30 de junio de 2026", (2026, 6)),
        ("Balance at 30/09/2026", (2026, 9)),
        ("six months ended June 30, 2026", (2026, 6)),
    ],
)
def test_period_mentions_recognises_common_spellings(text, expected):
    assert expected in period_mentions(normalize(text))


def test_period_mentions_ignores_dates_that_do_not_close_a_period():
    assert not period_mentions(normalize("Reglamento (UE) de 14 de marzo de 2019"))


def test_reported_period_is_decided_by_the_cover():
    pages = ["Presentación de Resultados 1S-2026", "1S 2025 1S 2025 1S 2025 1S 2026"]
    assert reported_period(pages) == (2026, 6)


def test_reported_period_ignores_a_single_forward_looking_mention():
    pages = ["Resultados 1S 2026. Resultados 1S 2026. Cierre previsto en el 1T 2027."]
    assert reported_period(pages) == (2026, 6)


def test_reported_period_breaks_ties_with_the_latest():
    assert reported_period(["9M 2026 frente a 1S 2026"]) == (2026, 9)


@pytest.mark.parametrize(
    "text, figure",
    [
        ("Beneficio neto 1.215 M€", "1.215"),
        ("Revenue €3,574m", "3.574"),
        ("FFO 180,0", "180"),
        ("Resultado del ejercicio 124.939 (miles de euros)", "124,9"),
        ("Ingresos 3.334,9", "3.334,9"),
    ],
)
def test_contains_figure_matches_other_spellings(text, figure):
    assert contains_figure(text, figure)


def test_contains_figure_rejects_a_different_figure():
    assert not contains_figure("Beneficio neto 1.002,0 M€", "808,1")


@pytest.fixture
def fake_pdf(monkeypatch):
    """Make every PDF read as the pages in `pages`."""
    pages: list[str] = []
    monkeypatch.setattr(fetch_reports, "read_pdf_pages", lambda path: pages)
    return pages


def test_verify_report_accepts_the_expected_report(fake_pdf, tmp_path):
    fake_pdf.extend(BBVA_PAGES)
    result = verify_report(tmp_path / "x.pdf", BBVA_H1, SOURCES)
    assert result.ok
    assert result.missing_figures == []


def test_verify_report_rejects_another_company(fake_pdf, tmp_path):
    fake_pdf.extend(BBVA_PAGES)
    result = verify_report(tmp_path / "x.pdf", BY_KEY["SANTANDER"], SOURCES)
    assert not result.ok
    assert "no menciona a Banco Santander" in result.problems[0]


def test_verify_report_rejects_a_company_that_is_only_mentioned(fake_pdf, tmp_path):
    fake_pdf.extend([*BBVA_PAGES, "Cliente: Telefónica"])
    result = verify_report(tmp_path / "x.pdf", BY_KEY["TELEFONICA"], SOURCES)
    assert result.problems == ["parece de BBVA (2 menciones frente a 1)"]


def test_verify_report_rejects_another_period(fake_pdf, tmp_path):
    fake_pdf.extend(["BBVA Resultados 9M 2026"])
    result = verify_report(tmp_path / "x.pdf", BBVA_H1, SOURCES)
    assert result.problems == ["es de 9M 2026, no de H1 2026"]


def test_verify_report_flags_a_missing_figure_without_rejecting(fake_pdf, tmp_path):
    fake_pdf.extend(["BBVA resultados 1S 2026, beneficio 5.000 M€"])
    result = verify_report(tmp_path / "x.pdf", BBVA_H1, SOURCES)
    assert result.ok
    assert result.missing_figures == ["6.051"]


def _client(content: bytes, status: int = 200) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, content=content))
    )


def _saved(pdf_dir: Path) -> list[str]:
    return sorted(p.name for p in pdf_dir.iterdir())


def test_fetch_report_saves_a_verified_pdf(fake_pdf, tmp_path):
    fake_pdf.extend(BBVA_PAGES)
    outcome = fetch_report(_client(b"%PDF-1.7 ..."), BBVA_H1, SOURCES, tmp_path)
    assert outcome.ok
    assert _saved(tmp_path) == [BBVA_H1.filename]


def test_fetch_report_discards_a_wrong_pdf(fake_pdf, tmp_path):
    fake_pdf.extend(["Banco Santander resultados 1S 2026"])
    outcome = fetch_report(_client(b"%PDF-1.7 ..."), BBVA_H1, SOURCES, tmp_path)
    assert not outcome.ok
    assert _saved(tmp_path) == []


def test_fetch_report_rejects_an_html_page(tmp_path):
    outcome = fetch_report(_client(b"<html>Just a moment...</html>"), BBVA_H1, SOURCES, tmp_path)
    assert outcome.error.startswith("el servidor no devolvió un PDF")
    assert _saved(tmp_path) == []


def test_fetch_report_reports_http_errors(tmp_path):
    outcome = fetch_report(_client(b"", status=403), BBVA_H1, SOURCES, tmp_path)
    assert outcome.error == "HTTP 403"


def test_fetch_report_reuses_a_verified_download(fake_pdf, tmp_path):
    fake_pdf.extend(BBVA_PAGES)
    (tmp_path / BBVA_H1.filename).write_bytes(b"%PDF-1.7 ...")
    outcome = fetch_report(_client(b"", status=500), BBVA_H1, SOURCES, tmp_path)
    assert outcome.ok and outcome.cached


def test_report_source_period_end():
    source = ReportSource("X", "X", "9M", "2026", "https://x", "https://x", ("x",), ())
    assert source.period_end == (2026, 9)
