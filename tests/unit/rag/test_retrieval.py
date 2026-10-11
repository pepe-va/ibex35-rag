"""Unit tests for retrieval filter extraction."""

from src.rag.retrieval import extract_companies, extract_filters


def test_extract_companies_preserves_query_order():
    companies = extract_companies("En el IBEX35, compara fundamentales de MERLIN y BBVA en 2024.")
    assert companies == ["MERLINPROPERTIES", "BBVA"]


def test_extract_filters_supports_multi_company_and_year():
    filters = extract_filters(
        "Compara el margen EBITDA y los ingresos de PUIG frente a ENAGAS en 2024."
    )
    assert filters["companies"] == ["PUIG", "ENAGAS"]
    assert filters["fiscal_year"] == "2024"


def test_extract_filters_keeps_company_even_if_ibex_is_mentioned():
    filters = extract_filters("En el IBEX35, ¿qué EBITDA reportó SANTANDER?")
    assert filters["company"] == "SANTANDER"


def test_extract_filters_general_market_without_companies_returns_empty_filters():
    filters = extract_filters("¿Cómo va el IBEX35 hoy?")
    assert filters == {}


def test_every_detectable_company_has_a_ticker():
    from src.ingestion.metadata import COMPANY_TICKER_MAP
    from src.rag.retrieval import _COMPANY_PATTERNS

    assert {company for _, company in _COMPANY_PATTERNS} == set(COMPANY_TICKER_MAP)
