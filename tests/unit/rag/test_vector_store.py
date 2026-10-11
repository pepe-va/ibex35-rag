"""Unit tests for chunk-level period annotation in the vector store."""

import pytest

from src.rag.vector_store.store import _annotate_chunk_period, _classify

BASE = {"company": "IBERDROLA", "fiscal_year": "2024", "fiscal_quarter": None}


@pytest.mark.parametrize(
    ("text", "year", "quarter"),
    [
        ("Resultados Q3 2023", "2023", "Q3"),
        ("Cifras del 3T 2025", "2025", "Q3"),
        ("Acumulado 9M 2025 frente a 2024 y 2024", "2024", "9M"),
        ("tercer trimestre de 2022", "2022", "Q3"),
        ("texto sin periodo", "2024", None),
    ],
)
def test_chunk_period_overrides_document_period(text, year, quarter):
    meta = _annotate_chunk_period(text, BASE)
    assert meta["fiscal_year"] == year
    assert meta["fiscal_quarter"] == quarter


def test_annotation_does_not_mutate_base_metadata():
    _annotate_chunk_period("Q1 2020", BASE)
    assert BASE["fiscal_year"] == "2024"


def test_classify_by_folder(tmp_path):
    assert _classify(tmp_path / "markdown/BBVA/report.md") == ("text", "report")
    assert _classify(tmp_path / "tables/BBVA/report/table_1_page_2.md") == ("table", "report")
    assert _classify(tmp_path / "other/file.md") == ("unknown", "file")
