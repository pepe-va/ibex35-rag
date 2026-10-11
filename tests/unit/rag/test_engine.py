"""Unit tests for the RAG engine's pure helpers: coverage gate, merging and reranking floor."""

from langchain_core.documents import Document

from src.rag.engine import (
    MIN_RERANKER_SCORE,
    RERANK_MIN_PER_COMPANY,
    _merge_interleaved,
    _with_company_floor,
    check_coverage,
    decide_abstention,
)


def _doc(company: str, score: float = 1.0, year: str = "2024", text: str = "") -> Document:
    metadata = {"company": company, "fiscal_year": year, "_score": score}
    return Document(page_content=text or f"{company} {year} {score}", metadata=metadata)


def test_coverage_classifies_companies():
    docs = [_doc("BBVA"), _doc("BBVA"), _doc("SANTANDER"), _doc("REPSOL", score=0.0)]
    report = check_coverage(docs, ["BBVA", "SANTANDER", "REPSOL"], requested_year=None)
    assert report.covered == ["BBVA"]
    assert report.partial == ["SANTANDER"]
    assert report.missing == ["REPSOL"]


def test_coverage_checks_requested_year():
    report = check_coverage([_doc("BBVA", year="2023")], ["BBVA"], requested_year="2024")
    assert report.period_ok is False
    assert report.available_periods == ["2023"]


def test_comparison_with_missing_company_abstains():
    report = check_coverage([_doc("BBVA"), _doc("BBVA")], ["BBVA", "REPSOL"], None)
    decision = decide_abstention(report, is_comparison=True, requires_specific_period=False)
    assert decision.abstain
    assert "REPSOL" in decision.reason


def test_irrelevant_chunks_abstain():
    docs = [_doc("BBVA", score=MIN_RERANKER_SCORE / 2)]
    report = check_coverage(docs, ["BBVA"], None)
    decision = decide_abstention(report, is_comparison=False, requires_specific_period=False)
    assert decision.abstain


def test_unavailable_period_abstains():
    report = check_coverage([_doc("BBVA", year="2023")] * 2, ["BBVA"], "2024")
    decision = decide_abstention(report, is_comparison=False, requires_specific_period=True)
    assert decision.abstain
    assert "2023" in decision.reason


def test_sufficient_coverage_does_not_abstain():
    report = check_coverage([_doc("BBVA"), _doc("BBVA")], ["BBVA"], "2024")
    decision = decide_abstention(report, is_comparison=False, requires_specific_period=True)
    assert not decision.abstain


def test_merge_interleaves_and_deduplicates():
    a1, a2 = _doc("A", text="a1"), _doc("A", text="a2")
    b1 = _doc("B", text="b1")
    merged = _merge_interleaved([[a1, a2], [b1, a1]], top_k=10)
    assert [d.page_content for d in merged] == ["a1", "b1", "a2"]


def test_merge_respects_top_k():
    docs = [
        [_doc("A", text=f"a{i}") for i in range(5)],
        [_doc("B", text=f"b{i}") for i in range(5)],
    ]
    assert len(_merge_interleaved(docs, top_k=3)) == 3


def test_company_floor_keeps_minimum_chunks_per_company():
    reranked = [(0.9 - i * 0.01, _doc("A", text=f"a{i}")) for i in range(10)]
    reranked += [(0.1, _doc("B", text=f"b{i}")) for i in range(5)]
    top = _with_company_floor(reranked, ["A", "B"], top_k=6)
    companies = [doc.metadata["company"] for _, doc in top]
    assert companies.count("B") == RERANK_MIN_PER_COMPANY
    assert len(top) == 6
    assert [score for score, _ in top] == sorted((score for score, _ in top), reverse=True)


def test_llm_request_timeout_reaches_the_ollama_client(settings):
    from unittest.mock import MagicMock, patch

    from src.rag.engine import RAGEngine

    with (
        patch("src.rag.engine.HuggingFaceCrossEncoder"),
        patch("src.rag.engine._best_device", return_value="cpu"),
    ):
        engine = RAGEngine(settings=settings, vector_store=MagicMock())
    assert engine._llm.client_kwargs["timeout"] == settings.llm_request_timeout
