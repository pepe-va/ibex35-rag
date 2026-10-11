"""Unit tests for auto-routing heuristics."""

from src.api.routers.query import _looks_like_market_question


def test_market_heuristic_detects_precio_de_mercado_ahora():
    question = "Combina para INDRA: último EBITDA reportado y precio de mercado ahora."
    assert _looks_like_market_question(question) is True
