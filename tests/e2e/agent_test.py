#!/usr/bin/env python3
"""
Agent test runner for the IBEX35 RAG system.

Generates questions automatically with Ollama and runs them against the API,
recording the chosen route, latency, tools used and answer quality.

Usage:
    uv run python tests/e2e/agent_test.py
    uv run python tests/e2e/agent_test.py --rounds 20
    uv run python tests/e2e/agent_test.py --duration 5m
    uv run python tests/e2e/agent_test.py --duration 30s --seed 42
    uv run python tests/e2e/agent_test.py --questions-file preguntas.txt
    make agent_test
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from src.core.config import Settings, get_settings

# ── Defaults ──────────────────────────────────────────────────────────────────


def _load_defaults_from_config() -> Settings:
    try:
        return get_settings()
    except Exception:
        # E2E scripts can run in lean environments without Grafana credentials.
        return Settings(grafana_admin_user="", grafana_admin_password="")


_SETTINGS = _load_defaults_from_config()

DEFAULT_API_URL = _SETTINGS.api_url
DEFAULT_ENDPOINT = _SETTINGS.test_endpoint
DEFAULT_OLLAMA_URL = _SETTINGS.ollama_base_url
# Use the router model if set; otherwise fall back to the main LLM.
DEFAULT_MODEL = _SETTINGS.effective_router_model
DEFAULT_GENERATOR_BACKEND = _SETTINGS.agent_test_generator_backend
DEFAULT_GENERATOR_API_KEY = (
    _SETTINGS.agent_test_generator_api_key.get_secret_value()
    if _SETTINGS.agent_test_generator_api_key is not None
    else (
        _SETTINGS.openai_api_key.get_secret_value() if _SETTINGS.openai_api_key is not None else ""
    )
)

# ── Categories for automatic question generation ──────────────────────────────

CATEGORIES = [
    "fundamentales",  # EBITDA, beneficio neto, deuda, ingresos
    "mercado_tiempo_real",  # quote, current price, ticker
    "comparativa",  # compare two or more companies
    "mixta",  # combines historical data + market
    "ambigua",  # question where the router must choose well
    "seguimiento",  # follow-up to the previous question
    "english_market",  # market, in English
    "english_fundamentals",  # fundamentals, in English
    "sector",  # question about a sector or group of companies
    "tendencia",  # evolution of a metric over time
]

# Sampling weights: the most frequent categories cover the main use case.
# Same order as CATEGORIES; they add up to 100.
CATEGORY_WEIGHTS = [40, 20, 15, 10, 5, 3, 2, 2, 2, 1]

# Categories outside the IBEX35 domain (5% by default)
OFF_TOPIC_CATEGORIES = [
    "fuera_dominio",  # banca retail / personal
    "empresa_no_ibex",  # empresas globales (Apple, Tesla…)
    "noise",  # questions with no financial meaning
]

CATEGORY_RULES: dict[str, dict[str, str | None]] = {
    "fundamentales": {
        "expected_route": "rag",
        "focus": "fundamentales de negocio (ingresos, EBITDA, beneficio, deuda, dividendo)",
    },
    "mercado_tiempo_real": {
        "expected_route": "agent",
        "focus": "cotización actual, precio hoy, variación intradía, rango 52 semanas",
    },
    "comparativa": {
        "expected_route": "rag",
        "focus": "comparar fundamentales entre dos empresas IBEX35",
    },
    "mixta": {
        "expected_route": "agent",
        "focus": "combinar un dato histórico de informes + precio actual",
    },
    "ambigua": {
        "expected_route": "rag",
        "focus": "pregunta de negocio/fundamentales sin precio en tiempo real",
    },
    "seguimiento": {
        "expected_route": "rag",
        "focus": "pregunta de continuidad sobre una métrica financiera previa",
    },
    "english_market": {
        "expected_route": "agent",
        "focus": "in English, ask about current stock price/market move",
    },
    "english_fundamentals": {
        "expected_route": "rag",
        "focus": "in English, ask about reported financial fundamentals",
    },
    "sector": {"expected_route": "rag", "focus": "sector view based on reported fundamentals"},
    "tendencia": {
        "expected_route": "agent",
        "focus": "short-term market trend / trading performance",
    },
    # Off-topic: no fixed expected route; only check that the API answers 200
    "fuera_dominio": {
        "expected_route": None,
        "focus": "banca retail / finanzas personales — fuera de dominio",
    },
    "empresa_no_ibex": {"expected_route": None, "focus": "empresa global no cotizada en IBEX35"},
    "noise": {"expected_route": None, "focus": "pregunta sin contenido financiero relevante"},
}

COMPANY_TERMS = [
    "iberdrola",
    "santander",
    "inditex",
    "bbva",
    "telefonica",
    "acciona",
    "acerinox",
    "acs",
    "aena",
    "amadeus",
    "arcelormittal",
    "cellnex",
    "enagas",
    "endesa",
    "ferrovial",
    "fluidra",
    "grifols",
    "iag",
    "indra",
    "logista",
    "mapfre",
    "merlin",
    "naturgy",
    "puig",
    "redeia",
    "repsol",
    "rovi",
    "sacyr",
    "solaria",
    "unicaja",
]

OFFTOPIC_TERMS = [
    "hipoteca",
    "hipotecario",
    "mi cuenta",
    "cuenta personal",
    "saldo disponible",
    "depósito",
    "deposito",
    "retirar",
    "retirada",
    "transferencia automática",
    "pago automático",
    "auto-routing",
    "prestamo personal",
    "préstamo personal",
]

RAG_HINTS = [
    "ebitda",
    "ingresos",
    "beneficio",
    "deuda",
    "dividendo",
    "margen",
    "guidance",
    "capex",
    "cash flow",
    "flujo de caja",
    "resultado",
    "fundamentales",
]

AGENT_HINTS = [
    "precio",
    "cotiza",
    "cotización",
    "hoy",
    "ahora",
    "tiempo real",
    "52 semanas",
    "52-week",
    "stock price",
    "trading",
    "mercado",
    "variación bursátil",
]

LOCAL_IMPROVISED_GOALS = {
    "rag": "Valida routing RAG con fundamentales documentales del IBEX35.",
    "agent": "Valida routing AGENT con señales de mercado en tiempo real.",
    "off_topic": "Valida comportamiento gracioso ante pregunta fuera de dominio.",
}

# ── SLA per category ──────────────────────────────────────────────────────────

SLA_BY_CATEGORY: dict[str, float] = {
    "fundamentales": 30.0,
    "comparativa": 45.0,
    "mercado_tiempo_real": 15.0,
    "mixta": 60.0,
    "ambigua": 35.0,
    "seguimiento": 35.0,
    "english_market": 15.0,
    "english_fundamentals": 30.0,
    "sector": 40.0,
    "tendencia": 20.0,
    "fuera_dominio": 8.0,  # must fail fast (abstention)
    "empresa_no_ibex": 8.0,
    "noise": 8.0,
    "manual": 60.0,
}

# Valid abstention signals in the answer
_ABSTENTION_SIGNALS = (
    "no dispongo",
    "no tengo información",
    "no tengo datos",
    "no hay información",
    "no puedo responder",
    "no puedo comparar",
    "i don't have",
    "i don't know",
    "no information available",
    "no puedo",
    "datos insuficientes",
    "información insuficiente",
    "no está disponible",
    "no se encuentra disponible",
    "no forma parte del universo",
)


# ── Assertions checked on the response ────────────────────────────────────────


@dataclass
class Assertion:
    name: str
    passed: bool
    detail: str = ""


def _assert_route(body: dict, expected: str | None) -> Assertion:
    route = body.get("route")
    if expected is None:
        return Assertion("route_present", route is not None, f"route={route}")
    passed = route == expected
    return Assertion(f"route=={expected}", passed, f"got route={route}")


def _assert_non_empty_answer(body: dict) -> Assertion:
    answer = body.get("answer", "")
    passed = bool(answer and len(answer.strip()) > 20)
    return Assertion("answer_non_empty", passed, f"len={len(answer)}")


def _assert_has_sources(body: dict) -> Assertion:
    sources = body.get("sources", [])
    passed = len(sources) > 0
    return Assertion("has_sources", passed, f"sources={len(sources)}")


def _assert_latency(latency: float, max_seconds: float) -> Assertion:
    passed = latency <= max_seconds
    return Assertion(f"latency<={max_seconds}s", passed, f"actual={latency:.2f}s")


def _assert_sla_by_category(latency: float, category: str) -> Assertion:
    """Category-specific SLA, stricter than the global one."""
    sla = SLA_BY_CATEGORY.get(category, 60.0)
    passed = latency <= sla
    return Assertion(f"sla_{category}<={sla}s", passed, f"actual={latency:.2f}s")


def _assert_both_companies_covered(body: dict, question: str) -> Assertion | None:
    """
    For comparison questions, check that the answer mentions both companies.
    An explicit abstention ('no dispongo de X') is also valid.
    Returns None if the question does not name 2 companies.
    """
    q_norm = _normalize_question(question)
    companies_in_q = [t for t in COMPANY_TERMS if re.search(rf"\b{re.escape(t)}\b", q_norm)]
    if len(companies_in_q) < 2:
        return None

    answer = (body.get("answer") or "").lower()

    # An explicit abstention is valid
    if any(sig in answer for sig in _ABSTENTION_SIGNALS):
        return Assertion(
            "both_companies_covered",
            True,
            f"abstención explícita (companies={companies_in_q})",
        )

    # Check that the answer mentions both
    missing = [c for c in companies_in_q if c not in answer]
    passed = len(missing) == 0
    detail = "ok" if passed else f"falta en respuesta: {missing}"
    return Assertion("both_companies_covered", passed, detail)


def _assert_period_adherence(body: dict, question: str) -> Assertion | None:
    """
    If the question asks for a specific year, the dominant year in the answer
    must match it. Returns None if the question has no year.
    """
    year_match = re.search(r"\b(20\d{2})\b", question)
    if not year_match:
        return None
    requested_year = year_match.group(1)

    answer = body.get("answer") or ""

    # An abstention is valid (no figures to check)
    if any(sig in answer.lower() for sig in _ABSTENTION_SIGNALS):
        return Assertion("period_adherence", True, f"abstención explícita para {requested_year}")

    years_in_answer = re.findall(r"\b(20\d{2})\b", answer)
    if not years_in_answer:
        # No years in the answer: probably a very generic answer
        return Assertion("period_adherence", True, "sin años en respuesta (indeterminado)")

    from collections import Counter

    dominant_year = Counter(years_in_answer).most_common(1)[0][0]
    passed = dominant_year == requested_year
    return Assertion(
        "period_adherence",
        passed,
        f"pedido={requested_year} dominante={dominant_year} "
        f"(all={Counter(years_in_answer).most_common(3)})",
    )


def _assert_clean_abstention_off_topic(body: dict, category: str) -> Assertion | None:
    """
    For out-of-domain categories (empresa_no_ibex, fuera_dominio):
    the answer must abstain and must not invent financial figures.
    """
    if category not in {"empresa_no_ibex", "fuera_dominio"}:
        return None

    answer = (body.get("answer") or "").lower()
    has_abstention = any(sig in answer for sig in _ABSTENTION_SIGNALS)

    # Detect invented financial figures (numbers with financial units)
    financial_numbers = re.compile(
        r"\b\d+[.,]?\d*\s*(?:millones?|miles?\s+de\s+millones?|bn|mm|%|€|\$|eur|usd)\b",
        re.I,
    )
    has_fabricated_numbers = bool(financial_numbers.search(body.get("answer") or ""))

    passed = has_abstention and not has_fabricated_numbers
    detail = f"abstention={has_abstention} fabricated_numbers={has_fabricated_numbers}"
    return Assertion("clean_abstention_off_topic", passed, detail)


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class RoundResult:
    round_index: int
    category: str
    question: str
    goal: str
    thread_id: str | None
    status_code: int
    route: str | None
    latency_seconds: float
    answer_preview: str
    tools_used: list[str] = field(default_factory=list)
    steps_taken: int = 0
    assertions: list[dict] = field(default_factory=list)
    response_json: dict[str, Any] | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status_code == 200

    @property
    def assertions_passed(self) -> int:
        return sum(1 for a in self.assertions if a["passed"])

    @property
    def assertions_total(self) -> int:
        return len(self.assertions)

    @property
    def is_off_topic(self) -> bool:
        return self.category in OFF_TOPIC_CATEGORIES


# ── Helpers ───────────────────────────────────────────────────────────────────


def _now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _preview(text: str, max_len: int = 160) -> str:
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= max_len else cleaned[: max_len - 3] + "..."


def _extract_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    with contextlib.suppress(json.JSONDecodeError):  # fall back to the first {...} block
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError("No JSON found in generator output")
    return json.loads(match.group(0))


def _join_url(base: str, path: str) -> str:
    b = base.rstrip("/")
    p = "/" + path.lstrip("/")
    if b.endswith("/v1") and p.startswith("/v1/"):
        p = p[3:]
    if b.endswith("/api") and p.startswith("/api/"):
        p = p[4:]
    return f"{b}{p}"


def _auth_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _mentions_company_or_index(question: str) -> bool:
    q = _normalize_question(question)
    if "ibex" in q:
        return True
    return any(re.search(rf"\b{re.escape(term)}\b", q) for term in COMPANY_TERMS)


def _count_companies_mentioned(question: str) -> int:
    q = _normalize_question(question)
    return sum(1 for term in COMPANY_TERMS if re.search(rf"\b{re.escape(term)}\b", q))


def _parse_duration(s: str) -> float:
    """Parse '5m', '30s', '1h', or plain seconds → float seconds."""
    s = s.strip()
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    if s.endswith("s"):
        return float(s[:-1])
    return float(s)


def _fmt_remaining(deadline: float | None) -> str:
    if deadline is None:
        return ""
    remaining = max(0.0, deadline - time.monotonic())
    mins, secs = divmod(int(remaining), 60)
    if mins:
        return f"  {DIM(f'({mins}m{secs:02d}s left)')}"
    return f"  {DIM(f'({secs}s left)')}"


def _percentile(sorted_data: list[float], p: float) -> float:
    if not sorted_data:
        return 0.0
    idx = int(len(sorted_data) * p)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


# ── Random category selection ─────────────────────────────────────────────────


def _pick_category(rng: random.Random, off_topic_ratio: float) -> str:
    """Return a category; off_topic_ratio is the probability of an off-topic one."""
    if rng.random() < off_topic_ratio:
        return rng.choice(OFF_TOPIC_CATEGORIES)
    return rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0]


# ── Question validation ───────────────────────────────────────────────────────


def _domain_problem(q_norm: str, category: str) -> str | None:
    """Why an in-topic question is outside the IBEX35 domain, if it is."""
    if any(term in q_norm for term in OFFTOPIC_TERMS):
        return "fuera de dominio (banca retail/personal)"
    if not _mentions_company_or_index(q_norm):
        return "no menciona empresa IBEX35 ni índice"
    if category == "comparativa" and _count_companies_mentioned(q_norm) < 2:
        return "comparativa sin al menos dos empresas"
    return None


def _route_problem(q_norm: str, category: str, expected_route: str | None) -> str | None:
    """Why the question does not fit its expected route, if it does not."""
    has_rag_hint = any(h in q_norm for h in RAG_HINTS)
    has_agent_hint = any(h in q_norm for h in AGENT_HINTS)
    if expected_route == "rag" and not has_rag_hint:
        return "esperado RAG pero faltan señales de fundamentales"
    if expected_route == "rag" and has_agent_hint and category not in {"ambigua", "seguimiento"}:
        return "esperado RAG pero incluye señales de mercado en tiempo real"
    if expected_route == "agent" and not has_agent_hint:
        return "esperado AGENT pero faltan señales de mercado en tiempo real"
    return None


def _validate_generated_question(
    *,
    question: str,
    category: str,
    expected_route: str | None,
    seen_questions: set[str],
) -> tuple[bool, str]:
    q_norm = _normalize_question(question)
    off_topic = category in OFF_TOPIC_CATEGORIES
    # Off-topic categories only need a non-trivial, new question.
    if len(q_norm) < (5 if off_topic else 10):
        return False, "pregunta demasiado corta"
    if q_norm in seen_questions:
        return False, "pregunta repetida"
    if off_topic:
        return True, "ok"
    problem = _domain_problem(q_norm, category) or _route_problem(q_norm, category, expected_route)
    return (False, problem) if problem else (True, "ok")


# ── Local question templates ──────────────────────────────────────────────────


def _company_label(term: str) -> str:
    special = {"bbva": "BBVA", "acs": "ACS", "iag": "IAG", "rovi": "ROVI", "aena": "AENA"}
    return special.get(term, term.upper())


def _build_local_improvised_question(
    *,
    category: str,
    expected_route: str | None,
    rng: random.Random,
) -> tuple[str, str, str | None]:
    c1, c2 = rng.sample(COMPANY_TERMS, 2)
    c1l = _company_label(c1)
    c2l = _company_label(c2)
    route = expected_route or (
        "agent"
        if category in {"mercado_tiempo_real", "mixta", "english_market", "tendencia"}
        else "rag"
    )

    rag_templates = [
        f"¿Qué EBITDA, ingresos y beneficio neto reportó {c1l} en 2024?",
        f"Resume la deuda neta, capex y flujo de caja de {c1l} en su último informe anual.",
        f"Compara el margen EBITDA y los ingresos de {c1l} frente a {c2l} en 2024.",
        f"¿Qué guidance y política de dividendo comunicó {c1l} en su último resultado?",
        f"En el IBEX35, compara fundamentales de {c1l} y {c2l}: ingresos, EBITDA y deuda.",
    ]

    agent_templates = [
        f"¿A cuánto cotiza {c1l} hoy y cuál es su variación intradía?",
        f"Dame el precio actual de {c1l}, su rango de 52 semanas y volumen reciente.",
        f"¿Cómo viene el trading de {c1l} esta semana en bolsa y qué tendencia muestra?",
        f"Combina para {c1l}: último EBITDA reportado y precio de mercado ahora.",
        f"Compare today market move: {c1l} vs {c2l}, stock price and intraday change.",
    ]

    off_topic_templates: dict[str, list[str]] = {
        "fuera_dominio": [
            "¿Cuánto me cobran de comisión en mi cuenta corriente del banco?",
            "¿Cómo puedo cancelar mi hipoteca anticipadamente?",
            "¿Qué depósito a plazo fijo me da más rentabilidad ahora mismo?",
            "Quiero hacer una transferencia internacional, ¿cuánto cuesta?",
        ],
        "empresa_no_ibex": [
            "¿Cuál es el precio actual de las acciones de Apple?",
            "¿Qué beneficio reportó Tesla en su último trimestre?",
            f"Compara los fundamentales de Amazon frente a {c1l}.",
            "¿Cómo va la cotización de NVIDIA hoy?",
        ],
        "noise": [
            "¿Cuál es el mejor equipo de fútbol de España?",
            "¿Qué tiempo hace hoy en Madrid?",
            "Recomiéndame una receta de paella valenciana.",
            "¿Cuánto mide la Torre Eiffel?",
        ],
    }

    category_overrides: dict[str, list[str]] = {
        "comparativa": [
            f"Compara ingresos, EBITDA y deuda neta de {c1l} y {c2l} en 2024.",
            f"Entre {c1l} y {c2l}, ¿quién mostró mejor margen y beneficio neto "
            "en el último reporte?",
        ],
        "english_market": [
            f"What is {c1l}'s stock price today and how is it trading intraday?",
            f"Compare {c1l} and {c2l}: current stock price, intraday move and 52-week range.",
        ],
        "english_fundamentals": [
            f"What EBITDA, revenue and net income did {c1l} report in 2024?",
            f"Compare fundamentals of {c1l} and {c2l}: revenue, EBITDA and net debt.",
        ],
        "seguimiento": [
            f"Siguiendo con {c1l}, ¿cómo evolucionaron su deuda neta y flujo de caja "
            "en el último año?",
            f"Sobre {c1l}, ¿qué cambió en margen EBITDA y capex entre 2023 y 2024?",
        ],
        "tendencia": [
            f"¿Qué tendencia de mercado muestra {c1l} hoy: precio actual, cambio diario "
            "y 52 semanas?",
            f"Analiza el momentum bursátil reciente de {c1l}: cotización ahora "
            "y desempeño semanal.",
        ],
    }

    # Off-topic categories
    if category in off_topic_templates:
        question = rng.choice(off_topic_templates[category])
        goal = LOCAL_IMPROVISED_GOALS["off_topic"]
        return question, goal, None

    templates = category_overrides.get(category)
    if not templates:
        templates = agent_templates if route == "agent" else rag_templates

    question = rng.choice(templates)
    goal = LOCAL_IMPROVISED_GOALS.get(route, "Valida routing del sistema.")
    return question, goal, route


def generate_local_improvised_question(
    *,
    category: str,
    expected_route: str | None,
    seen_questions: set[str],
    rng: random.Random,
) -> tuple[str, str, str | None]:
    last_reason = "sin motivo"
    for _ in range(20):
        question, goal, route = _build_local_improvised_question(
            category=category,
            expected_route=expected_route,
            rng=rng,
        )
        valid, reason = _validate_generated_question(
            question=question,
            category=category,
            expected_route=expected_route or route,
            seen_questions=seen_questions,
        )
        if valid:
            return question, goal, route
        last_reason = reason
    raise ValueError(f"no se pudo improvisar pregunta válida ({last_reason})")


# ── Question generation ───────────────────────────────────────────────────────


def _generator_prompt(
    category: str,
    round_index: int,
    rounds_total: int | None,
    history: list[RoundResult],
    expected_route: str | None,
    focus: str,
    feedback: str | None,
) -> str:
    recent = [f"- [{r.category}] {r.question} → route={r.route or '?'}" for r in history[-6:]]
    history_block = "\n".join(recent) if recent else "  (ninguna)"
    companies = ", ".join(COMPANY_TERMS[:20]) + ", ..."
    feedback_block = f"\nÚltimo intento rechazado: {feedback}\n" if feedback else ""
    round_label = f"{round_index}/{rounds_total}" if rounds_total else str(round_index)

    # Off-topic categories get a special prompt
    if category in OFF_TOPIC_CATEGORIES:
        return f"""Eres un tester QA. Genera UNA pregunta fuera del dominio IBEX35.
Categoría: {category} — {focus}
Escribe como usuario real. No menciones empresas del IBEX35.
Responde SOLO con JSON válido.

Formato:
{{
  "question": "pregunta aquí",
  "goal": "qué valida esta pregunta",
  "expected_route": null
}}
{feedback_block}
""".strip()

    return f"""Eres un tester QA de una API financiera del IBEX35 con auto-routing RAG/Agente.
Genera UNA pregunta realista para la categoría indicada.

Ronda: {round_label}
Categoría: {category}
Ruta esperada: {expected_route}
Objetivo de contenido: {focus}

Reglas:
- No repitas ni parafrasees las preguntas del historial.
- Escribe como lo haría un usuario final (natural, directo).
- Entre 10 y 200 caracteres.
- Debe estar ENFOCADA en IBEX35 (empresa o índice).
- Prohibido banca retail/personal: hipotecas personales, saldo de mi cuenta, \
depósitos o transferencias de usuario.
- Menciona al menos una empresa del universo IBEX35: {companies}
- Categoría "comparativa": menciona al menos dos empresas.
- Responde SOLO con JSON válido, sin texto extra.

Formato:
{{
  "question": "pregunta aquí",
  "goal": "qué valida esta pregunta",
  "expected_route": "rag" | "agent"
}}

Historial reciente:
{history_block}
{feedback_block}
""".strip()


# Generator backends to try, in order, for each --generator-backend value.
_BACKEND_ORDER = {
    "ollama": ["ollama", "openai"],
    "openai": ["openai", "ollama"],
    "auto": ["ollama", "openai"],
}


def _request_question(
    client: httpx.Client,
    backend: str,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    headers: dict[str, str],
) -> dict[str, Any]:
    """Ask one generator backend for a question and return its parsed JSON."""
    if backend == "ollama":
        resp = client.post(
            _join_url(base_url, "/api/generate"),
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": temperature},
            },
            headers=headers or None,
        )
        resp.raise_for_status()
        return _extract_json(resp.json().get("response", ""))

    resp = client.post(
        _join_url(base_url, "/v1/chat/completions"),
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        },
        headers=headers or None,
    )
    resp.raise_for_status()
    content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, list):
        content = " ".join(
            str(block.get("text", "")) for block in content if isinstance(block, dict)
        )
    return _extract_json(str(content))


def _should_try_next_backend(error: Exception, is_last: bool) -> bool:
    """Fall back on 'endpoint not found' or unparsable output, unless this was the last option."""
    if is_last:
        return False
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code if error.response is not None else None
        return status in {404, 405}
    return True


def generate_question(
    *,
    client: httpx.Client,
    ollama_url: str,
    model: str,
    category: str,
    round_index: int,
    rounds_total: int | None,
    history: list[RoundResult],
    temperature: float,
    expected_route: str | None,
    focus: str,
    feedback: str | None = None,
    backend: str = "auto",
    api_key: str | None = None,
) -> tuple[str, str, str | None, str]:
    prompt = _generator_prompt(
        category=category,
        round_index=round_index,
        rounds_total=rounds_total,
        history=history,
        expected_route=expected_route,
        focus=focus,
        feedback=feedback,
    )
    candidates = _BACKEND_ORDER.get(backend)
    if candidates is None:
        raise ValueError(f"backend no soportado: {backend}")

    for candidate in candidates:
        try:
            data = _request_question(
                client, candidate, ollama_url, model, prompt, temperature, _auth_headers(api_key)
            )
        except (httpx.HTTPStatusError, KeyError, TypeError, ValueError) as exc:
            if _should_try_next_backend(exc, is_last=candidate == candidates[-1]):
                continue
            raise
        question = str(data.get("question", "")).strip()
        if len(question) < 5:
            raise ValueError("Generator returned an empty question")
        return question, str(data.get("goal", "")).strip(), data.get("expected_route"), candidate

    raise RuntimeError("No se pudo obtener respuesta del generador")


def _probe_matches(backend: str, path: str, payload: Any) -> bool:
    """True if a probe response looks like the given generator backend."""
    if not isinstance(payload, dict):
        return False
    if backend == "ollama" and path == "/api/version":
        return "version" in payload
    if backend == "ollama" and path == "/api/tags":
        return isinstance(payload.get("models"), list)
    if backend == "openai":
        return isinstance(payload.get("data"), list)
    return False


def detect_generator_backend(
    *,
    client: httpx.Client,
    base_url: str,
    api_key: str | None = None,
) -> str:
    headers = _auth_headers(api_key)
    probes: list[tuple[str, str]] = [
        ("ollama", "/api/version"),
        ("openai", "/v1/models"),
        ("openai", "/models"),
        ("ollama", "/api/tags"),
    ]
    for backend, path in probes:
        try:
            resp = client.get(_join_url(base_url, path), headers=headers or None)
            if not (200 <= resp.status_code < 400):
                continue
            if _probe_matches(backend, path, resp.json()):
                return backend
        except httpx.HTTPError:
            continue
        except ValueError:
            continue
    raise RuntimeError(
        f"No se pudo detectar backend del generador en {base_url}. "
        "Prueba --generator-backend ollama u openai."
    )


# ── Question execution ────────────────────────────────────────────────────────


def execute_question(
    *,
    client: httpx.Client,
    api_url: str,
    endpoint: str,
    question: str,
    thread_id: str | None,
    company_filter: str | None,
) -> tuple[int, float, dict[str, Any] | None, str | None]:
    start = time.perf_counter()
    payload = {
        "question": question,
        **({"thread_id": thread_id} if thread_id else {}),
        **({"company_filter": company_filter} if company_filter else {}),
    }
    resp = client.post(
        f"{api_url}{endpoint}",
        json=payload,
    )
    latency = time.perf_counter() - start
    try:
        body = resp.json()
    except ValueError:
        body = None
    error = None
    if resp.status_code >= 400:
        error = str((body or {}).get("detail") or (body or {}).get("error") or resp.text[:300])
    return resp.status_code, latency, body, error


# ── Runner principal ──────────────────────────────────────────────────────────


def _color(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


OK = lambda s: _color(s, "32")  # verde   # noqa: E731
FAIL = lambda s: _color(s, "31")  # rojo    # noqa: E731
DIM = lambda s: _color(s, "2")  # gris    # noqa: E731
BOLD = lambda s: _color(s, "1")  # negrita # noqa: E731
CYAN = lambda s: _color(s, "36")  # cyan    # noqa: E731


# ── Run ───────────────────────────────────────────────────────────────────────

ROUTES = {"rag", "agent"}
# Categories whose answers must stick to the period asked for.
PERIOD_CATEGORIES = {"fundamentales", "comparativa", "english_fundamentals", "seguimiento"}


@dataclass
class _Question:
    text: str
    goal: str
    expected_route: str | None


@dataclass
class _RunState:
    """State shared by every round of one run."""

    args: argparse.Namespace
    client: httpx.Client
    rng: random.Random
    thread_id: str | None
    deadline: float | None
    manual_questions: list[str]
    generator_backend: str
    results: list[RoundResult] = field(default_factory=list)
    seen_questions: set[str] = field(default_factory=set)


def _merge_route(expected: str | None, generated: Any) -> str | None:
    """Keep the category's route; otherwise adopt a valid route proposed by the generator."""
    if expected is None and generated in ROUTES:
        return str(generated)
    return expected


# ── Set-up ────────────────────────────────────────────────────────────────────


def _deadline(args: argparse.Namespace) -> float | None:
    """Absolute deadline in duration mode, None in rounds mode. Raises ValueError if invalid."""
    if not args.duration:
        return None
    return time.monotonic() + _parse_duration(args.duration)


def _load_manual_questions(args: argparse.Namespace) -> list[str]:
    """Questions from --questions-file (sets args.rounds to their count). Empty without it."""
    if not args.questions_file:
        return []
    path = Path(args.questions_file)
    if not path.exists():
        raise FileNotFoundError(str(path))
    questions = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    args.rounds = len(questions)
    print(BOLD(f"Usando {args.rounds} preguntas de {path}"))
    return questions


def _print_header(
    args: argparse.Namespace,
    run_id: str,
    manual_questions: list[str],
    deadline: float | None,
    thread_id: str | None,
) -> None:
    print(BOLD(f"\nIBEX35 RAG — Agent Test  [{run_id}]"))
    print(f"API:       {args.api_url}{args.endpoint}")
    if not manual_questions:
        print(f"Generador: {args.model} @ {args.ollama_url}")
    print(
        f"Modo:      duración {args.duration}" if deadline else f"Modo:      {args.rounds} rondas"
    )
    if args.seed is not None:
        print(f"Seed:      {args.seed}")
    print(f"Off-topic: {args.off_topic_ratio * 100:.0f}%")
    if args.company_filter:
        print(f"Empresa:   {args.company_filter}")
    print(f"Thread ID: {thread_id}" if thread_id else "Thread ID: (none, modo Chainlit)")
    print()


def _resolve_generator_backend(
    args: argparse.Namespace, client: httpx.Client, manual_questions: list[str]
) -> str | None:
    """The question generator to use; None if it cannot be detected and there is no fallback."""
    backend: str = args.generator_backend
    if manual_questions:
        return backend
    if backend == "auto":
        try:
            backend = detect_generator_backend(
                client=client, base_url=args.ollama_url, api_key=args.generator_api_key
            )
        except Exception as exc:
            if not args.local_improvise_fallback:
                print(FAIL(f"No se pudo detectar backend de generador: {exc}"))
                return None
            backend = "local"
            print(DIM(f"Backend remoto no detectable ({exc}); usando improvisación local"))
    print(f"Backend:   {backend}\n")
    return backend


# ── Question generation ───────────────────────────────────────────────────────


def _round_category(state: _RunState) -> tuple[str, str | None, str]:
    """(category, expected route, content focus) for the next round."""
    if state.manual_questions:
        return "manual", None, "pregunta manual"
    category = _pick_category(state.rng, state.args.off_topic_ratio)
    rules = CATEGORY_RULES.get(category, {})
    expected_route = rules.get("expected_route")
    focus = str(rules.get("focus") or "pregunta financiera IBEX35")
    return category, expected_route if isinstance(expected_route, str) else None, focus


def _generate_once(
    state: _RunState,
    round_index: int,
    category: str,
    expected_route: str | None,
    focus: str,
    feedback: str | None,
) -> tuple[str, str, Any]:
    """One generation attempt with the current backend; the backend may switch."""
    if state.generator_backend == "local":
        return generate_local_improvised_question(
            category=category,
            expected_route=expected_route,
            seen_questions=state.seen_questions,
            rng=state.rng,
        )
    args = state.args
    question, goal, route, backend = generate_question(
        client=state.client,
        ollama_url=args.ollama_url,
        model=args.model,
        category=category,
        round_index=round_index,
        rounds_total=None if state.deadline else args.rounds,
        history=state.results,
        temperature=args.temperature,
        expected_route=expected_route,
        focus=focus,
        feedback=feedback,
        backend=state.generator_backend,
        api_key=args.generator_api_key,
    )
    if backend != state.generator_backend:
        print(DIM(f"      backend switch: {state.generator_backend} -> {backend}"))
        state.generator_backend = backend
    return question, goal, route


def _local_fallback(
    state: _RunState, category: str, expected_route: str | None, error: Exception
) -> _Question | None:
    """After a remote generator error, improvise locally if allowed (switches backend)."""
    if not state.args.local_improvise_fallback or state.generator_backend == "local":
        return None
    question, goal, route = generate_local_improvised_question(
        category=category,
        expected_route=expected_route,
        seen_questions=state.seen_questions,
        rng=state.rng,
    )
    print(DIM(f"      fallback a improvisación local ({error})"))
    state.generator_backend = "local"
    state.seen_questions.add(_normalize_question(question))
    return _Question(question, goal, _merge_route(expected_route, route))


def _record_generator_error(
    state: _RunState, round_index: int, category: str, message: str
) -> None:
    print(FAIL(f"[{round_index:02d}] generator_error: {message}"))
    state.results.append(
        RoundResult(
            round_index=round_index,
            category=category,
            question="",
            goal="",
            thread_id=state.thread_id,
            status_code=0,
            route=None,
            latency_seconds=0.0,
            answer_preview="",
            error=f"generator_error: {message}",
        )
    )


def _generate_valid_question(
    state: _RunState,
    round_index: int,
    category: str,
    expected_route: str | None,
    focus: str,
) -> _Question | None:
    """Generate until a question passes validation; None (error recorded) if attempts run out."""
    attempts = state.args.max_generation_attempts
    feedback: str | None = None
    for attempt in range(1, attempts + 1):
        label = f"intento {attempt}/{attempts}" if attempt > 1 else "generando pregunta"
        print(DIM(f"      ↳ [1/3] {label} ({state.generator_backend})..."), end="  ", flush=True)
        started = time.perf_counter()
        try:
            text, goal, route = _generate_once(
                state, round_index, category, expected_route, focus, feedback
            )
            print(DIM(f"{time.perf_counter() - started:.1f}s"), flush=True)
        except Exception as exc:
            print(DIM(f"{time.perf_counter() - started:.1f}s  ERROR"), flush=True)
            feedback = f"error al generar: {exc}"
            try:
                fallback = _local_fallback(state, category, expected_route, exc)
            except Exception as local_exc:
                fallback = None
                feedback = f"error local fallback: {local_exc}"
            if fallback:
                return fallback
            if attempt < attempts:
                continue
            _record_generator_error(state, round_index, category, str(exc))
            return None

        expected_route = _merge_route(expected_route, route)
        is_valid, reason = _validate_generated_question(
            question=text,
            category=category,
            expected_route=expected_route,
            seen_questions=state.seen_questions,
        )
        if is_valid:
            state.seen_questions.add(_normalize_question(text))
            return _Question(text, goal, expected_route)

        feedback = reason
        print(DIM(f"      ✗ pregunta inválida: {reason}"), flush=True)
        if state.args.verbose_generation:
            print(DIM(f"      retry({attempt}/{attempts}) -> {reason}"))

    message = f"no se pudo generar pregunta válida tras {attempts} intentos"
    _record_generator_error(state, round_index, category, message)
    return None


# ── Execution ─────────────────────────────────────────────────────────────────


def _round_assertions(
    body: dict[str, Any],
    latency: float,
    category: str,
    question: _Question,
    max_latency: float,
) -> list[Assertion]:
    """Assertions for a 200 response, in a fixed order."""
    off_topic = category in OFF_TOPIC_CATEGORIES
    assertions = [
        _assert_non_empty_answer(body),
        _assert_latency(latency, max_latency),
        _assert_sla_by_category(latency, category),
    ]
    if question.expected_route and not off_topic:
        assertions.append(_assert_route(body, question.expected_route))
    if body.get("route") == "rag" and not off_topic:
        assertions.append(_assert_has_sources(body))

    # Quality assertions (comparisons, periods, out-of-domain); None means not applicable.
    quality: list[Assertion | None] = []
    if category == "comparativa" or _count_companies_mentioned(question.text) >= 2:
        quality.append(_assert_both_companies_covered(body, question.text))
    if category in PERIOD_CATEGORIES:
        quality.append(_assert_period_adherence(body, question.text))
    if off_topic:
        quality.append(_assert_clean_abstention_off_topic(body, category))
    assertions.extend(a for a in quality if a)
    return assertions


def _print_round(
    state: _RunState,
    round_index: int,
    category: str,
    question: _Question,
    result: RoundResult,
    assertions: list[Assertion],
) -> None:
    passed = sum(1 for a in assertions if a.passed)
    route = result.route
    status_str = OK("200") if result.status_code == 200 else FAIL(f"{result.status_code}")
    route_str = (OK("agent") if route == "agent" else DIM("rag")) if route else DIM("-")
    assert_str = (OK if passed == len(assertions) else FAIL)(f"{passed}/{len(assertions)}")
    offtopic_tag = CYAN(" [off-topic]") if category in OFF_TOPIC_CATEGORIES else ""

    print(
        f"[{round_index:02d}] {category:<22} status={status_str} route={route_str:<6} "
        f"lat={result.latency_seconds:.1f}s  assert={assert_str}{offtopic_tag}"
        f"{_fmt_remaining(state.deadline)}"
    )
    print(f"      {DIM('Q:')} {question.text}")
    if question.goal:
        print(f"      {DIM('→')} {question.goal}")
    if result.tools_used:
        print(f"      {DIM('tools:')} {', '.join(result.tools_used)}  ({result.steps_taken} pasos)")
    if result.answer_preview:
        print(f"      {DIM('A:')} {result.answer_preview}")
    for a in assertions:
        print(f"      {OK('✓') if a.passed else FAIL('✗')} {a.name}  {DIM(a.detail)}")
    if result.error:
        print(f"      {FAIL('error:')} {result.error}")
    print()


def _execute_round(
    state: _RunState,
    round_index: int,
    category: str,
    question: _Question,
    round_started: float,
) -> None:
    args = state.args
    print(DIM(f"      ↳ [2/3] ejecutando POST {args.endpoint}..."), end="  ", flush=True)
    started = time.perf_counter()
    status, latency, body, error = execute_question(
        client=state.client,
        api_url=args.api_url,
        endpoint=args.endpoint,
        question=question.text,
        thread_id=state.thread_id,
        company_filter=args.company_filter,
    )
    print(DIM(f"{time.perf_counter() - started:.1f}s"), flush=True)
    body = body or {}

    print(DIM("      ↳ [3/3] evaluando assertions..."), end="  ", flush=True)
    started = time.perf_counter()
    assertions = (
        _round_assertions(body, latency, category, question, args.max_latency)
        if status == 200
        else []
    )
    passed = sum(1 for a in assertions if a.passed)
    print(DIM(f"{time.perf_counter() - started:.2f}s  ({passed}/{len(assertions)} ok)"), flush=True)
    print(DIM(f"      ↳ total ronda: {time.perf_counter() - round_started:.1f}s"), flush=True)

    answer = body.get("answer", "")
    result = RoundResult(
        round_index=round_index,
        category=category,
        question=question.text,
        goal=question.goal,
        thread_id=state.thread_id,
        status_code=status,
        route=body.get("route"),
        latency_seconds=latency,
        answer_preview=_preview(answer),
        tools_used=body.get("tools_used", []),
        steps_taken=body.get("steps_taken", 0),
        assertions=[asdict(a) for a in assertions],
        response_json=body if body else None,
        error=error,
    )
    _print_round(state, round_index, category, question, result, assertions)
    state.results.append(result)


def _out_of_rounds(state: _RunState, round_index: int) -> bool:
    if state.deadline is not None:
        return time.monotonic() >= state.deadline
    return round_index > state.args.rounds


def _pause(state: _RunState) -> None:
    # In duration mode, do not sleep past the deadline.
    delay = state.args.delay_seconds
    within_deadline = state.deadline is None or time.monotonic() + delay < state.deadline
    if delay > 0 and within_deadline:
        time.sleep(delay)


def _play_round(state: _RunState, round_index: int) -> bool:
    """Play one round. Returns False when the manual questions run out."""
    category, expected_route, focus = _round_category(state)
    round_started = time.perf_counter()
    rounds_label = f"{round_index}" if state.deadline else f"{round_index}/{state.args.rounds}"
    print(
        DIM(f"[{rounds_label}] categoria={category}  esperado={expected_route or '?'}"), flush=True
    )

    if state.manual_questions:
        if round_index > len(state.manual_questions):
            return False
        question: _Question | None = _Question(
            state.manual_questions[round_index - 1], "", expected_route
        )
    else:
        question = _generate_valid_question(state, round_index, category, expected_route, focus)
        if question is None:
            return True

    _execute_round(state, round_index, category, question, round_started)
    _pause(state)
    return True


def _run_rounds(state: _RunState) -> None:
    round_index = 1
    while not _out_of_rounds(state, round_index) and _play_round(state, round_index):
        round_index += 1


# ── Summary and report ────────────────────────────────────────────────────────


@dataclass
class _Summary:
    total: int
    ok_200: int
    errors: int
    in_topic_total: int
    off_topic_total: int
    routes: dict[str, int]
    route_coverage_ok: bool
    avg_latency: float
    p50: float
    p95: float
    p99: float
    wall_elapsed: float
    assertions_passed: int
    assertions_total: int
    by_category: dict[str, dict[str, Any]]

    @property
    def passed(self) -> bool:
        return (
            self.errors == 0
            and self.assertions_passed == self.assertions_total
            and self.route_coverage_ok
        )

    def as_report(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "ok_200": self.ok_200,
            "errors": self.errors,
            "in_topic_total": self.in_topic_total,
            "off_topic_total": self.off_topic_total,
            "routes": self.routes,
            "route_coverage_ok": self.route_coverage_ok,
            "avg_latency_seconds": round(self.avg_latency, 2),
            "latency_p50": round(self.p50, 2),
            "latency_p95": round(self.p95, 2),
            "latency_p99": round(self.p99, 2),
            "wall_elapsed_seconds": round(self.wall_elapsed, 1),
            "assertions_passed": self.assertions_passed,
            "assertions_total": self.assertions_total,
            "by_category": self.by_category,
        }


def _by_category(results: list[RoundResult]) -> dict[str, dict[str, Any]]:
    """Per-category totals, OK count, assertions and latency (OK rounds only)."""
    grouped: dict[str, list[RoundResult]] = {}
    for r in results:
        grouped.setdefault(r.category, []).append(r)

    summary: dict[str, dict[str, Any]] = {}
    for category, rounds in grouped.items():
        lats = sorted(r.latency_seconds for r in rounds if r.ok)
        summary[category] = {
            "total": len(rounds),
            "ok": sum(1 for r in rounds if r.ok),
            "assertions_passed": sum(r.assertions_passed for r in rounds),
            "assertions_total": sum(r.assertions_total for r in rounds),
            "avg_lat": round(statistics.mean(lats), 2) if lats else 0.0,
            "p95_lat": round(_percentile(lats, 0.95), 2) if lats else 0.0,
        }
    return summary


def _summarize(results: list[RoundResult], wall_elapsed: float) -> _Summary:
    ok_count = sum(1 for r in results if r.ok)
    routes: dict[str, int] = {}
    for r in results:
        if r.route:
            routes[r.route] = routes.get(r.route, 0) + 1
    lats_ok = sorted(r.latency_seconds for r in results if r.ok)
    off_topic_total = sum(1 for r in results if r.is_off_topic)
    return _Summary(
        total=len(results),
        ok_200=ok_count,
        errors=len(results) - ok_count,
        in_topic_total=len(results) - off_topic_total,
        off_topic_total=off_topic_total,
        routes=routes,
        route_coverage_ok=routes.get("rag", 0) > 0 and routes.get("agent", 0) > 0,
        avg_latency=statistics.mean(lats_ok) if lats_ok else 0.0,
        p50=statistics.median(lats_ok) if lats_ok else 0.0,
        p95=_percentile(lats_ok, 0.95),
        p99=_percentile(lats_ok, 0.99),
        wall_elapsed=wall_elapsed,
        assertions_passed=sum(r.assertions_passed for r in results),
        assertions_total=sum(r.assertions_total for r in results),
        by_category=_by_category(results),
    )


def _print_category_table(by_category: dict[str, dict[str, Any]]) -> None:
    print(f"\n  {'Categoría':<24} {'total':>5} {'ok':>4} {'asserts':>9} {'avg_lat':>8} {'p95':>6}")
    print(f"  {'─' * 24} {'─' * 5} {'─' * 4} {'─' * 9} {'─' * 8} {'─' * 6}")
    for cat, cs in sorted(by_category.items()):
        a_str = f"{cs['assertions_passed']}/{cs['assertions_total']}"
        cat_label = CYAN(f"{cat:<24}") if cat in OFF_TOPIC_CATEGORIES else f"{cat:<24}"
        print(
            f"  {cat_label} {cs['total']:>5} {cs['ok']:>4} {a_str:>9} "
            f"{cs['avg_lat']:>7.1f}s {cs['p95_lat']:>5.1f}s"
        )


def _print_summary(summary: _Summary, results: list[RoundResult]) -> None:
    print(BOLD("─" * 70))
    print(BOLD("Resumen"))
    print(
        f"  Total rondas  : {summary.total}  "
        f"({summary.in_topic_total} in-topic, {summary.off_topic_total} off-topic)"
    )
    print(f"  OK (200)      : {OK(str(summary.ok_200))}")
    print(f"  Errores       : {(FAIL if summary.errors else OK)(str(summary.errors))}")
    print(f"  Rutas         : {summary.routes}")
    print(f"  Cobertura     : {(OK if summary.route_coverage_ok else FAIL)('rag+agent')}")
    print(
        f"  Latencia avg  : {summary.avg_latency:.1f}s   "
        f"p50={summary.p50:.1f}s  p95={summary.p95:.1f}s  p99={summary.p99:.1f}s"
    )
    all_passed = summary.assertions_passed == summary.assertions_total
    print(
        "  Assertions    : "
        + (OK if all_passed else FAIL)(f"{summary.assertions_passed}/{summary.assertions_total}")
    )
    print(f"  Tiempo total  : {summary.wall_elapsed:.0f}s ({summary.wall_elapsed / 60:.1f}min)")

    off_topic = [r for r in results if r.is_off_topic]
    if off_topic:
        off_ok = sum(1 for r in off_topic if r.ok)
        print(
            f"\n  Off-topic     : {off_ok}/{len(off_topic)} OK  "
            f"(categorías: {', '.join(sorted({r.category for r in off_topic}))})"
        )
    if len(summary.by_category) > 1:
        _print_category_table(summary.by_category)


def _write_report(
    args: argparse.Namespace, run_id: str, state: _RunState, summary: _Summary
) -> Path:
    report = {
        "run_id": run_id,
        "generated_at_utc": _now_utc(),
        "config": {
            "api_url": args.api_url,
            "endpoint": args.endpoint,
            "ollama_url": args.ollama_url,
            "model": args.model,
            "generator_backend": None if state.manual_questions else state.generator_backend,
            "thread_id": state.thread_id,
            "rounds": None if state.deadline else args.rounds,
            "duration": args.duration or None,
            "seed": args.seed,
            "off_topic_ratio": args.off_topic_ratio,
            "delay_seconds": args.delay_seconds,
            "temperature": args.temperature,
            "max_latency": args.max_latency,
            "max_generation_attempts": args.max_generation_attempts,
            "local_improvise_fallback": args.local_improvise_fallback,
            "company_filter": args.company_filter,
        },
        "summary": summary.as_report(),
        "results": [asdict(r) for r in state.results],
    }
    output_path = Path(args.output or f"reports/agent_tests/agent_test_{run_id}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def run(args: argparse.Namespace) -> int:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Chainlit-compatible by default: no explicit thread_id in requests.
    thread_id = args.thread_id.strip() or None
    wall_start = time.monotonic()
    rng = random.Random(args.seed)  # set --seed for reproducible runs

    try:
        deadline = _deadline(args)
    except ValueError:
        print(FAIL(f"ERROR: --duration '{args.duration}' no es válido. Usa p.ej. 5m, 30s, 1h."))
        return 1
    try:
        manual_questions = _load_manual_questions(args)
    except FileNotFoundError as exc:
        print(FAIL(f"ERROR: fichero {exc} no encontrado"))
        return 1

    _print_header(args, run_id, manual_questions, deadline, thread_id)

    with httpx.Client(timeout=httpx.Timeout(args.timeout)) as client:
        backend = _resolve_generator_backend(args, client, manual_questions)
        if backend is None:
            return 1
        state = _RunState(
            args=args,
            client=client,
            rng=rng,
            thread_id=thread_id,
            deadline=deadline,
            manual_questions=manual_questions,
            generator_backend=backend,
        )
        _run_rounds(state)

    summary = _summarize(state.results, time.monotonic() - wall_start)
    _print_summary(summary, state.results)
    output_path = _write_report(args, run_id, state, summary)
    print(f"\n  Reporte       : {output_path}")
    return 0 if summary.passed else 1


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Agent test: genera y ejecuta preguntas contra la API RAG IBEX35.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--api-url", default=DEFAULT_API_URL, help="URL base de la API")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Endpoint a testear")
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="URL de Ollama (generador)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Modelo generador de preguntas")
    p.add_argument(
        "--generator-backend",
        default=DEFAULT_GENERATOR_BACKEND,
        choices=["auto", "ollama", "openai", "local"],
        help="Backend del generador de preguntas",
    )
    p.add_argument(
        "--generator-api-key",
        default=DEFAULT_GENERATOR_API_KEY,
        help="API key para backend OpenAI-compatible (si aplica)",
    )
    # Rounds vs duration mode (mutually exclusive in meaning, not in argparse)
    p.add_argument(
        "--rounds", type=int, default=10, help="Número de rondas (ignorado si --duration)"
    )
    p.add_argument(
        "--duration",
        default="",
        help="Duración total del test: '5m', '30s', '1h'. Si se especifica, ignora --rounds.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla RNG para reproducibilidad (None = aleatorio)",
    )
    p.add_argument(
        "--off-topic-ratio",
        type=float,
        default=0.05,
        help="Fracción de rondas con preguntas off-topic (0.0–1.0)",
    )
    p.add_argument("--delay-seconds", type=float, default=1.0, help="Pausa entre rondas (s)")
    p.add_argument("--timeout", type=float, default=180.0, help="Timeout HTTP (s)")
    p.add_argument("--temperature", type=float, default=0.7, help="Temperatura del generador")
    p.add_argument("--max-latency", type=float, default=60.0, help="Latencia máxima aceptable (s)")
    p.add_argument(
        "--max-generation-attempts",
        type=int,
        default=6,
        help="Reintentos para generar pregunta válida",
    )
    p.add_argument(
        "--local-improvise-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Si falla el generador remoto, improvisa preguntas localmente",
    )
    p.add_argument(
        "--verbose-generation", action="store_true", help="Mostrar motivos de rechazo del generador"
    )
    p.add_argument("--company-filter", default=None, help="Forzar filtro de empresa")
    p.add_argument(
        "--thread-id",
        default="",
        help="ID de sesión para memoria de agente. Vacío = comportamiento Chainlit por defecto.",
    )
    p.add_argument(
        "--questions-file", default=None, help="Fichero .txt con preguntas (una por línea)"
    )
    p.add_argument("--output", default="", help="Ruta del reporte JSON")
    return p.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except KeyboardInterrupt:
        print("\nInterrumpido")
        raise SystemExit(130) from None
