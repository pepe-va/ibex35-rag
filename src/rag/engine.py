"""RAG engine: hybrid search, cross-encoder reranking, coverage gate and generation."""

import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from itertools import chain, zip_longest
from typing import TYPE_CHECKING, Any

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from opentelemetry import trace as otel_trace
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.core.config import Settings
from src.core.logging import get_logger
from src.observability.metrics import EMPTY_RETRIEVAL, EVAL_ABSTENTION_RATIO, EVAL_STAGE_LATENCY
from src.rag.evaluation import (
    is_abstention,
    record_answer_metrics,
    record_retrieval_metrics,
    schedule_ragas_evaluation,
)
from src.rag.prompts import SYSTEM_PROMPT
from src.rag.retrieval import extract_filters, extract_query_intent, reorder_for_llm

if TYPE_CHECKING:
    from src.rag.vector_store.store import VectorStoreManager

logger = get_logger(__name__)
tracer = otel_trace.get_tracer(__name__)

# ── Retrieval constants ───────────────────────────────────────────────────────

# Candidates retrieved per company in multi-company searches (independent of top_k).
PER_COMPANY_K = 10
# Minimum chunks kept per requested company after reranking.
RERANK_MIN_PER_COMPANY = 3
# Minimum reranker score for a chunk to count as relevant.
MIN_RERANKER_SCORE = 0.05
# Relevant chunks needed for a company to count as covered.
MIN_CHUNKS_FOR_COVERAGE = 2
# Lower bound of the candidate pool for single-company or generic searches.
MIN_CANDIDATES = 10

# Identity of a chunk, used to deduplicate merged search results.
DocKey = tuple[str, str, int, int, str, str]

COMPANY_FILTER_KEYS = ("company", "companies")
PERIOD_FILTER_KEYS = ("fiscal_year", "fiscal_quarter")
CONTEXT_SEPARATOR = "\n\n---\n\n"


# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class QueryResult:
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    latency_seconds: float = 0.0
    num_retrieved: int = 0
    from_cache: bool = False
    query: str = ""


@dataclass
class CoverageReport:
    """How well the retrieved chunks cover the request, checked before calling the LLM."""

    requested: list[str] = field(default_factory=list)
    covered: list[str] = field(default_factory=list)  # >= MIN_CHUNKS_FOR_COVERAGE relevant chunks
    missing: list[str] = field(default_factory=list)  # no relevant chunks
    partial: list[str] = field(default_factory=list)  # some relevant chunks, but too few
    period_ok: bool = True  # the requested year is present in the chunks
    min_score_ok: bool = True  # at least one chunk is relevant
    available_periods: list[str] = field(default_factory=list)  # years found in the chunks

    @property
    def is_sufficient(self) -> bool:
        """True if there is enough coverage to generate without abstaining."""
        return not self.missing and self.min_score_ok


@dataclass
class AbstentionDecision:
    abstain: bool
    reason: str = ""


# ── Coverage gate ─────────────────────────────────────────────────────────────


def _score(doc: Document) -> float:
    return float(doc.metadata.get("_score", 0.0))


def _company(doc: Document) -> str:
    return str(doc.metadata.get("company", "")).upper()


def _has_relevant_chunk(documents: list[Document]) -> bool:
    return any(_score(doc) >= MIN_RERANKER_SCORE for doc in documents)


def check_coverage(
    documents: list[Document],
    requested_companies: list[str],
    requested_year: str | None,
) -> CoverageReport:
    """Classify each requested company as covered, partial or missing, and check the period."""
    report = CoverageReport(
        requested=list(requested_companies),
        min_score_ok=_has_relevant_chunk(documents),
    )
    if not requested_companies:
        # No specific companies were asked for: the strict gate does not apply.
        return report

    for company in requested_companies:
        relevant = sum(
            _company(doc) == company and _score(doc) >= MIN_RERANKER_SCORE for doc in documents
        )
        if relevant == 0:
            report.missing.append(company)
        elif relevant < MIN_CHUNKS_FOR_COVERAGE:
            report.partial.append(company)
        else:
            report.covered.append(company)

    if requested_year:
        years = {
            str(doc.metadata["fiscal_year"]) for doc in documents if doc.metadata.get("fiscal_year")
        }
        report.available_periods = sorted(years)
        report.period_ok = requested_year in years

    return report


def decide_abstention(
    coverage: CoverageReport,
    is_comparison: bool,
    requires_specific_period: bool,
) -> AbstentionDecision:
    """Decide whether to answer without calling the LLM."""
    # Gate 1: a compared company has no data at all.
    if coverage.missing and is_comparison:
        missing = ", ".join(coverage.missing)
        covered = ", ".join(coverage.covered) if coverage.covered else "ninguna"
        return AbstentionDecision(
            abstain=True,
            reason=(
                f"No dispongo de datos suficientes de {missing} para realizar esta comparativa. "
                f"Solo tengo información de: {covered}."
            ),
        )

    # Gate 2: no chunk reaches the minimum relevance.
    if not coverage.min_score_ok and coverage.requested:
        companies = ", ".join(coverage.requested)
        return AbstentionDecision(
            abstain=True,
            reason=(
                f"Las fuentes recuperadas para {companies} tienen relevancia muy baja "
                f"y no son suficientes para responder con fiabilidad."
            ),
        )

    # Gate 3: the requested period is not available, but others are.
    if requires_specific_period and not coverage.period_ok and coverage.available_periods:
        available = ", ".join(coverage.available_periods)
        return AbstentionDecision(
            abstain=True,
            reason=(
                f"No tengo datos del periodo solicitado. "
                f"Los periodos disponibles en mi base de conocimiento son: {available}."
            ),
        )

    return AbstentionDecision(abstain=False)


# ── Search helpers ────────────────────────────────────────────────────────────


def _to_qdrant_filter(filters: dict[str, Any]) -> Filter | None:
    if not filters:
        return None
    return Filter(
        must=[
            FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
            for key, value in filters.items()
        ]
    )


def _without_period(filters: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in filters.items() if key not in PERIOD_FILTER_KEYS}


def _has_period(filters: dict[str, Any]) -> bool:
    return any(key in filters for key in PERIOD_FILTER_KEYS)


def _requested_companies(filters: dict[str, Any]) -> list[str]:
    if isinstance(filters.get("companies"), list):
        return [str(company).upper() for company in filters["companies"] if company]
    if filters.get("company"):
        return [str(filters["company"]).upper()]
    return []


def _doc_key(doc: Document) -> DocKey:
    meta = doc.metadata or {}
    return (
        str(meta.get("company", "")),
        str(meta.get("source_file", "")),
        int(meta.get("page") or 0),
        int(meta.get("chunk_index") or 0),
        str(meta.get("content_type", "")),
        doc.page_content[:128],
    )


def _merge_interleaved(doc_lists: list[list[Document]], top_k: int) -> list[Document]:
    """Round-robin merge of ranked lists, without duplicates, up to top_k documents."""
    merged: list[Document] = []
    seen: set[DocKey] = set()
    for doc in chain.from_iterable(zip_longest(*doc_lists)):
        if doc is None:
            continue
        key = _doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
        if len(merged) >= top_k:
            break
    return merged


def _with_company_floor(
    reranked: list[tuple[float, Document]],
    companies: list[str],
    top_k: int,
) -> list[tuple[float, Document]]:
    """Keep the best RERANK_MIN_PER_COMPANY chunks of each company, then fill by score."""
    selected: list[tuple[float, Document]] = []
    remaining: list[tuple[float, Document]] = []
    counts = dict.fromkeys(companies, 0)
    for score, doc in reranked:
        company = _company(doc)
        if company in counts and counts[company] < RERANK_MIN_PER_COMPANY:
            selected.append((score, doc))
            counts[company] += 1
        else:
            remaining.append((score, doc))

    selected.extend(remaining[: max(0, top_k - len(selected))])
    selected.sort(key=lambda pair: pair[0], reverse=True)
    return selected[:top_k]


# ── Prompt and response helpers ───────────────────────────────────────────────


def _source(doc: Document) -> dict[str, Any]:
    return {
        "company": doc.metadata.get("company", "UNKNOWN"),
        "ticker": doc.metadata.get("ticker", "UNKNOWN"),
        "content_type": doc.metadata.get("content_type", "text"),
        "page": doc.metadata.get("page"),
        "score": round(_score(doc), 4),
    }


def _build_context(documents: list[Document]) -> str:
    return CONTEXT_SEPARATOR.join(doc.page_content for doc in reorder_for_llm(documents))


def _build_messages(context: str, question: str, coverage_block: str = "") -> list[tuple[str, str]]:
    return [
        ("system", SYSTEM_PROMPT),
        ("human", f"Contexto:\n{context}\n\nPregunta: {question}\n{coverage_block}\nRespuesta:"),
    ]


def _coverage_block(
    requested: list[str],
    documents: list[Document],
    coverage: CoverageReport,
    requested_year: str | None,
) -> str:
    """Tell the LLM which companies and periods the context does and does not cover."""
    block = ""
    if requested:
        present = sorted({_company(doc) for doc in documents if doc.metadata.get("company")})
        missing = [company for company in requested if company not in present]
        block = (
            "\nCobertura de empresas solicitadas:\n"
            f"- Solicitadas: {', '.join(requested)}\n"
            f"- Presentes en contexto: {', '.join(present) if present else 'ninguna'}\n"
            f"- Faltantes: {', '.join(missing) if missing else 'ninguna'}\n"
            "Regla obligatoria: no inventes datos de empresas faltantes; "
            "indica explícitamente que faltan datos y limita la comparación a lo disponible.\n"
        )
    if requested_year and not coverage.period_ok and coverage.available_periods:
        block += (
            f"\nADVERTENCIA DE PERIODO: se solicitó {requested_year} pero "
            f"los documentos recuperados corresponden a: {', '.join(coverage.available_periods)}. "
            f"Indica explícitamente el periodo real de cada cifra.\n"
        )
    return block


def _out_of_domain_answer(companies: list[str]) -> str:
    return (
        f"La consulta hace referencia a {', '.join(companies)}, "
        f"que no forma parte del universo de empresas IBEX35 cubiertas por este sistema. "
        f"Solo puedo responder sobre empresas del IBEX35."
    )


def _text(content: Any) -> str:
    return content if isinstance(content, str) else str(content)


def _step_end(step: str, started_at: float, **extra: Any) -> dict[str, Any]:
    duration = round(time.perf_counter() - started_at, 2)
    return {"type": "step_end", "step": step, "duration_seconds": duration, **extra}


@contextmanager
def _stage(name: str) -> Iterator[otel_trace.Span]:
    """Trace a pipeline stage as span `rag.<name>` and record its latency."""
    started_at = time.perf_counter()
    with tracer.start_as_current_span(f"rag.{name}") as span:
        yield span
    EVAL_STAGE_LATENCY.labels(stage=name).observe(time.perf_counter() - started_at)


# ── Engine ────────────────────────────────────────────────────────────────────


class RAGEngine:
    """Hybrid retrieval + cross-encoder reranking + Ollama generation."""

    def __init__(self, settings: Settings, vector_store: "VectorStoreManager") -> None:
        self.settings = settings
        self._vs = vector_store
        self._llm = ChatOllama(
            model=settings.ollama_llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            client_kwargs={"timeout": settings.llm_request_timeout},
        )
        device = _best_device()
        logger.info("reranker_device", device=device)
        self._reranker = HuggingFaceCrossEncoder(
            model_name=settings.reranker_model,
            model_kwargs={"device": device},
        )

    @property
    def _candidate_pool_size(self) -> int:
        return max(self.settings.similarity_top_k * 2, MIN_CANDIDATES)

    # ── Hybrid search ─────────────────────────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        company_filter: str | None = None,
    ) -> list[Document]:
        """Dense + BM25 search with metadata filters extracted from the query.

        Multi-company questions search each company in parallel with
        PER_COMPANY_K candidates, so no company is crowded out of the merge.
        """
        filters = self._resolve_filters(query, company_filter)
        companies = _requested_companies(filters)
        base_filters = {
            key: value for key, value in filters.items() if key not in COMPANY_FILTER_KEYS
        }
        if len(companies) > 1:
            return self._search_each_company(query, companies, base_filters, filters)
        return self._search_one(query, k, companies, base_filters, filters)

    def _resolve_filters(self, query: str, company_filter: str | None) -> dict[str, Any]:
        if company_filter:
            return {"company": company_filter.upper()}
        started_at = time.perf_counter()
        filters = extract_filters(query)
        EVAL_STAGE_LATENCY.labels(stage="extract_filters").observe(time.perf_counter() - started_at)
        return filters

    def _search(self, query: str, filters: dict[str, Any], k: int) -> list[Document]:
        return self._vs.store.similarity_search(query=query, k=k, filter=_to_qdrant_filter(filters))

    def _search_company(
        self, query: str, company: str, base_filters: dict[str, Any]
    ) -> list[Document]:
        """Search one company; drop the period filters if they return nothing."""
        scoped = {**base_filters, "company": company}
        results = self._search(query, scoped, PER_COMPANY_K)
        if results or not _has_period(scoped):
            return results
        results = self._search(query, _without_period(scoped), PER_COMPANY_K)
        logger.info(
            "hybrid_search_relaxed_temporal_filters",
            company=company,
            original_filters=scoped,
            results=len(results),
        )
        return results

    def _search_each_company(
        self,
        query: str,
        companies: list[str],
        base_filters: dict[str, Any],
        filters: dict[str, Any],
    ) -> list[Document]:
        search = partial(self._search_company, query, base_filters=base_filters)
        # Qdrant client calls are thread-safe.
        with ThreadPoolExecutor(
            max_workers=len(companies), thread_name_prefix="rag_search"
        ) as pool:
            per_company = list(pool.map(search, companies))

        missing = [
            company for company, docs in zip(companies, per_company, strict=True) if not docs
        ]
        results = _merge_interleaved(
            [docs for docs in per_company if docs], PER_COMPANY_K * len(companies)
        )
        logger.info(
            "hybrid_search",
            query=query[:60],
            filters=filters,
            requested_companies=companies,
            missing_companies=missing,
            results=len(results),
        )
        return results

    def _search_one(
        self,
        query: str,
        k: int,
        companies: list[str],
        base_filters: dict[str, Any],
        filters: dict[str, Any],
    ) -> list[Document]:
        scoped = dict(base_filters)
        if companies:
            scoped["company"] = companies[0]
        results = self._search(query, scoped, k)

        if not results and _has_period(scoped):
            relaxed = _without_period(scoped)
            results = self._search(query, relaxed, k)
            logger.info(
                "hybrid_search_relaxed_temporal_filters",
                original_filters=scoped,
                relaxed_filters=relaxed,
                results=len(results),
            )

        # Generic questions (no company) fall back to an unfiltered search.
        if not results and scoped and not companies:
            results = self._search(query, {}, k)
            logger.info(
                "hybrid_search_unfiltered_fallback", original_filters=scoped, results=len(results)
            )

        logger.info("hybrid_search", query=query[:60], filters=filters, results=len(results))
        return results

    # ── Reranking ─────────────────────────────────────────────────────────────

    def rerank_results(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
        requested_companies: list[str] | None = None,
    ) -> list[Document]:
        """Rerank with the cross-encoder and keep top_k chunks.

        With two or more requested companies, each keeps at least
        RERANK_MIN_PER_COMPANY chunks before the rest of the budget is filled.
        """
        if not documents:
            return []
        scores = self._reranker.score([(query, doc.page_content) for doc in documents])
        reranked = sorted(
            zip(scores, documents, strict=True), key=lambda pair: pair[0], reverse=True
        )
        if requested_companies and len(requested_companies) > 1:
            top = _with_company_floor(reranked, requested_companies, top_k)
        else:
            top = reranked[:top_k]

        for score, doc in top:
            doc.metadata["_score"] = float(score)
        return [doc for _, doc in top]

    def _requested_companies(self, question: str, company_filter: str | None) -> list[str]:
        """The explicit filter wins over the companies mentioned in the question."""
        if company_filter:
            return [company_filter.upper()]
        return extract_query_intent(question).companies

    def _retrieve(
        self, question: str, company_filter: str | None, requested: list[str]
    ) -> list[Document]:
        with _stage("retrieve") as span:
            candidates = self.hybrid_search(
                question, k=self._candidate_pool_size, company_filter=company_filter
            )
            span.set_attribute("candidates", len(candidates))
        with _stage("rerank") as span:
            documents = self.rerank_results(
                question,
                candidates,
                top_k=self.settings.rerank_top_n,
                requested_companies=requested,
            )
            span.set_attribute("top_k", len(documents))
        return documents

    def _generate(self, context: str, question: str, coverage_block: str) -> str:
        with _stage("generate") as span:
            response = self._llm.invoke(_build_messages(context, question, coverage_block))
            answer = _text(response.content)
            span.set_attribute("answer_len", len(answer))
        return answer

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, question: str, company_filter: str | None = None) -> QueryResult:
        """Hybrid search → rerank → coverage gate → generation with Ollama."""
        with tracer.start_as_current_span("rag.query") as span:
            span.set_attribute("question", question[:200])
            if company_filter:
                span.set_attribute("company_filter", company_filter)
            try:
                return self._answer(question, company_filter, span)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(otel_trace.StatusCode.ERROR, str(exc))
                logger.exception("query_failed", error=str(exc))
                raise

    def _answer(
        self, question: str, company_filter: str | None, span: otel_trace.Span
    ) -> QueryResult:
        started_at = time.perf_counter()
        intent = extract_query_intent(question)

        # Questions only about non-IBEX35 companies never reach the index.
        if intent.has_out_of_domain and not intent.companies:
            EVAL_ABSTENTION_RATIO.observe(1.0)
            return QueryResult(
                answer=_out_of_domain_answer(intent.out_of_domain_companies),
                latency_seconds=time.perf_counter() - started_at,
                query=question,
            )

        requested = [company_filter.upper()] if company_filter else intent.companies
        documents = self._retrieve(question, company_filter, requested)
        if not documents:
            EMPTY_RETRIEVAL.inc()
            span.set_attribute("empty_retrieval", True)
        context = _build_context(documents)
        record_retrieval_metrics(question, documents, requested, context)

        coverage = check_coverage(documents, requested, intent.fiscal_year)
        decision = decide_abstention(
            coverage,
            is_comparison=intent.is_comparison,
            requires_specific_period=intent.requires_specific_period,
        )
        if decision.abstain:
            return _abstention_result(question, documents, coverage, decision, span, started_at)

        coverage_block = _coverage_block(requested, documents, coverage, intent.fiscal_year)
        answer = self._generate(context, question, coverage_block)
        abstained = is_abstention(answer)
        EVAL_ABSTENTION_RATIO.observe(1.0 if abstained else 0.0)
        if abstained:
            span.set_attribute("abstention", True)
        record_answer_metrics(question, answer, documents, context, self.settings)

        latency = time.perf_counter() - started_at
        span.set_attribute("latency_seconds", round(latency, 3))
        span.set_attribute("num_sources", len(documents))
        sources = [_source(doc) for doc in documents]
        logger.info(
            "query_complete",
            latency=round(latency, 3),
            sources=len(sources),
            company_filter=company_filter,
            coverage_missing=coverage.missing,
            period_ok=coverage.period_ok,
        )
        schedule_ragas_evaluation(
            question, answer, [doc.page_content for doc in documents], self.settings
        )
        return QueryResult(
            answer=answer,
            sources=sources,
            latency_seconds=latency,
            num_retrieved=len(sources),
            query=question,
        )

    # ── Streaming query ───────────────────────────────────────────────────────

    def stream_query(
        self, question: str, company_filter: str | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield step_start/step_end per stage, a token event per LLM chunk, then done.

        Unlike `query`, streaming skips the out-of-domain check and the coverage gate.
        """
        try:
            yield from self._stream_events(question, company_filter)
        except Exception as exc:
            logger.exception("stream_query_failed", error=str(exc))
            yield {"type": "error", "error": str(exc)}

    def _stream_events(self, question: str, company_filter: str | None) -> Iterator[dict[str, Any]]:
        started_at = time.perf_counter()

        step_started = time.perf_counter()
        yield {"type": "step_start", "step": "retrieve", "label": "Buscando en Qdrant"}
        candidates = self.hybrid_search(
            question, k=self._candidate_pool_size, company_filter=company_filter
        )
        yield _step_end("retrieve", step_started)

        step_started = time.perf_counter()
        yield {"type": "step_start", "step": "rerank", "label": "Rerankeando resultados"}
        documents = self.rerank_results(
            question,
            candidates,
            top_k=self.settings.rerank_top_n,
            requested_companies=self._requested_companies(question, company_filter),
        )
        sources = [_source(doc) for doc in documents]
        yield _step_end("rerank", step_started, sources=sources)

        step_started = time.perf_counter()
        yield {"type": "step_start", "step": "generate", "label": "Generando respuesta"}
        tokens: list[str] = []
        for chunk in self._llm.stream(_build_messages(_build_context(documents), question)):
            token = _text(chunk.content)
            if token:
                tokens.append(token)
                yield {"type": "token", "content": token}
        yield _step_end("generate", step_started)

        yield {
            "type": "done",
            "answer": "".join(tokens),
            "latency_seconds": round(time.perf_counter() - started_at, 2),
            "sources": sources,
            "query": question,
        }


def _best_device() -> str:
    # torch is heavy; import it only when an engine is actually built.
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _abstention_result(
    question: str,
    documents: list[Document],
    coverage: CoverageReport,
    decision: AbstentionDecision,
    span: otel_trace.Span,
    started_at: float,
) -> QueryResult:
    """Answer with the coverage-gate reason, keeping the sources that were found."""
    span.set_attribute("abstention", True)
    span.set_attribute("abstention_reason", decision.reason[:200])
    EVAL_ABSTENTION_RATIO.observe(1.0)
    logger.info(
        "coverage_gate_abstention",
        reason=decision.reason[:200],
        missing=coverage.missing,
        covered=coverage.covered,
    )
    return QueryResult(
        answer=decision.reason,
        sources=[_source(doc) for doc in documents],
        latency_seconds=time.perf_counter() - started_at,
        num_retrieved=len(documents),
        query=question,
    )
