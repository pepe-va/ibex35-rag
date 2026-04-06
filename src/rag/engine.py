"""RAG engine: hybrid search + reranking + Ollama generation (notebook 07)."""

import re
import threading
import time
from dataclasses import dataclass, field

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from opentelemetry import trace as otel_trace
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.config import Settings
from src.logging_config import get_logger
from src.monitoring.metrics import (
    CONTEXT_BYTES,
    EMPTY_RETRIEVAL,
    EVAL_ABSTENTION,
    EVAL_ANSWER_RELEVANCY,
    EVAL_CHUNK_UTILIZATION,
    EVAL_FAITHFULNESS,
    EVAL_STAGE_LATENCY,
    EVAL_TOKEN_IOU,
    RETRIEVED_CHUNKS,
    RETRIEVAL_SCORE,
)
from src.rag.prompts import SYSTEM_PROMPT
from src.rag.schema import ChunkMetadata

logger = get_logger(__name__)
tracer = otel_trace.get_tracer(__name__)

RERANKER_MODEL = "BAAI/bge-reranker-base"

# ── IBEX35 company alias map (regex → canonical name) ─────────────────────────

_COMPANY_PATTERNS: list[tuple[re.Pattern, str]] = [
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
    (re.compile(r"\brovi\b", re.I), "ROVI"),
    (re.compile(r"\bsacyr\b", re.I), "SACYR"),
    (re.compile(r"\bsolaria\b", re.I), "SOLARIA"),
    (re.compile(r"\bunicaja\b", re.I), "UNICAJA"),
]

_GENERAL_MARKET = re.compile(r"\bibex\b|\bmercado espa[nñ]ol\b|\btodas las empresas\b", re.I)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_QUARTER_RE = re.compile(r"\b(q[1-4]|trimestre\s*[1-4])\b", re.I)


# ── RAGAS evaluation (background) ─────────────────────────────────────────────

def _run_ragas_evaluation(
    question: str,
    answer: str,
    contexts: list[str],
    settings: Settings,
) -> None:
    try:
        from langchain_ollama import ChatOllama as _ChatOllama
        from langchain_ollama import OllamaEmbeddings as _OllamaEmbeddings
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import AnswerRelevancy, Faithfulness

        llm_wrapper = LangchainLLMWrapper(
            _ChatOllama(model=settings.ollama_llm_model, base_url=settings.ollama_base_url, temperature=0.0)
        )
        embed_wrapper = LangchainEmbeddingsWrapper(
            _OllamaEmbeddings(model=settings.ollama_embed_model, base_url=settings.ollama_base_url)
        )
        sample = SingleTurnSample(user_input=question, response=answer, retrieved_contexts=contexts)
        result = evaluate(
            dataset=EvaluationDataset(samples=[sample]),
            metrics=[Faithfulness(), AnswerRelevancy()],
            llm=llm_wrapper,
            embeddings=embed_wrapper,
        )
        row = result.to_pandas().iloc[0].to_dict()
        faith = float(row.get("faithfulness") or 0)
        relevancy = float(row.get("answer_relevancy") or 0)
        EVAL_FAITHFULNESS.observe(faith)
        EVAL_ANSWER_RELEVANCY.observe(relevancy)
        logger.info(
            "ragas_evaluation",
            faithfulness=round(faith, 3),
            answer_relevancy=round(relevancy, 3),
        )
    except Exception as exc:
        logger.warning("ragas_evaluation_failed", error=str(exc))


# ── QueryResult ────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    latency_seconds: float = 0.0
    num_retrieved: int = 0
    from_cache: bool = False
    query: str = ""

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "latency_seconds": round(self.latency_seconds, 3),
            "num_retrieved": self.num_retrieved,
            "from_cache": self.from_cache,
            "query": self.query,
        }


# ── RAGEngine ──────────────────────────────────────────────────────────────────

class RAGEngine:
    """Hybrid retrieval + cross-encoder reranking + Ollama generation."""

    def __init__(self, settings: Settings, vector_store: "VectorStoreManager") -> None:  # type: ignore[name-defined]  # noqa: F821
        self.settings = settings
        self._vs = vector_store
        self._llm = ChatOllama(
            model=settings.ollama_llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
        )
        self._reranker = HuggingFaceCrossEncoder(
            model_name=RERANKER_MODEL,
            model_kwargs={"device": "cpu"},
        )

    # ── 07: filter extraction ─────────────────────────────────────────────────

    def extract_filters(self, user_query: str) -> dict:
        """
        Extract Qdrant metadata filters from a natural-language query using regex.

        Returns a dict with only the matched fields, e.g.:
          {"company": "IBERDROLA", "fiscal_year": "2024"}
        """
        filters: dict = {}

        # General market queries → no company filter
        if _GENERAL_MARKET.search(user_query):
            return filters

        # Company
        for pattern, canonical in _COMPANY_PATTERNS:
            if pattern.search(user_query):
                filters["company"] = canonical
                break

        # Fiscal year
        m = _YEAR_RE.search(user_query)
        if m:
            filters["fiscal_year"] = m.group(1)

        # Fiscal quarter
        m = _QUARTER_RE.search(user_query)
        if m:
            raw_q = m.group(1).lower()
            if raw_q.startswith("q"):
                filters["fiscal_quarter"] = raw_q.upper()
            else:
                # "trimestre 2" → "Q2"
                num = re.search(r"[1-4]", raw_q)
                if num:
                    filters["fiscal_quarter"] = f"Q{num.group()}"

        return filters

    # ── 07: hybrid search ─────────────────────────────────────────────────────

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        company_filter: str | None = None,
    ) -> list[Document]:
        """
        Perform hybrid search (dense + sparse BM25) with LLM-extracted filters.

        If company_filter is provided it overrides the LLM-extracted company.
        """
        store = self._vs._get_store()

        if company_filter:
            filters = {"company": company_filter.upper()}
        else:
            t0 = time.perf_counter()
            filters = self.extract_filters(query)
            EVAL_STAGE_LATENCY.labels(stage="extract_filters").observe(time.perf_counter() - t0)

        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
                for key, value in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        results = store.similarity_search(query=query, k=k, filter=qdrant_filter)
        logger.info("hybrid_search", query=query[:60], filters=filters, results=len(results))
        return results

    # ── 07: reranking ─────────────────────────────────────────────────────────

    def rerank_results(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        """
        Rerank documents with a cross-encoder (BAAI/bge-reranker-base).

        Args:
            query: The search query.
            documents: Candidate documents from hybrid_search.
            top_k: Number of top results to return.

        Returns:
            List of top_k Document objects sorted by cross-encoder score.
        """
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._reranker.score(pairs)

        reranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        top = reranked[:top_k]
        for score, doc in top:
            doc.metadata["_score"] = float(score)
        return [doc for _, doc in top]

    # ── Query (hybrid + rerank + generate) ───────────────────────────────────

    def query(self, question: str, company_filter: str | None = None) -> QueryResult:
        """Execute hybrid search → rerank → generate with Ollama."""
        start = time.perf_counter()

        with tracer.start_as_current_span("rag.query") as span:
            span.set_attribute("question", question[:200])
            if company_filter:
                span.set_attribute("company_filter", company_filter)

            try:
                # 1. extract_filters + hybrid_search (stage: retrieve)
                t0 = time.perf_counter()
                with tracer.start_as_current_span("rag.retrieve") as s:
                    candidates = self.hybrid_search(
                        question,
                        k=max(self.settings.similarity_top_k * 2, 10),
                        company_filter=company_filter,
                    )
                    s.set_attribute("candidates", len(candidates))
                EVAL_STAGE_LATENCY.labels(stage="retrieve").observe(time.perf_counter() - t0)

                # 2. Rerank (stage: rerank)
                t0 = time.perf_counter()
                with tracer.start_as_current_span("rag.rerank") as s:
                    documents = self.rerank_results(
                        question, candidates, top_k=self.settings.rerank_top_n
                    )
                    s.set_attribute("top_k", len(documents))
                EVAL_STAGE_LATENCY.labels(stage="rerank").observe(time.perf_counter() - t0)

                # 3. Retrieval metrics
                RETRIEVED_CHUNKS.observe(len(documents))
                if not documents:
                    EMPTY_RETRIEVAL.inc()
                    span.set_attribute("empty_retrieval", True)
                for doc in documents:
                    score = doc.metadata.get("_score", 0.0)
                    if score:
                        RETRIEVAL_SCORE.observe(float(score))

                # 4a. Chunk utilization: fraction of chunks with positive reranker score
                if documents:
                    positive = sum(1 for d in documents if d.metadata.get("_score", 0.0) > 0)
                    EVAL_CHUNK_UTILIZATION.observe(positive / len(documents))

                # 4b. Token IoU: overlap between query tokens and context tokens
                q_tokens = set(question.lower().split())
                context = "\n\n---\n\n".join(doc.page_content for doc in documents)
                ctx_tokens = set(context.lower().split())
                if q_tokens and ctx_tokens:
                    iou = len(q_tokens & ctx_tokens) / len(q_tokens | ctx_tokens)
                    EVAL_TOKEN_IOU.observe(iou)

                CONTEXT_BYTES.observe(len(context.encode("utf-8")))

                # 5. Generate (stage: generate)
                t0 = time.perf_counter()
                with tracer.start_as_current_span("rag.generate") as s:
                    messages = [
                        ("system", SYSTEM_PROMPT),
                        (
                            "human",
                            f"Contexto:\n{context}\n\nPregunta: {question}\n\nRespuesta:",
                        ),
                    ]
                    response = self._llm.invoke(messages)
                    answer = response.content
                    s.set_attribute("answer_len", len(answer))
                EVAL_STAGE_LATENCY.labels(stage="generate").observe(time.perf_counter() - t0)

                # 6. Abstention detection
                _abstention_signals = (
                    "no dispongo", "no tengo información", "no tengo datos",
                    "no hay información", "no puedo responder", "i don't have",
                    "i don't know", "no information available",
                )
                if any(s in answer.lower() for s in _abstention_signals):
                    EVAL_ABSTENTION.inc()
                    span.set_attribute("abstention", True)

                latency = time.perf_counter() - start
                span.set_attribute("latency_seconds", round(latency, 3))
                span.set_attribute("num_sources", len(documents))

                sources = [
                    {
                        "company": doc.metadata.get("company", "UNKNOWN"),
                        "ticker": doc.metadata.get("ticker", "UNKNOWN"),
                        "content_type": doc.metadata.get("content_type", "text"),
                        "page": doc.metadata.get("page"),
                        "score": round(float(doc.metadata.get("_score", 0.0)), 4),
                    }
                    for doc in documents
                ]

                logger.info(
                    "query_complete",
                    latency=round(latency, 3),
                    sources=len(sources),
                    company_filter=company_filter,
                )

                # Background RAGAS evaluation
                contexts = [doc.page_content for doc in documents]
                threading.Thread(
                    target=_run_ragas_evaluation,
                    args=(question, answer, contexts, self.settings),
                    daemon=True,
                ).start()

                return QueryResult(
                    answer=answer,
                    sources=sources,
                    latency_seconds=latency,
                    num_retrieved=len(sources),
                    query=question,
                )

            except Exception as exc:
                span.record_exception(exc)
                span.set_status(otel_trace.StatusCode.ERROR, str(exc))
                logger.exception("query_failed", error=str(exc))
                raise
