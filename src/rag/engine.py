"""RAG query engine: retrieval + generation with RAGAS evaluation."""

import threading
import time
from dataclasses import dataclass, field

from llama_index.core import VectorStoreIndex
from llama_index.core.callbacks import CallbackManager, CBEventType
from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.ollama import Ollama

from src.config import Settings
from src.logging_config import get_logger
from src.monitoring.metrics import (
    CONTEXT_BYTES,
    EMPTY_RETRIEVAL,
    LLM_COMPLETION_TOKENS,
    LLM_LOAD_DURATION,
    LLM_PROMPT_TOKENS,
    LLM_TOTAL_DURATION,
    RETRIEVED_CHUNKS,
    RETRIEVAL_SCORE,
)
from src.rag.prompts import QA_TEMPLATE, REFINE_TEMPLATE

logger = get_logger(__name__)


class _OllamaMetricsHandler(BaseCallbackHandler):
    """Captures Ollama-specific fields from LLM responses and feeds Prometheus."""

    def __init__(self) -> None:
        super().__init__([], [])

    def on_event_start(self, event_type, payload=None, event_id="", **kwargs) -> str:
        return event_id

    def on_event_end(self, event_type, payload=None, event_id="", **kwargs) -> None:
        if event_type != CBEventType.LLM or not payload:
            return
        response = payload.get("response")
        raw = getattr(response, "raw", None) or {}
        if raw.get("prompt_eval_count"):
            LLM_PROMPT_TOKENS.observe(raw["prompt_eval_count"])
        if raw.get("eval_count"):
            LLM_COMPLETION_TOKENS.observe(raw["eval_count"])
        if raw.get("total_duration"):
            LLM_TOTAL_DURATION.observe(raw["total_duration"] / 1e9)
        if raw.get("load_duration"):
            LLM_LOAD_DURATION.observe(raw["load_duration"] / 1e9)

    def start_trace(self, trace_id=None) -> None: ...
    def end_trace(self, trace_id=None, trace_map=None) -> None: ...


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


def _run_ragas_evaluation(
    question: str,
    answer: str,
    contexts: list[str],
    settings: Settings,
) -> None:
    """Evaluate RAG response with RAGAS metrics (runs in background thread)."""
    try:
        from llama_index.embeddings.ollama import OllamaEmbedding
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.embeddings import LlamaIndexEmbeddingsWrapper
        from ragas.llms import LlamaIndexLLMWrapper
        from ragas.metrics import AnswerRelevancy, Faithfulness

        llm_wrapper = LlamaIndexLLMWrapper(
            Ollama(
                model=settings.ollama_llm_model,
                base_url=settings.ollama_base_url,
                temperature=0.0,
            )
        )
        embed_wrapper = LlamaIndexEmbeddingsWrapper(
            OllamaEmbedding(
                model_name=settings.ollama_embed_model,
                base_url=settings.ollama_base_url,
            )
        )

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
        dataset = EvaluationDataset(samples=[sample])
        result = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), AnswerRelevancy()],
            llm=llm_wrapper,
            embeddings=embed_wrapper,
        )
        row = result.to_pandas().iloc[0].to_dict()
        logger.info(
            "ragas_evaluation",
            faithfulness=round(float(row.get("faithfulness") or 0), 3),
            answer_relevancy=round(float(row.get("answer_relevancy") or 0), 3),
        )
    except Exception as exc:
        logger.warning("ragas_evaluation_failed", error=str(exc))


class RAGEngine:
    """LlamaIndex-based RAG engine with RAGAS evaluation."""

    def __init__(self, settings: Settings, index: VectorStoreIndex) -> None:
        self.settings = settings
        self.index = index
        self._cb = CallbackManager([_OllamaMetricsHandler()])
        self._llm = Ollama(
            model=settings.ollama_llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            request_timeout=settings.llm_request_timeout,
            callback_manager=self._cb,
        )
        self._query_engine = self._build_query_engine()

    def _build_query_engine(self) -> RetrieverQueryEngine:
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.settings.similarity_top_k,
        )
        synthesizer = get_response_synthesizer(
            llm=self._llm,
            text_qa_template=QA_TEMPLATE,
            refine_template=REFINE_TEMPLATE,
            response_mode="compact",
            streaming=False,
        )
        # get_response_synthesizer propagates Settings.callback_manager and overwrites
        # the LLM's callback_manager — restore ours so _OllamaMetricsHandler fires.
        self._llm.callback_manager = self._cb
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=synthesizer,
        )

    def _build_query_engine_for_company(self, company: str) -> RetrieverQueryEngine:
        """Build a query engine filtered to a specific company."""
        from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.settings.similarity_top_k,
            filters=MetadataFilters(
                filters=[ExactMatchFilter(key="company", value=company.upper())]
            ),
        )
        synthesizer = get_response_synthesizer(
            llm=self._llm,
            text_qa_template=QA_TEMPLATE,
            refine_template=REFINE_TEMPLATE,
            response_mode="compact",
        )
        self._llm.callback_manager = self._cb
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=synthesizer,
        )

    def _extract_sources(self, response) -> list[dict]:
        sources = []
        for node in response.source_nodes:
            sources.append(
                {
                    "company": node.metadata.get("company", "UNKNOWN"),
                    "ticker": node.metadata.get("ticker", "UNKNOWN"),
                    "score": round(node.score or 0.0, 4),
                }
            )
        return sources

    def query(self, question: str, company_filter: str | None = None) -> QueryResult:
        """Execute a RAG query and trigger background RAGAS evaluation."""
        start = time.perf_counter()

        try:
            engine = (
                self._build_query_engine_for_company(company_filter)
                if company_filter
                else self._query_engine
            )
            response = engine.query(question)

            latency = time.perf_counter() - start
            sources = self._extract_sources(response)

            # Retrieval metrics
            RETRIEVED_CHUNKS.observe(len(response.source_nodes))
            if not response.source_nodes:
                EMPTY_RETRIEVAL.inc()
            else:
                for node in response.source_nodes:
                    if node.score is not None:
                        RETRIEVAL_SCORE.observe(node.score)

            ctx_bytes = sum(len(n.get_content().encode("utf-8")) for n in response.source_nodes)
            CONTEXT_BYTES.observe(ctx_bytes)

            logger.info(
                "query_complete",
                latency=round(latency, 3),
                sources=len(sources),
                company_filter=company_filter,
            )

            # Fire-and-forget RAGAS evaluation in background
            contexts = [node.get_content() for node in response.source_nodes]
            threading.Thread(
                target=_run_ragas_evaluation,
                args=(question, str(response), contexts, self.settings),
                daemon=True,
            ).start()

            return QueryResult(
                answer=str(response),
                sources=sources,
                latency_seconds=latency,
                num_retrieved=len(sources),
                query=question,
            )

        except Exception as exc:
            logger.exception("query_failed", error=str(exc))
            raise
