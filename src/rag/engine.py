"""RAG query engine: retrieval + generation with MLflow tracking."""

import time
from dataclasses import dataclass, field

import mlflow
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.ollama import Ollama

from src.config import Settings
from src.logging_config import get_logger
from src.rag.prompts import QA_TEMPLATE, REFINE_TEMPLATE

logger = get_logger(__name__)


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


class RAGEngine:
    """LlamaIndex-based RAG engine with MLflow instrumentation."""

    def __init__(self, settings: Settings, index: VectorStoreIndex) -> None:
        self.settings = settings
        self.index = index
        self._llm = Ollama(
            model=settings.ollama_llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
            request_timeout=settings.llm_request_timeout,
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
        """Execute a RAG query and log metrics to MLflow."""
        start = time.perf_counter()

        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.settings.mlflow_experiment_name)

        with mlflow.start_run(run_name="query", nested=True):
            mlflow.set_tag("pipeline", "query")
            mlflow.log_param("question_length", len(question))
            mlflow.log_param("llm_model", self.settings.ollama_llm_model)
            mlflow.log_param("similarity_top_k", self.settings.similarity_top_k)
            if company_filter:
                mlflow.log_param("company_filter", company_filter)

            try:
                engine = (
                    self._build_query_engine_for_company(company_filter)
                    if company_filter
                    else self._query_engine
                )
                response = engine.query(question)

                latency = time.perf_counter() - start
                sources = self._extract_sources(response)

                mlflow.log_metric("latency_seconds", round(latency, 3))
                mlflow.log_metric("num_sources", len(sources))
                mlflow.log_metric("answer_length", len(str(response)))

                logger.info(
                    "query_complete",
                    latency=round(latency, 3),
                    sources=len(sources),
                    company_filter=company_filter,
                )

                return QueryResult(
                    answer=str(response),
                    sources=sources,
                    latency_seconds=latency,
                    num_retrieved=len(sources),
                    query=question,
                )

            except Exception as exc:
                mlflow.log_param("error", str(exc))
                logger.exception("query_failed", error=str(exc))
                raise
