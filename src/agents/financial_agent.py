"""Financial agent: combines RAG knowledge with real-time market tools."""

import time
from dataclasses import dataclass, field

import mlflow
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.ollama import Ollama

from src.agents.tools import ALL_TOOLS
from src.config import Settings
from src.logging_config import get_logger
from src.rag.engine import RAGEngine

logger = get_logger(__name__)

AGENT_SYSTEM_PROMPT = """\
Eres un analista financiero senior especializado en el mercado español (IBEX35).
Tienes acceso a:
1. Una base de conocimiento con resultados financieros oficiales de las empresas del IBEX35.
2. Herramientas para obtener precios de mercado en tiempo real.

Estrategia:
- Para preguntas sobre fundamentales (revenue, EBITDA, deuda, estrategia): usa la base de conocimiento RAG.
- Para preguntas sobre precios o rentabilidad bursátil: usa las herramientas de mercado.
- Para análisis completos: combina ambas fuentes.
- Siempre cita las fuentes de los datos que usas.
- Cuando compares empresas, usa tablas markdown.
- Responde en el idioma de la pregunta (español o inglés).
"""


@dataclass
class AgentResult:
    answer: str
    latency_seconds: float = 0.0
    steps_taken: int = 0
    tools_used: list[str] = field(default_factory=list)
    query: str = ""

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "latency_seconds": round(self.latency_seconds, 3),
            "steps_taken": self.steps_taken,
            "tools_used": self.tools_used,
            "query": self.query,
        }


class FinancialAgent:
    """ReAct agent with RAG knowledge base + real-time market tools."""

    def __init__(self, settings: Settings, rag_engine: RAGEngine) -> None:
        self.settings = settings
        self.rag_engine = rag_engine
        self._agent = self._build_agent()

    def _build_agent(self) -> ReActAgent:
        llm = Ollama(
            model=self.settings.ollama_llm_model,
            base_url=self.settings.ollama_base_url,
            temperature=self.settings.llm_temperature,
            request_timeout=self.settings.llm_request_timeout,
        )

        # Wrap RAG engine as a LlamaIndex tool
        rag_tool = QueryEngineTool(
            query_engine=self.rag_engine._query_engine,
            metadata=ToolMetadata(
                name="ibex35_financial_reports",
                description=(
                    "Consulta resultados financieros oficiales de empresas del IBEX35: "
                    "ingresos, EBITDA, beneficio neto, deuda, dividendos, guidance y estrategia. "
                    "Usa esta herramienta para preguntas sobre fundamentales de negocio."
                ),
            ),
        )

        all_tools = [rag_tool, *ALL_TOOLS]

        return ReActAgent.from_tools(
            tools=all_tools,
            llm=llm,
            max_iterations=8,
            verbose=False,
            system_prompt=AGENT_SYSTEM_PROMPT,
        )

    def run(self, question: str) -> AgentResult:
        """Execute the agent and log to MLflow."""
        start = time.perf_counter()

        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.settings.mlflow_experiment_name)

        with mlflow.start_run(run_name="agent", nested=True):
            mlflow.set_tag("pipeline", "agent")
            mlflow.log_param("question_length", len(question))
            mlflow.log_param("model", self.settings.ollama_llm_model)

            try:
                response = self._agent.chat(question)
                latency = time.perf_counter() - start

                # Extract tools used from agent steps
                tools_used: list[str] = []
                steps = 0
                if hasattr(response, "sources"):
                    for src in response.sources:
                        if hasattr(src, "tool_name"):
                            tools_used.append(src.tool_name)
                            steps += 1

                mlflow.log_metric("latency_seconds", round(latency, 3))
                mlflow.log_metric("steps_taken", steps)
                mlflow.log_param("tools_used", ",".join(tools_used) if tools_used else "none")

                logger.info(
                    "agent_complete",
                    latency=round(latency, 3),
                    steps=steps,
                    tools=tools_used,
                )

                return AgentResult(
                    answer=str(response),
                    latency_seconds=latency,
                    steps_taken=steps,
                    tools_used=tools_used,
                    query=question,
                )

            except Exception as exc:
                mlflow.log_param("error", str(exc))
                logger.exception("agent_failed", error=str(exc))
                raise
