"""Financial agent: combines RAG knowledge with real-time market tools."""

import time
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

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
    thread_id: str = "default"

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "latency_seconds": round(self.latency_seconds, 3),
            "steps_taken": self.steps_taken,
            "tools_used": self.tools_used,
            "query": self.query,
        }


class FinancialAgent:
    """Tool-calling agent with RAG knowledge base + real-time market tools.

    Uses the new LangChain create_agent API (LangGraph-based) with:
    - MemorySaver checkpointer for per-session conversation memory
    - ModelCallLimitMiddleware to cap LLM calls per request
    """

    def __init__(self, settings: Settings, rag_engine: RAGEngine) -> None:
        self.settings = settings
        self.rag_engine = rag_engine
        self._agent = self._build_agent()

    def _build_agent(self):
        llm = ChatOllama(
            model=self.settings.ollama_llm_model,
            base_url=self.settings.ollama_base_url,
            temperature=self.settings.llm_temperature,
        )

        # Wrap RAG engine as a LangChain tool
        def _rag_query(question: str) -> str:
            result = self.rag_engine.query(question)
            return result.answer

        rag_tool = StructuredTool.from_function(
            func=_rag_query,
            name="ibex35_financial_reports",
            description=(
                "Consulta resultados financieros oficiales de empresas del IBEX35: "
                "ingresos, EBITDA, beneficio neto, deuda, dividendos, guidance y estrategia. "
                "Usa esta herramienta para preguntas sobre fundamentales de negocio."
            ),
        )

        all_tools = [rag_tool, *ALL_TOOLS]

        return create_agent(
            model=llm,
            tools=all_tools,
            system_prompt=AGENT_SYSTEM_PROMPT,
            checkpointer=MemorySaver(),
            middleware=[ModelCallLimitMiddleware(run_limit=8, exit_behavior="end")],
        )

    def run(self, question: str, thread_id: str | None = None) -> AgentResult:
        """Execute the agent with optional session memory via thread_id."""
        start = time.perf_counter()
        tid = thread_id or "default"
        config = {"configurable": {"thread_id": tid}}

        try:
            response = self._agent.invoke(
                {"messages": [HumanMessage(content=question)]},
                config=config,
            )
            latency = time.perf_counter() - start

            # Extract last AI message as answer
            messages = response.get("messages", [])
            answer = ""
            tools_used: list[str] = []
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tools_used.append(tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", ""))
                if hasattr(msg, "content") and not getattr(msg, "tool_calls", None):
                    answer = msg.content if isinstance(msg.content, str) else str(msg.content)

            logger.info(
                "agent_complete",
                latency=round(latency, 3),
                tools=tools_used,
                thread_id=tid,
            )

            return AgentResult(
                answer=answer,
                latency_seconds=latency,
                steps_taken=len([m for m in messages if hasattr(m, "tool_calls") and m.tool_calls]),
                tools_used=list(dict.fromkeys(tools_used)),  # deduplicate preserving order
                query=question,
                thread_id=tid,
            )

        except Exception as exc:
            logger.exception("agent_failed", error=str(exc))
            raise
