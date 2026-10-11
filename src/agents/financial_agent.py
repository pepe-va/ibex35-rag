"""Financial agent: combines the RAG knowledge base with live market tools."""

import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StreamMode

from src.agents.agent_tools import ALL_TOOLS
from src.core.config import Settings
from src.core.logging import get_logger
from src.rag.engine import RAGEngine

logger = get_logger(__name__)

# LLM calls allowed per request before the agent stops.
MAX_MODEL_CALLS = 8
# LangGraph node that runs the LLM in `create_agent`.
MODEL_NODE = "model"

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
"""  # noqa: E501

REPORTS_TOOL_DESCRIPTION = (
    "Consulta resultados financieros oficiales de empresas del IBEX35: "
    "ingresos, EBITDA, beneficio neto, deuda, dividendos, guidance y estrategia. "
    "Usa esta herramienta para preguntas sobre fundamentales de negocio."
)


@dataclass
class StepDetail:
    tool: str
    duration_seconds: float


@dataclass
class AgentResult:
    answer: str
    latency_seconds: float = 0.0
    steps_taken: int = 0
    tools_used: list[str] = field(default_factory=list)
    steps_detail: list[StepDetail] = field(default_factory=list)
    query: str = ""
    thread_id: str = "default"


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _tool_call_name(tool_call: Any) -> str:
    if isinstance(tool_call, dict):
        return str(tool_call.get("name", ""))
    return str(getattr(tool_call, "name", ""))


def _updated_messages(chunk: Any) -> Iterator[Any]:
    """Messages carried by one chunk of a LangGraph `updates` stream."""
    if not isinstance(chunk, dict):
        return
    for update in chunk.values():
        messages = update.get("messages", []) if isinstance(update, dict) else []
        if isinstance(messages, list):
            yield from messages


@dataclass
class _RunTracker:
    """Tool calls, their durations and the last answer seen during an agent run."""

    started_at: float
    answer: str = ""
    tools_used: list[str] = field(default_factory=list)
    steps_detail: list[StepDetail] = field(default_factory=list)
    pending_tool: str | None = None
    tool_started_at: float | None = None

    def observe(self, message: Any) -> None:
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            for tool_call in tool_calls:
                self.pending_tool = _tool_call_name(tool_call)
                self.tools_used.append(self.pending_tool)
                self.tool_started_at = time.perf_counter()
            return
        if getattr(message, "name", None) and self.pending_tool:
            self._finish_tool(self.pending_tool)
        text = _message_text(message)
        if text:
            self.answer = text

    def _finish_tool(self, tool: str) -> None:
        duration = round(time.perf_counter() - (self.tool_started_at or self.started_at), 2)
        self.steps_detail.append(StepDetail(tool=tool, duration_seconds=duration))
        self.pending_tool = None
        self.tool_started_at = None


class FinancialAgent:
    """Tool-calling agent over the RAG knowledge base and live market tools.

    Built with LangChain's `create_agent` (LangGraph), with a MemorySaver
    checkpointer for per-thread memory and a cap on LLM calls per request.
    """

    def __init__(self, settings: Settings, rag_engine: RAGEngine) -> None:
        self.settings = settings
        self.rag_engine = rag_engine
        self._agent = self._build_agent()

    def _query_reports(self, question: str) -> str:
        return self.rag_engine.query(question).answer

    def _build_agent(self) -> CompiledStateGraph[Any, Any, Any, Any]:
        llm = ChatOllama(
            model=self.settings.ollama_llm_model,
            base_url=self.settings.ollama_base_url,
            temperature=self.settings.llm_temperature,
        )
        reports_tool = StructuredTool.from_function(
            func=self._query_reports,
            name="ibex35_financial_reports",
            description=REPORTS_TOOL_DESCRIPTION,
        )
        return create_agent(
            model=llm,
            tools=[reports_tool, *ALL_TOOLS],
            system_prompt=AGENT_SYSTEM_PROMPT,
            checkpointer=MemorySaver(),
            middleware=[ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end")],
        )

    def _stream(self, question: str, thread_id: str, stream_mode: StreamMode) -> Iterator[Any]:
        return self._agent.stream(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode=stream_mode,
        )

    def run(self, question: str, thread_id: str | None = None) -> AgentResult:
        """Run the agent. Without a thread_id each call gets a fresh, unshared memory."""
        thread = thread_id or uuid.uuid4().hex
        tracker = _RunTracker(started_at=time.perf_counter())
        try:
            for chunk in self._stream(question, thread, stream_mode="updates"):
                for message in _updated_messages(chunk):
                    tracker.observe(message)
        except Exception as exc:
            logger.exception("agent_failed", error=str(exc))
            raise

        latency = time.perf_counter() - tracker.started_at
        logger.info(
            "agent_complete", latency=round(latency, 3), tools=tracker.tools_used, thread_id=thread
        )
        return AgentResult(
            answer=tracker.answer,
            latency_seconds=latency,
            steps_taken=len(tracker.steps_detail),
            tools_used=list(dict.fromkeys(tracker.tools_used)),
            steps_detail=tracker.steps_detail,
            query=question,
            thread_id=thread,
        )

    def stream_run(self, question: str, thread_id: str | None = None) -> Iterator[dict[str, Any]]:
        """Yield tool_start/tool_end per tool call, a token event per LLM chunk, then done."""
        try:
            yield from self._stream_events(question, thread_id or uuid.uuid4().hex)
        except Exception as exc:
            logger.exception("agent_stream_failed", error=str(exc))
            yield {"type": "error", "error": str(exc)}

    def _stream_events(self, question: str, thread_id: str) -> Iterator[dict[str, Any]]:
        started_at = time.perf_counter()
        # Tool-call chunks arrive before their ToolMessage: remember id → name.
        tool_names: dict[str, str] = {}
        tool_started_at: dict[str, float] = {}
        steps_detail: list[dict[str, Any]] = []

        for message, metadata in self._stream(question, thread_id, stream_mode="messages"):
            if isinstance(message, AIMessageChunk):
                for call in message.tool_call_chunks or []:
                    name, call_id = call.get("name", ""), call.get("id", "")
                    if name and call_id and call_id not in tool_names:
                        tool_names[call_id] = name
                        tool_started_at[name] = time.perf_counter()
                        yield {"type": "tool_start", "tool": name}
                is_answer_text = (
                    metadata.get("langgraph_node") == MODEL_NODE and not message.tool_call_chunks
                )
                if message.content and is_answer_text:
                    yield {"type": "token", "content": message.content}
            elif isinstance(message, ToolMessage):
                name = tool_names.get(message.tool_call_id) or message.name or "unknown"
                duration = round(time.perf_counter() - tool_started_at.get(name, started_at), 2)
                steps_detail.append({"tool": name, "duration_seconds": duration})
                yield {"type": "tool_end", "tool": name, "duration_seconds": duration}

        tools_used = list(tool_names.values())
        latency = round(time.perf_counter() - started_at, 2)
        logger.info("agent_stream_complete", latency=latency, tools=tools_used)
        yield {
            "type": "done",
            "latency_seconds": latency,
            "steps_taken": len(steps_detail),
            "tools_used": list(dict.fromkeys(tools_used)),
            "steps_detail": steps_detail,
        }
