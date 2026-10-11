"""LangSmith → Prometheus background collector.

Polls the LangSmith API every LANGSMITH_COLLECTOR_INTERVAL_SECONDS and records the
recent LLM runs as the `llm_*` metrics defined in src/observability/metrics.py.
Requires LANGSMITH_TRACING=true and LANGSMITH_API_KEY.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from src.core.config import Settings
from src.core.logging import get_logger
from src.observability.metrics import (
    LLM_COMPLETION_TOKENS,
    LLM_ERRORS,
    LLM_PROMPT_TOKENS,
    LLM_REQUEST_DURATION,
    LLM_RUNS,
    LLM_TOKENS_PER_SECOND,
    LLM_TOTAL_TOKENS,
)

if TYPE_CHECKING:
    from langsmith import Client
    from langsmith.schemas import Run

logger = get_logger(__name__)

# Overlap between polls so runs that finish around a boundary are not missed.
POLL_OVERLAP_SECONDS = 10
MAX_RUNS_PER_POLL = 100


def _extract_tokens(run: "Run") -> tuple[int, int]:
    """Return (prompt_tokens, completion_tokens) for a LangSmith run.

    Depending on the SDK version and the provider, LangSmith stores token usage
    in different places; they are checked in order of preference.
    """
    # 1. Direct attributes (SDK >= 0.1)
    prompt = getattr(run, "prompt_tokens", None)
    completion = getattr(run, "completion_tokens", None)
    if prompt is not None and completion is not None:
        return int(prompt or 0), int(completion or 0)

    # 2. run.extra → usage_metadata / token_usage (Ollama or OpenAI through LangChain)
    extra: dict[str, Any] = getattr(run, "extra", None) or {}
    for key in ("usage_metadata", "token_usage"):
        usage = extra.get(key) or {}
        if usage:
            p = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            c = usage.get("output_tokens") or usage.get("completion_tokens") or 0
            return int(p), int(c)

    # 3. run.outputs → llm_output → token_usage (legacy LangChain)
    outputs: dict[str, Any] = getattr(run, "outputs", None) or {}
    token_usage: dict[str, Any] = (outputs.get("llm_output") or {}).get("token_usage") or {}
    if token_usage:
        return (
            int(token_usage.get("prompt_tokens") or 0),
            int(token_usage.get("completion_tokens") or 0),
        )

    return 0, 0


def _record_run(run: "Run") -> None:
    LLM_RUNS.inc()
    if getattr(run, "error", None):
        LLM_ERRORS.inc()

    prompt_tokens, completion_tokens = _extract_tokens(run)
    for histogram, tokens in (
        (LLM_PROMPT_TOKENS, prompt_tokens),
        (LLM_COMPLETION_TOKENS, completion_tokens),
        (LLM_TOTAL_TOKENS, prompt_tokens + completion_tokens),
    ):
        if tokens > 0:
            histogram.observe(tokens)

    start_time = getattr(run, "start_time", None)
    end_time = getattr(run, "end_time", None)
    if not (start_time and end_time):
        return
    latency = (end_time - start_time).total_seconds()
    if latency <= 0:
        return
    LLM_REQUEST_DURATION.observe(latency)
    if completion_tokens > 0:
        LLM_TOKENS_PER_SECOND.observe(completion_tokens / latency)


def _langsmith_client(api_key: str, endpoint: str) -> "Client":
    # Imported lazily so that main.py configures LangSmith first.
    from langsmith import Client

    return Client(api_key=api_key, api_url=endpoint)


def _recent_llm_runs(client: "Client", project: str, since: datetime) -> list["Run"]:
    return list(
        client.list_runs(
            project_name=project, run_type="llm", start_time=since, limit=MAX_RUNS_PER_POLL
        )
    )


async def collect_langsmith_metrics(settings: Settings) -> None:
    """Background task: copy LangSmith LLM run statistics into Prometheus."""
    api_key = settings.langsmith_api_key
    if not (settings.langsmith_tracing and api_key):
        logger.info("langsmith_collection_disabled", reason="tracing disabled or api key missing")
        return

    client = _langsmith_client(api_key.get_secret_value(), settings.langsmith_endpoint)
    interval = settings.langsmith_collector_interval_seconds
    while True:
        await asyncio.sleep(interval)
        since = datetime.now(tz=UTC) - timedelta(seconds=interval + POLL_OVERLAP_SECONDS)
        try:
            runs = await asyncio.to_thread(
                _recent_llm_runs, client, settings.langsmith_project, since
            )
            for run in runs:
                _record_run(run)
        except Exception as exc:
            logger.warning("langsmith_collection_failed", error=str(exc))
            continue
        if runs:
            logger.info(
                "langsmith_metrics_collected", runs=len(runs), project=settings.langsmith_project
            )
