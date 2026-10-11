"""Chainlit frontend for the IBEX35 RAG API."""

import asyncio
import contextlib
import os
from collections.abc import Callable
from typing import NamedTuple

import chainlit as cl
import httpx
from chainlit.input_widget import Select

API_URL = os.getenv("API_URL", "http://localhost:8080")
API_TIMEOUT_SECONDS = 120.0
ALL_COMPANIES = "Todas"

# Mirrors COMPANY_TICKER_MAP in src/ingestion/metadata.py (the frontend ships in its own image).
COMPANIES = [
    ALL_COMPANIES,
    "ACCIONA", "ACERINOX", "ACS", "AENA", "AMADEUS", "ARCELORMITTAL",
    "BBVA", "CELLNEX", "ENAGAS", "ENDESA", "FERROVIAL", "FLUIDRA",
    "GRIFOLS", "IAG", "IBERDROLA", "INDITEX", "INDRA", "LOGISTA",
    "MAPFRE", "MERLINPROPERTIES", "NATURGY", "PUIG", "REDEIA", "REPSOL", "ROVI",
    "SACYR", "SANTANDER", "SOLARIA", "TELEFONICA", "UNICAJA",
]  # fmt: skip


class Step(NamedTuple):
    """A progress step shown while the API call runs."""

    in_progress: str
    done: str
    # Seconds to wait before moving on to the next step; None waits for the API.
    timeout: float | None
    # Builds a detail appended to `done` from the API response, if any.
    detail: Callable[[dict], str] | None = None


class Mode(NamedTuple):
    label: str
    endpoint: str
    steps: list[Step]


def _rag_detail(data: dict) -> str:
    return f"{len(data.get('sources', []))} fragmentos en {data.get('latency_seconds', 0):.1f}s"


def _agent_detail(data: dict) -> str:
    steps = data.get("steps_detail", [])
    if steps:
        return "\n" + "\n".join(f"  🔧 {s['tool']} — {s['duration_seconds']}s" for s in steps)
    tools = ", ".join(data.get("tools_used", [])) or "ninguna"
    return f"{data.get('steps_taken', 0)} pasos — {tools}"


def _route_detail(data: dict) -> str:
    return "Agente (PDFs + mercado)" if data.get("route") == "agent" else "RAG (solo PDFs)"


AUTO = Mode(
    label="Auto — enrutamiento inteligente",
    endpoint="/api/v1/ask",
    steps=[
        Step("1. Determinando ruta óptima", "1. Ruta seleccionada", 1.5, _route_detail),
        Step("2. Procesando consulta", "2. Consulta procesada", None, _rag_detail),
    ],
)
RAG = Mode(
    label="RAG — solo PDFs",
    endpoint="/api/v1/query",
    steps=[
        Step("1. Extrayendo filtros", "1. Filtros extraídos", 1.5),
        Step("2. Buscando en Qdrant", "2. Búsqueda completada", 3.0),
        Step("3. Rerankeando resultados", "3. Resultados rerankeados", 2.0),
        Step("4. Generando respuesta con LLM", "4. Respuesta generada", None, _rag_detail),
    ],
)
AGENT = Mode(
    label="Agente — PDFs + mercado en tiempo real",
    endpoint="/api/v1/agent",
    steps=[
        Step("1. Iniciando agente ReAct", "1. Agente iniciado", 2.0),
        Step("2. Consultando herramientas", "2. Herramientas usadas", 5.0),
        Step("3. Sintetizando respuesta", "3. Respuesta sintetizada", None, _agent_detail),
    ],
)
MODES = {mode.label: mode for mode in (AUTO, RAG, AGENT)}


# ── Progress ──────────────────────────────────────────────────────────────────


def _render_steps(steps: list[Step], current: int, data: dict | None) -> str:
    lines = []
    for i, step in enumerate(steps):
        if i < current:
            detail = f" — {step.detail(data)}" if step.detail and data is not None else ""
            lines.append(f"✅ {step.done}{detail}")
        elif i == current:
            lines.append(f"🔄 {step.in_progress}...")
        else:
            lines.append(f"⬜ {step.in_progress}")
    return "\n".join(lines)


async def _call_api(client: httpx.AsyncClient, url: str, payload: dict) -> dict:
    response = await client.post(url, json=payload)
    response.raise_for_status()
    return response.json()


async def _run_with_steps(
    client: httpx.AsyncClient, url: str, payload: dict, steps: list[Step]
) -> dict:
    """Call the API while advancing a live progress message one step at a time."""
    api_task = asyncio.create_task(_call_api(client, url, payload))
    data = None

    progress = cl.Message(content=_render_steps(steps, 0, None))
    await progress.send()

    for i, step in enumerate(steps):
        if data is not None or step.timeout is None:
            data = await api_task
        else:
            # Not done yet: advance the progress display anyway.
            with contextlib.suppress(TimeoutError):
                data = await asyncio.wait_for(asyncio.shield(api_task), timeout=step.timeout)
        progress.content = _render_steps(steps, i + 1, data)
        await progress.update()

    if data is None:
        data = await api_task
        progress.content = _render_steps(steps, len(steps), data)
        await progress.update()

    return data


# ── Chat handlers ─────────────────────────────────────────────────────────────


@cl.on_chat_start
async def start() -> None:
    await cl.ChatSettings(
        [
            Select(id="mode", label="Modo", values=list(MODES), initial_index=0),
            Select(id="company", label="Filtrar por empresa", values=COMPANIES, initial_index=0),
        ]
    ).send()

    await cl.Message(
        content=(
            "Hola, soy el asistente financiero del IBEX35.\n\n"
            "Puedo responder preguntas sobre los informes financieros de las **30 empresas** "
            "del índice. Usa el panel de ajustes (⚙️) para elegir el modo y filtrar por empresa.\n\n"
            "**Ejemplos:**\n"
            "- ¿Cuál fue el EBITDA de IBERDROLA en 2024?\n"
            "- Compara el beneficio neto de BBVA y SANTANDER\n"
            "- ¿Cómo cotiza INDITEX hoy respecto a su beneficio reportado?"
        )
    ).send()


@cl.on_settings_update
async def update_settings(settings: dict) -> None:
    cl.user_session.set("mode", settings["mode"])
    cl.user_session.set("company", settings["company"])


def _error_message(error: httpx.HTTPStatusError) -> str:
    if error.response.status_code == httpx.codes.TOO_MANY_REQUESTS:
        return "Demasiadas peticiones. Espera un momento y vuelve a intentarlo."
    return f"Error de la API ({error.response.status_code}): {error.response.text}"


def _answer_elements(data: dict, mode: Mode) -> list[cl.Text]:
    """Sources, route or agent steps, and latency shown under the answer."""
    elements: list[cl.Text] = []

    sources = data.get("sources", [])
    if sources:
        lines = "\n".join(
            f"- **{s['company']}** `{s.get('ticker', '')}` — relevancia: {s.get('score', 0):.2f}"
            for s in sources
        )
        elements.append(cl.Text(name="Fuentes", content=lines, display="inline"))

    if mode is AGENT or (mode is AUTO and data.get("route") == "agent"):
        tools = ", ".join(data.get("tools_used", []))
        route_note = " | **Ruta auto:** agente" if mode is AUTO else ""
        content = f"**Pasos:** {data.get('steps_taken', 0)} | **Herramientas:** {tools}{route_note}"
        elements.append(cl.Text(name="Proceso del agente", content=content, display="inline"))
    elif mode is AUTO:
        elements.append(cl.Text(name="Ruta", content="**Ruta auto:** RAG", display="inline"))

    cache_note = " ⚡ desde caché" if data.get("from_cache", False) else ""
    latency = data.get("latency_seconds", 0)
    elements.append(
        cl.Text(name="Info", content=f"Latencia: {latency:.2f}s{cache_note}", display="inline")
    )
    return elements


@cl.on_message
async def main(message: cl.Message) -> None:
    mode = MODES.get(cl.user_session.get("mode"), AUTO)
    company = cl.user_session.get("company") or ALL_COMPANIES
    payload = {
        "question": message.content,
        "company_filter": None if company == ALL_COMPANIES else company,
        # Per-chat-session agent memory (never shared across users)
        "thread_id": cl.user_session.get("id"),
    }

    async with httpx.AsyncClient(timeout=API_TIMEOUT_SECONDS) as client:
        try:
            data = await _run_with_steps(client, f"{API_URL}{mode.endpoint}", payload, mode.steps)
        except httpx.HTTPStatusError as error:
            await cl.Message(content=_error_message(error)).send()
            return
        except Exception as error:
            await cl.Message(content=f"No se pudo conectar con la API: {error}").send()
            return

    answer = cl.Message(content=data.get("answer", "Sin respuesta"))
    answer.elements = _answer_elements(data, mode)
    await answer.send()
