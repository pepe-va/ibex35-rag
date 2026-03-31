"""Chainlit frontend for the IBEX35 RAG API."""

import os

import httpx
import chainlit as cl
from chainlit.input_widget import Select

API_URL = os.getenv("API_URL", "http://localhost:8080")

COMPANIES = [
    "Todas",
    "ACCIONA", "ACERINOX", "ACS", "AENA", "AMADEUS", "ARCELORMITTAL",
    "BBVA", "CELLNEX", "ENAGAS", "ENDESA", "FERROVIAL", "FLUIDRA",
    "GRIFOLS", "IAG", "IBERDROLA", "INDITEX", "INDRA", "LOGISTA",
    "MAPFRE", "MERLINPROPERTIES", "NATURGY", "PUIG", "REDEIA", "ROVI",
    "SACYR", "SANTANDER", "SOLARIA", "TELEFONICA", "UNICAJA",
]


@cl.on_chat_start
async def start() -> None:
    await cl.ChatSettings(
        [
            Select(
                id="mode",
                label="Modo",
                values=["RAG — solo PDFs", "Agente — PDFs + mercado en tiempo real"],
                initial_index=0,
            ),
            Select(
                id="company",
                label="Filtrar por empresa",
                values=COMPANIES,
                initial_index=0,
            ),
        ]
    ).send()

    await cl.Message(
        content=(
            "Hola, soy el asistente financiero del IBEX35.\n\n"
            "Puedo responder preguntas sobre los informes financieros de las **29 empresas** "
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


@cl.on_message
async def main(message: cl.Message) -> None:
    mode = cl.user_session.get("mode") or "RAG — solo PDFs"
    company = cl.user_session.get("company") or "Todas"

    use_agent = mode.startswith("Agente")
    company_filter = None if company == "Todas" else company
    endpoint = "/api/v1/agent" if use_agent else "/api/v1/query"

    msg = cl.Message(content="")
    await msg.send()

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"{API_URL}{endpoint}",
                json={"question": message.content, "company_filter": company_filter},
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                msg.content = "Demasiadas peticiones. Espera un momento y vuelve a intentarlo."
            else:
                msg.content = f"Error de la API ({e.response.status_code}): {e.response.text}"
            await msg.update()
            return
        except Exception as e:
            msg.content = f"No se pudo conectar con la API: {e}"
            await msg.update()
            return

    msg.content = data.get("answer", "Sin respuesta")

    elements: list[cl.Text] = []

    sources = data.get("sources", [])
    if sources:
        lines = "\n".join(
            f"- **{s['company']}** `{s.get('ticker', '')}` — relevancia: {s.get('score', 0):.2f}"
            for s in sources
        )
        elements.append(cl.Text(name="Fuentes", content=lines, display="inline"))

    if use_agent:
        steps = data.get("steps_taken", 0)
        tools = ", ".join(data.get("tools_used", []))
        elements.append(
            cl.Text(
                name="Proceso del agente",
                content=f"**Pasos:** {steps} | **Herramientas:** {tools}",
                display="inline",
            )
        )

    latency = data.get("latency_seconds", 0)
    from_cache = data.get("from_cache", False)
    cache_note = " ⚡ desde caché" if from_cache else ""
    elements.append(
        cl.Text(name="Info", content=f"Latencia: {latency:.2f}s{cache_note}", display="inline")
    )

    msg.elements = elements
    await msg.update()
