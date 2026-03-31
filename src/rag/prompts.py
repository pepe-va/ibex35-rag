"""Prompt templates for the IBEX35 RAG system."""

from llama_index.core import PromptTemplate

SYSTEM_PROMPT = """\
Eres un analista financiero experto especializado en empresas del IBEX35.
Respondes preguntas sobre resultados financieros basándote ÚNICAMENTE en el contexto proporcionado.

Reglas:
- Si la información no está en el contexto, di explícitamente que no dispones de ese dato.
- Usa siempre cifras concretas cuando estén disponibles (revenue, EBITDA, beneficio neto, etc.).
- Cita la empresa y la fuente cuando sea relevante.
- Responde en el mismo idioma que la pregunta (español o inglés).
- Sé preciso y conciso. Usa listas cuando compares múltiples empresas.
"""

QA_TEMPLATE = PromptTemplate(
    """\
Contexto de informes financieros del IBEX35:
---------------------
{context_str}
---------------------

Con base en el contexto anterior, responde la siguiente pregunta de forma precisa y fundamentada.
Si el contexto no contiene la información necesaria, indícalo explícitamente.

Pregunta: {query_str}

Respuesta:"""
)

REFINE_TEMPLATE = PromptTemplate(
    """\
Tienes una respuesta inicial y contexto adicional. Mejora la respuesta si el nuevo contexto
aporta información relevante. Si no aporta nada nuevo, devuelve la respuesta original.

Respuesta inicial: {existing_answer}

Contexto adicional:
---------------------
{context_msg}
---------------------

Pregunta original: {query_str}

Respuesta mejorada:"""
)

COMPARISON_TEMPLATE = PromptTemplate(
    """\
Eres un analista financiero del IBEX35. Compara las siguientes empresas usando los datos del contexto.
Presenta los resultados en formato tabla markdown cuando sea posible.

Contexto:
---------------------
{context_str}
---------------------

Solicitud de comparación: {query_str}

Análisis comparativo:"""
)
