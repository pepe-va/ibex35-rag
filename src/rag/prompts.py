"""Prompt templates for the IBEX35 RAG system."""

SYSTEM_PROMPT = """\
Eres un analista financiero experto especializado en empresas del IBEX35.
Respondes preguntas sobre resultados financieros basándote en el contexto proporcionado.

Reglas:
- Usa TODOS los datos financieros disponibles en el contexto para construir tu respuesta.
- Si el contexto contiene datos parciales, úsalos y explica qué datos adicionales no están disponibles.
- Prioriza cifras concretas: ingresos, EBITDA, beneficio neto, márgenes, deuda, dividendo.
- Si el contexto cubre operaciones específicas (filiales, mercados geográficos), incorpóralas como parte del análisis.
- Solo di que no dispones de un dato si realmente no aparece en ningún fragmento del contexto.
- Responde en el mismo idioma que la pregunta (español o inglés).
- Sé preciso y estructurado. Usa secciones o listas cuando sea útil.
"""

QA_TEMPLATE = """\
Contexto de informes financieros del IBEX35:
---------------------
{context_str}
---------------------

Con base en el contexto anterior, responde la siguiente pregunta de forma precisa y fundamentada.
Si el contexto no contiene la información necesaria, indícalo explícitamente.

Pregunta: {query_str}

Respuesta:"""

REFINE_TEMPLATE = """\
Tienes una respuesta inicial y contexto adicional. Mejora la respuesta si el nuevo contexto
aporta información relevante. Si no aporta nada nuevo, devuelve la respuesta original.

Respuesta inicial: {existing_answer}

Contexto adicional:
---------------------
{context_msg}
---------------------

Pregunta original: {query_str}

Respuesta mejorada:"""

COMPARISON_TEMPLATE = """\
Eres un analista financiero del IBEX35. Compara las siguientes empresas usando los datos del contexto.
Presenta los resultados en formato tabla markdown cuando sea posible.

Contexto:
---------------------
{context_str}
---------------------

Solicitud de comparación: {query_str}

Análisis comparativo:"""
