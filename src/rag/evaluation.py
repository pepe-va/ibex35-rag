"""Online RAG evaluation: cheap proxy metrics and background RAGAS scoring.

Nothing here has a ground truth; the values are trend indicators for the
dashboards, not a substitute for an offline evaluation with labelled data.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from math import exp
from typing import cast

from langchain_core.documents import Document

from src.core.config import Settings
from src.core.logging import get_logger
from src.observability.metrics import (
    CONTEXT_BYTES,
    EVAL_ANSWER_CORRECTNESS,
    EVAL_ANSWER_RELEVANCY,
    EVAL_ANSWER_SIMILARITY,
    EVAL_CHUNK_ATTRIBUTION,
    EVAL_CHUNK_UTILIZATION,
    EVAL_CONTEXT_CLIFF,
    EVAL_CONTEXTUAL_PRECISION,
    EVAL_CONTEXTUAL_RECALL,
    EVAL_COST_PER_INFERENCE,
    EVAL_FAITHFULNESS,
    RETRIEVAL_SCORE,
    RETRIEVED_CHUNKS,
)

logger = get_logger(__name__)

_RAGAS_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ragas-eval")

# A chunk counts as used by the answer if it shares at least this share of the answer's tokens.
ATTRIBUTION_MIN_OVERLAP = 0.03
# Context size (KB) at which the context-cliff proxy saturates.
CONTEXT_CLIFF_KB = 32.0
# Token budget for the unigram BLEU proxy.
BLEU_MAX_TOKENS = 256

_ABSTENTION_SIGNALS = (
    "no dispongo", "no tengo información", "no tengo datos",
    "no hay información", "no puedo responder", "i don't have",
    "i don't know", "no information available", "no puedo",
    "datos insuficientes", "información insuficiente",
    "no está disponible", "no se encuentra disponible",
)  # fmt: skip


# ── Tokens and arithmetic ─────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _normalize_scores_01(scores: list[float]) -> list[float]:
    """Normalize arbitrary reranker scores into [0, 1]."""
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi - lo <= 1e-9:
        return [1.0 for _ in scores]
    return [_clamp01((s - lo) / (hi - lo)) for s in scores]


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


# ── Relevance helpers ─────────────────────────────────────────────────────────


def _relevance_from_scores(scores_norm: list[float]) -> list[int]:
    """Build binary relevance labels from normalized reranker scores."""
    if not scores_norm:
        return []
    threshold = max(0.35, _median(scores_norm))
    rel = [1 if s >= threshold else 0 for s in scores_norm]
    if not any(rel):
        rel[0] = 1
    return rel


def _contextual_precision_from_binary(rel_binary: list[int]) -> float:
    hits = 0
    weighted_sum = 0.0
    total_relevant = sum(rel_binary)
    if total_relevant == 0:
        return 0.0
    for i, rel in enumerate(rel_binary, start=1):
        if rel:
            hits += 1
            weighted_sum += hits / i
    return _clamp01(_safe_div(weighted_sum, total_relevant))


# ── Lexical proxy metrics ─────────────────────────────────────────────────────


def _lcs_len(a: list[str], b: list[str], max_tokens: int = 128) -> int:
    a = a[:max_tokens]
    b = b[:max_tokens]
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[-1]


def _rouge_l(candidate_tokens: list[str], reference_tokens: list[str]) -> float:
    if not candidate_tokens or not reference_tokens:
        return 0.0
    lcs = _lcs_len(candidate_tokens, reference_tokens)
    precision = _safe_div(lcs, len(candidate_tokens))
    recall = _safe_div(lcs, len(reference_tokens))
    if precision + recall == 0:
        return 0.0
    return _clamp01(2 * precision * recall / (precision + recall))


def _bleu_unigram(candidate_tokens: list[str], reference_tokens: list[str]) -> float:
    if not candidate_tokens:
        return 0.0
    ref_counts: dict[str, int] = {}
    for t in reference_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    matches = 0
    cand_counts: dict[str, int] = {}
    for t in candidate_tokens:
        cand_counts[t] = cand_counts.get(t, 0) + 1
    for token, count in cand_counts.items():
        matches += min(count, ref_counts.get(token, 0))
    precision = _safe_div(matches, len(candidate_tokens))
    brevity_penalty = 1.0
    if len(candidate_tokens) < len(reference_tokens):
        brevity_penalty = exp(1 - _safe_div(len(reference_tokens), len(candidate_tokens)))
    return _clamp01(precision * brevity_penalty)


def _token_set_similarity(a_tokens: list[str], b_tokens: list[str]) -> float:
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    return _clamp01(_safe_div(inter, union))


# ── Public API ────────────────────────────────────────────────────────────────


def is_abstention(answer: str) -> bool:
    """True if the generated answer declines to answer for lack of data."""
    text = answer.lower()
    return any(signal in text for signal in _ABSTENTION_SIGNALS)


def _chunk_relevance(
    documents: list[Document], requested: list[str], scores: list[float]
) -> list[int]:
    """Binary relevance per chunk: belongs to a requested company, else score-based."""
    if requested:
        targets = set(requested)
        relevance = [
            int(str(doc.metadata.get("company", "")).upper() in targets) for doc in documents
        ]
        if any(relevance):
            return relevance
    return _relevance_from_scores(scores)


def _contextual_recall(question: str, documents: list[Document], context: str) -> float:
    question_tokens = set(_tokenize(question))
    companies = " ".join(str(doc.metadata.get("company", "")) for doc in documents)
    context_tokens = set(_tokenize(f"{context}\n{companies}"))
    if not (question_tokens and context_tokens):
        return 0.0
    return _clamp01(len(question_tokens & context_tokens) / len(question_tokens))


def record_retrieval_metrics(
    question: str,
    documents: list[Document],
    requested_companies: list[str],
    context: str,
) -> None:
    """Record retrieval-quality metrics for the reranked chunks."""
    RETRIEVED_CHUNKS.observe(len(documents))
    scores = _normalize_scores_01(
        [float(doc.metadata.get("_score", 0.0) or 0.0) for doc in documents]
    )
    for score in scores:
        RETRIEVAL_SCORE.observe(score)

    if documents:
        relevance = _chunk_relevance(documents, requested_companies, scores)
        EVAL_CHUNK_UTILIZATION.observe(sum(relevance) / len(documents))
        EVAL_CONTEXTUAL_PRECISION.observe(_contextual_precision_from_binary(relevance))
    else:
        EVAL_CHUNK_UTILIZATION.observe(0.0)
        EVAL_CONTEXTUAL_PRECISION.observe(0.0)

    EVAL_CONTEXTUAL_RECALL.observe(_contextual_recall(question, documents, context))
    CONTEXT_BYTES.observe(len(context.encode("utf-8")))


def _chunk_attribution(answer_tokens: list[str], documents: list[Document]) -> float:
    """Share of chunks that contributed tokens to the answer."""
    answer_set = set(answer_tokens)
    if not (documents and answer_set):
        return 0.0
    attributed = sum(
        _safe_div(len(answer_set & set(_tokenize(doc.page_content))), len(answer_set))
        >= ATTRIBUTION_MIN_OVERLAP
        for doc in documents
    )
    return _clamp01(attributed / len(documents))


def record_answer_metrics(
    question: str,
    answer: str,
    documents: list[Document],
    context: str,
    settings: Settings,
) -> None:
    """Record generation-quality, context-cliff and cost metrics for an answer."""
    answer_tokens = _tokenize(answer)
    reference_tokens = _tokenize(documents[0].page_content if documents else "")

    EVAL_CHUNK_ATTRIBUTION.observe(_chunk_attribution(answer_tokens, documents))

    similarity = _token_set_similarity(answer_tokens, reference_tokens)
    bleu = _bleu_unigram(answer_tokens[:BLEU_MAX_TOKENS], reference_tokens[:BLEU_MAX_TOKENS])
    rouge_l = _rouge_l(answer_tokens, reference_tokens)
    EVAL_ANSWER_SIMILARITY.observe(similarity)
    EVAL_ANSWER_CORRECTNESS.observe(_clamp01((bleu + rouge_l + similarity) / 3))

    context_kb = len(context.encode("utf-8")) / 1024
    EVAL_CONTEXT_CLIFF.observe(_clamp01((context_kb / CONTEXT_CLIFF_KB) * (1.0 - similarity)))

    prompt_tokens = len(_tokenize(question)) + len(_tokenize(context))
    estimated_cost = (prompt_tokens / 1000.0) * settings.rag_eval_input_cost_per_1k + (
        len(answer_tokens) / 1000.0
    ) * settings.rag_eval_output_cost_per_1k
    EVAL_COST_PER_INFERENCE.observe(max(0.0, estimated_cost))


# ── RAGAS evaluation (background) ────────────────────────────────────────────


def _run_ragas_evaluation(
    question: str,
    answer: str,
    contexts: list[str],
    settings: Settings,
) -> None:
    try:
        from langchain_ollama import ChatOllama as _ChatOllama
        from langchain_ollama import OllamaEmbeddings as _OllamaEmbeddings
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.dataset_schema import EvaluationResult
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import AnswerRelevancy, Faithfulness

        llm_wrapper = LangchainLLMWrapper(
            _ChatOllama(
                model=settings.ollama_llm_model, base_url=settings.ollama_base_url, temperature=0.0
            )
        )
        embed_wrapper = LangchainEmbeddingsWrapper(
            _OllamaEmbeddings(model=settings.ollama_embed_model, base_url=settings.ollama_base_url)
        )
        sample = SingleTurnSample(user_input=question, response=answer, retrieved_contexts=contexts)
        result = evaluate(
            dataset=EvaluationDataset(samples=[sample]),
            metrics=[Faithfulness(), AnswerRelevancy()],
            llm=llm_wrapper,
            embeddings=embed_wrapper,
        )
        # evaluate() only returns an Executor when called with return_executor=True.
        row = cast("EvaluationResult", result).to_pandas().iloc[0].to_dict()
        faith = float(row.get("faithfulness") or 0)
        relevancy = float(row.get("answer_relevancy") or 0)
        EVAL_FAITHFULNESS.observe(faith)
        EVAL_ANSWER_RELEVANCY.observe(relevancy)
        logger.info(
            "ragas_evaluation",
            faithfulness=round(faith, 3),
            answer_relevancy=round(relevancy, 3),
        )
    except Exception as exc:
        context_tokens: list[str] = []
        for ctx in contexts:
            context_tokens.extend(_tokenize(ctx))
        answer_tokens = _tokenize(answer)
        question_tokens = _tokenize(question)
        faith_proxy = _token_set_similarity(answer_tokens, context_tokens)
        relevancy_proxy = _token_set_similarity(answer_tokens, question_tokens)
        EVAL_FAITHFULNESS.observe(faith_proxy)
        EVAL_ANSWER_RELEVANCY.observe(relevancy_proxy)
        logger.warning("ragas_evaluation_failed", error=str(exc))


def schedule_ragas_evaluation(
    question: str,
    answer: str,
    contexts: list[str],
    settings: Settings,
) -> None:
    """Queue RAGAS eval in a bounded executor to avoid spawning unbounded threads."""
    try:
        _RAGAS_EXECUTOR.submit(_run_ragas_evaluation, question, answer, contexts, settings)
    except Exception as exc:
        logger.warning("ragas_eval_schedule_failed", error=str(exc))
