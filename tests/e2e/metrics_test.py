#!/usr/bin/env python3
"""
Checks RAG eval metrics end-to-end from the API `/metrics` endpoint.

Optionally sends a few `/api/v1/ask` requests first to generate fresh samples.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict


def _load_rag_eval_metrics() -> list[str]:
    """Return currently instrumented rag_eval metric base names."""
    try:
        from src.observability import metrics as prom_metrics

        names: list[str] = []
        for attr in dir(prom_metrics):
            if not attr.startswith("EVAL_"):
                continue
            metric = getattr(prom_metrics, attr)
            name = getattr(metric, "_name", "")
            if isinstance(name, str) and name.startswith("rag_eval_"):
                names.append(name)
        if names:
            return sorted(set(names))
    except Exception as exc:
        # Fallback for execution contexts where src.* cannot be imported.
        print(f"(src.* not importable, using the built-in metric list: {exc})")

    return [
        "rag_eval_contextual_precision",
        "rag_eval_contextual_recall",
        "rag_eval_faithfulness",
        "rag_eval_answer_relevancy",
        "rag_eval_chunk_utilization_ratio",
        "rag_eval_chunk_attribution_ratio",
        "rag_eval_answer_correctness",
        "rag_eval_answer_similarity",
        "rag_eval_abstention_ratio",
        "rag_eval_context_cliff_score",
        "rag_eval_stage_latency_seconds",
        "rag_eval_cost_per_inference",
    ]


RAG_EVAL_METRICS = _load_rag_eval_metrics()

WARMUP_QUESTIONS = [
    "¿Cuál fue el EBITDA de IBERDROLA en 2024?",
    "Compara ingresos y deuda de BBVA y SANTANDER en 2024.",
    "¿Qué guidance y dividendo reportó ENDESA en su último informe?",
]


def _http_get(url: str, timeout: float) -> tuple[int, str]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body
    except Exception as exc:
        return 0, str(exc)


def _http_post_json(url: str, payload: dict, timeout: float) -> tuple[int, dict | str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw
    except Exception as exc:
        return 0, str(exc)


def _parse_prometheus_text(text: str) -> dict[str, float]:
    """
    Returns metric_name -> aggregated value.
    Aggregates all samples of the same metric name (important for labeled histograms).
    """
    values: dict[str, float] = defaultdict(float)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        lhs, rhs = parts
        metric_name = lhs.split("{", 1)[0]
        try:
            values[metric_name] += float(rhs)
        except ValueError:
            continue
    return dict(values)


def _warmup_queries(api_url: str, endpoint: str, rounds: int, timeout: float) -> None:
    run_id = int(time.time())
    print(f"Warmup: sending {rounds} ask requests to generate metric samples...")
    for i in range(rounds):
        q = f"{WARMUP_QUESTIONS[i % len(WARMUP_QUESTIONS)]} [metrics-check:{run_id}-{i}]"
        payload = {"question": q, "thread_id": f"metrics-check-{run_id}"}
        code, body = _http_post_json(f"{api_url}{endpoint}", payload, timeout=timeout)
        route = body.get("route") if isinstance(body, dict) else None
        print(f"  - [{i + 1:02d}] status={code} route={route or '-'}")


def _collect_metrics_with_wait(
    metrics_url: str,
    timeout: float,
    wait_seconds: float,
) -> tuple[dict[str, float], str]:
    deadline = time.time() + max(0.0, wait_seconds)
    while True:
        code, text = _http_get(metrics_url, timeout=timeout)
        if code != 200:
            raise RuntimeError(f"GET {metrics_url} failed: {code} {text[:200]}")
        values = _parse_prometheus_text(text)
        # Enough signal when at least one eval metric has count > 0.
        if any(values.get(f"{m}_count", 0.0) > 0 for m in RAG_EVAL_METRICS):
            return values, text
        if time.time() >= deadline:
            return values, text
        time.sleep(1.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check RAG eval metrics in /metrics (presence + activity).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--api-url", default=os.getenv("API_URL", "http://localhost:8080"))
    parser.add_argument("--endpoint", default="/api/v1/ask")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--poke",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send warmup ask requests before reading /metrics",
    )
    parser.add_argument("--rounds", type=int, default=3, help="Warmup requests")
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=20.0,
        help="Max wait for non-zero metric counts (for async evaluators)",
    )
    parser.add_argument(
        "--require-nonzero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit non-zero if any rag_eval metric has count == 0",
    )
    args = parser.parse_args()

    metrics_url = f"{args.api_url}/metrics"

    if args.poke and args.rounds > 0:
        _warmup_queries(args.api_url, args.endpoint, args.rounds, timeout=args.timeout)

    values, _raw = _collect_metrics_with_wait(
        metrics_url=metrics_url,
        timeout=args.timeout,
        wait_seconds=args.wait_seconds,
    )

    missing: list[str] = []
    zero_count: list[str] = []

    print("\nRAG eval metrics status:")
    for base in RAG_EVAL_METRICS:
        count_name = f"{base}_count"
        sum_name = f"{base}_sum"
        has_count = count_name in values
        has_sum = sum_name in values
        if not has_count or not has_sum:
            missing.append(base)
            print(f"  - {base:<36} MISSING")
            continue

        count_val = values.get(count_name, 0.0)
        sum_val = values.get(sum_name, 0.0)
        avg_val = (sum_val / count_val) if count_val > 0 else 0.0
        status = "OK" if count_val > 0 else "ZERO_COUNT"
        if count_val <= 0:
            zero_count.append(base)
        print(
            f"  - {base:<36} {status:<10} count={count_val:.0f} sum={sum_val:.6f} avg={avg_val:.6f}"
        )

    print("\nSummary:")
    print(f"  missing    : {len(missing)}")
    print(f"  zero_count : {len(zero_count)}")

    if missing:
        print("  result     : FAIL (missing metrics)")
        return 2
    if args.require_nonzero and zero_count:
        print("  result     : FAIL (metrics without samples)")
        return 1
    print("  result     : PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
