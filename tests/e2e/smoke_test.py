#!/usr/bin/env python3
"""
Smoke test: checks that the whole IBEX35 RAG stack works.

Usage:
    uv run python tests/e2e/smoke_test.py
    make smoke
"""

import base64
import contextlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

API_URL = "http://localhost:8080"
PROMETHEUS_URL = "http://localhost:9090"
LOKI_URL = "http://localhost:3100"
GRAFANA_URL = "http://localhost:3000"
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"

GRAFANA_USER = os.getenv("GRAFANA_ADMIN_USER", "admin")
GRAFANA_PASS = os.getenv("GRAFANA_ADMIN_PASSWORD", "ibex35-rag")

EXPECTED_SERVICES = [
    "api",
    "ollama",
    "qdrant",
    "redis",
    "frontend",
    "prometheus",
    "grafana",
    "otel-collector",
    "tempo",
    "loki",
    "promtail",
    "node-exporter",
    "cadvisor",
    "alertmanager",
    "redis-exporter",
]
HEALTHY_SERVICES = {"ollama", "qdrant", "redis", "cadvisor"}

# Metrics that must always exist (gauges, initialised at API start-up)
REQUIRED_METRICS = [
    "rag_vector_store_documents_current",
    "rag_companies_indexed_current",
    "rag_pages_indexed_current",
    "rag_cache_keys_current",
    "rag_cache_bytes_used",
    "rag_top_k_current",
    "rag_ingestion_last_duration_seconds",
]
# Activity metrics (counters: only appear after queries/ingestions)
ACTIVITY_METRICS = [
    "rag_query_requests_total",
    "rag_cache_hits_total",
    "rag_cache_misses_total",
    "rag_ingestion_runs_total",
]

EXPECTED_SCRAPE_JOBS = [
    "ibex35-rag-api",
    "prometheus",
    "node-exporter",
    "cadvisor",
    "otel-collector",
    "redis-exporter",
]

EXPECTED_ALERT_RULES = [
    "RAGAPIDown",
    "RAGAPIHighLatency",
    "RAGCacheHitRateLow",
    "IngestionFailed",
    "VectorStoreEmpty",
    "RAGEmptyRetrievalHigh",
    "HighMemoryUsage",
    "HighCPUUsage",
    "NodeExporterDown",
    "RedisDown",
]

EXPECTED_LOKI_SERVICES = ["api", "redis", "prometheus", "grafana"]
# qdrant is not critical for this log check; validate these key services.

# ──────────────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

_results: list[tuple[bool, str]] = []  # (passed, description)


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET}  {msg}")
    _results.append((True, msg))


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET}  {msg}")
    _results.append((False, msg))


def warn(msg: str) -> None:
    print(f"  {YELLOW}~{RESET}  {msg}")


def section(title: str) -> None:
    print(f"\n{CYAN}{BOLD}[ {title} ]{RESET}")


# ──────────────────────────────────────────────────────────────────────────────
# HTTP helper
# ──────────────────────────────────────────────────────────────────────────────


def http_get(url: str, timeout: int = 5, auth: tuple[str, str] | None = None) -> tuple[int, Any]:
    """Returns (status_code, parsed_body). Body is dict/list if JSON else str."""
    req = urllib.request.Request(url)
    if auth:
        creds = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
        req.add_header("Authorization", f"Basic {creds}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            ct = resp.headers.get("Content-Type", "")
            if "application/json" in ct or raw.lstrip().startswith(("{", "[")):
                try:
                    return resp.status, json.loads(raw)
                except json.JSONDecodeError:
                    pass
            return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        return e.code, raw
    except Exception as exc:
        return 0, str(exc)


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    """Extract metric_name → last value from Prometheus text format."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        name_labels = parts[0]
        value_str = parts[1]
        # Strip labels: metric_name{...} → metric_name
        name = name_labels.split("{")[0]
        with contextlib.suppress(ValueError):  # skip non-numeric samples
            values[name] = float(value_str)
    return values


# ──────────────────────────────────────────────────────────────────────────────
# Checks
# ──────────────────────────────────────────────────────────────────────────────


def check_containers() -> None:
    section("Contenedores Docker")
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "{{.Name}}\t{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
        running: dict[str, str] = {}
        for line in lines:
            parts = line.split("\t", 1)
            if len(parts) == 2:
                # Container name is usually <project>-<service>-<index>
                raw_name, status = parts
                chunks = raw_name.split("-")
                service = "-".join(chunks[1:-1]) if len(chunks) >= 3 else raw_name
                running[service] = status

        for svc in EXPECTED_SERVICES:
            status = running.get(svc, "")
            if not status:
                fail(f"{svc:<20} NO ENCONTRADO")
            elif "Up" in status or "running" in status.lower():
                health_tag = ""
                if svc in HEALTHY_SERVICES:
                    if "healthy" in status.lower():
                        health_tag = " (healthy)"
                    else:
                        health_tag = f" {YELLOW}(not healthy yet){RESET}"
                ok(f"{svc:<20} {status.split('(')[0].strip()}{health_tag}")
            else:
                fail(f"{svc:<20} {status}")
    except FileNotFoundError:
        fail("docker compose no encontrado en PATH")
    except subprocess.TimeoutExpired:
        fail("timeout al ejecutar docker compose ps")


def check_api() -> None:
    section("API Backend  →  http://localhost:8080")

    # /health/live
    code, body = http_get(f"{API_URL}/health/live")
    if code == 200 and isinstance(body, dict) and body.get("status") == "ok":
        ok("/health/live → ok")
    else:
        fail(f"/health/live → {code} {str(body)[:60]}")

    # /health/ready
    code, body = http_get(f"{API_URL}/health/ready")
    if code == 200:
        ok("/health/ready → 200")
    else:
        fail(f"/health/ready → {code}")

    # /health
    code, body = http_get(f"{API_URL}/health", timeout=10)
    if code == 200 and isinstance(body, dict):
        status = body.get("status", "?")
        count = body.get("collection_count", 0)
        redis_ok = body.get("redis") == "healthy"
        vs_ok = body.get("vector_store") == "healthy"
        ollama_ok = body.get("ollama") == "healthy"
        color = GREEN if status == "healthy" else YELLOW
        marks = {
            name: "✓" if healthy else "✗"
            for name, healthy in (
                ("redis", redis_ok),
                ("vector_store", vs_ok),
                ("ollama", ollama_ok),
            )
        }
        ok(
            f"/health → {color}{status}{RESET} | chunks={count} | "
            + " ".join(f"{name}={mark}" for name, mark in marks.items())
        )
        if count == 0:
            fail("collection_count = 0  (ejecuta: make ingest)")
        else:
            ok(f"collection_count = {count}  (esperado > 0)")
    else:
        fail(f"/health → {code} {str(body)[:80]}")

    # /docs
    code, _ = http_get(f"{API_URL}/docs", timeout=5)
    if code == 200:
        ok("/docs → 200 (Swagger disponible)")
    else:
        fail(f"/docs → {code}")


def check_metrics() -> None:
    section("Métricas Prometheus  →  GET /metrics")
    code, body = http_get(f"{API_URL}/metrics", timeout=10)
    if code != 200 or not isinstance(body, str):
        fail(f"/metrics → {code}")
        return

    values = parse_prometheus_metrics(body)

    # Required metrics (gauges — always present)
    missing = [m for m in REQUIRED_METRICS if m not in values]
    if missing:
        for m in missing:
            fail(f"métrica ausente: {m}")
    else:
        ok(f"{len(REQUIRED_METRICS)} métricas Gauge presentes")

    # Activity metrics (counters — only after queries/ingestions)
    present_activity = [m for m in ACTIVITY_METRICS if m in values]
    absent_activity = [m for m in ACTIVITY_METRICS if m not in values]
    if absent_activity:
        warn(
            f"métricas sin actividad aún: {', '.join(m.split('rag_')[1] for m in absent_activity)}"
        )
    if present_activity:
        ok(f"métricas de actividad: {', '.join(m.split('rag_')[1] for m in present_activity)}")

    # Value checks
    def check_metric(name: str, op: str, threshold: float, label: str) -> None:
        val = values.get(name)
        if val is None:
            return  # already reported as missing or absent activity metric
        ops = {">": val > threshold, ">=": val >= threshold, "==": val == threshold}
        passed = ops.get(op, False)
        display = f"{label} = {val:.0f}"
        if passed:
            ok(display)
        else:
            fail(f"{display}  (esperado {op} {threshold:.0f})")

    check_metric("rag_companies_indexed_current", ">", 0, "companies_indexed")
    check_metric("rag_pages_indexed_current", ">", 0, "pages_indexed")
    check_metric("rag_vector_store_documents_current", ">", 0, "chunks_indexed")
    check_metric("rag_top_k_current", ">", 0, "top_k")
    check_metric("rag_cache_keys_current", ">=", 0, "cache_keys")
    check_metric("rag_cache_bytes_used", ">=", 0, "cache_bytes_used")


def _job_health(active_targets: list[dict[str, Any]]) -> dict[str, str]:
    """Health per scrape job; a job is "up" if any of its instances is up."""
    job_health: dict[str, str] = {}
    for target in active_targets:
        job = target.get("labels", {}).get("job", "")
        health = target.get("health", "unknown")
        if job not in job_health or health == "up":
            job_health[job] = health
    return job_health


def _check_scrape_targets() -> None:
    code, body = http_get(f"{PROMETHEUS_URL}/api/v1/targets", timeout=10)
    if code != 200 or not isinstance(body, dict):
        fail(f"/api/v1/targets → {code}")
        return

    job_health = _job_health(body.get("data", {}).get("activeTargets", []))
    for job in EXPECTED_SCRAPE_JOBS:
        health = job_health.get(job)
        if health == "up":
            ok(f"target {job:<28} → up")
        else:
            fail(f"target {job:<28} → {health or 'NO ENCONTRADO'}")


def _check_alert_rules() -> None:
    code, body = http_get(f"{PROMETHEUS_URL}/api/v1/rules", timeout=10)
    if code != 200 or not isinstance(body, dict):
        fail(f"/api/v1/rules → {code}")
        return

    loaded_rules = {
        rule.get("name", "")
        for group in body.get("data", {}).get("groups", [])
        for rule in group.get("rules", [])
        if rule.get("type") == "alerting"
    }
    missing_rules = [r for r in EXPECTED_ALERT_RULES if r not in loaded_rules]
    if missing_rules:
        fail(f"reglas ausentes: {', '.join(missing_rules)}")
    else:
        ok(f"{len(EXPECTED_ALERT_RULES)}/{len(EXPECTED_ALERT_RULES)} alert rules cargadas")


def check_prometheus() -> None:
    section("Prometheus  →  http://localhost:9090")
    _check_scrape_targets()
    _check_alert_rules()


def check_loki() -> None:
    section("Loki  →  http://localhost:3100")

    # Services with logs
    code, body = http_get(f"{LOKI_URL}/loki/api/v1/label/service/values", timeout=8)
    if code != 200 or not isinstance(body, dict):
        fail(f"/label/service/values → {code}")
        return

    services = set(body.get("data", []))
    shown = ", ".join(sorted(services)[:8]) + ("..." if len(services) > 8 else "")
    ok(f"{len(services)} servicios con logs: {shown}")

    missing = [s for s in EXPECTED_LOKI_SERVICES if s not in services]
    if missing:
        fail(f"servicios sin logs: {', '.join(missing)}")
    else:
        ok(f"servicios clave presentes: {', '.join(EXPECTED_LOKI_SERVICES)}")

    # Recent logs from api
    now_ns = int(time.time() * 1e9)
    start_ns = now_ns - 5 * 60 * int(1e9)  # last 5 min
    params = urllib.parse.urlencode(
        {
            "query": '{service="api"}',
            "start": str(start_ns),
            "end": str(now_ns),
            "limit": "5",
        }
    )
    code, body = http_get(f"{LOKI_URL}/loki/api/v1/query_range?{params}", timeout=8)
    if code == 200 and isinstance(body, dict):
        streams = body.get("data", {}).get("result", [])
        total = sum(len(s.get("values", [])) for s in streams)
        if total > 0:
            ok(f"api tiene logs recientes (últimos 5m: {total} entradas)")
        else:
            warn("api sin logs en los últimos 5 minutos (normal si no hay tráfico)")
    else:
        fail(f"query_range api → {code}")


def check_grafana() -> None:
    section("Grafana  →  http://localhost:3000")

    # Health
    code, body = http_get(f"{GRAFANA_URL}/api/health", timeout=8)
    if code == 200 and isinstance(body, dict) and body.get("database") == "ok":
        ok(f"database ok (version: {body.get('version', '?')})")
    else:
        fail(f"/api/health → {code} {str(body)[:60]}")
        return

    # Dashboard
    code, body = http_get(
        f"{GRAFANA_URL}/api/dashboards/uid/ibex35-todo-en-uno",
        timeout=8,
        auth=(GRAFANA_USER, GRAFANA_PASS),
    )
    if code == 200 and isinstance(body, dict):
        panels = body.get("dashboard", {}).get("panels", [])
        title = body.get("dashboard", {}).get("title", "?")
        if len(panels) >= 40:
            ok(f"dashboard '{title}': {len(panels)} panels")
        else:
            fail(f"dashboard '{title}': solo {len(panels)} panels (esperado ≥ 40)")
    else:
        fail(f"dashboard ibex35-todo-en-uno → {code} {str(body)[:80]}")

    # Data sources
    code, body = http_get(
        f"{GRAFANA_URL}/api/datasources",
        timeout=8,
        auth=(GRAFANA_USER, GRAFANA_PASS),
    )
    if code == 200 and isinstance(body, list):
        ds_names = {d.get("name") for d in body}
        for expected in ("Prometheus", "Loki", "Tempo"):
            if expected in ds_names:
                ok(f"datasource {expected} configurado")
            else:
                fail(f"datasource {expected} NO encontrado")
    else:
        fail(f"/api/datasources → {code}")


def check_langsmith() -> None:
    section("LangSmith  →  tracing + métricas LLM")

    try:
        from src.core.config import get_settings

        settings = get_settings()
    except Exception as exc:
        fail(f"no se pudo cargar settings: {exc}")
        return

    if not settings.langsmith_tracing:
        warn("LANGSMITH_TRACING != true — tracing desactivado, saltando check")
        return
    if not settings.langsmith_api_key:
        fail("LANGSMITH_API_KEY no configurada")
        return

    api_key = settings.langsmith_api_key.get_secret_value()
    endpoint = settings.langsmith_endpoint
    project = settings.langsmith_project

    try:
        from langsmith import Client  # type: ignore[import]
    except ImportError:
        fail("langsmith no instalado (uv add langsmith)")
        return

    try:
        client = Client(api_key=api_key, api_url=endpoint)
        projects = list(client.list_projects(limit=10))
        project_names = {p.name for p in projects}
        ok(f"API accesible ({endpoint}) — {len(projects)} proyecto/s encontrado/s")
    except Exception as exc:
        fail(f"no se pudo conectar a LangSmith: {exc}")
        return

    if project in project_names:
        ok(f"proyecto '{project}' existe en LangSmith")
    else:
        warn(f"proyecto '{project}' aún no existe (se creará con el primer run LLM)")

    try:
        runs = list(client.list_runs(project_name=project, run_type="llm", limit=1))
        if runs:
            ok("proyecto tiene runs LLM registrados")
        else:
            warn(f"proyecto '{project}' sin runs LLM aún (normal en primer arranque)")
    except Exception:
        warn(f"proyecto '{project}' sin runs aún")


def check_services() -> None:
    section("Servicios Base")

    # Qdrant
    code, body = http_get(f"{QDRANT_URL}/collections", timeout=8)
    if code == 200 and isinstance(body, dict):
        collections = body.get("result", {}).get("collections", [])
        ok(f"Qdrant /collections → ok ({len(collections)} colección/es)")

        collection = os.getenv("QDRANT_COLLECTION", "ibex35")
        code_c, body_c = http_get(f"{QDRANT_URL}/collections/{collection}", timeout=8)
        if code_c == 200 and isinstance(body_c, dict):
            points = body_c.get("result", {}).get("points_count", 0)
            if points > 0:
                ok(f"Qdrant colección '{collection}' con {points} vectores")
            else:
                fail(f"Qdrant colección '{collection}' vacía (ejecuta: make ingest)")
        else:
            fail(f"Qdrant colección '{collection}' → {code_c}")
    else:
        fail(f"Qdrant /collections → {code} {str(body)[:60]}")

    # Ollama
    code, body = http_get(f"{OLLAMA_URL}/api/tags", timeout=10)
    if code == 200 and isinstance(body, dict):
        models = [m.get("name", "") for m in body.get("models", [])]
        llm_model = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
        embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

        llm_present = any(llm_model.split(":")[0] in m for m in models)
        embed_present = any(embed_model.split(":")[0] in m for m in models)

        models_short = [m.split(":")[0] for m in models]
        ok(f"Ollama modelos disponibles: {', '.join(models_short)}")

        if llm_present:
            ok(f"LLM model '{llm_model}' presente")
        else:
            fail(f"LLM model '{llm_model}' NO encontrado (disponibles: {models_short})")

        if embed_present:
            ok(f"Embed model '{embed_model.split(':')[0]}' presente")
        else:
            fail(f"Embed model '{embed_model}' NO encontrado")
    else:
        fail(f"Ollama /api/tags → {code} {str(body)[:60]}")

    # Redis (via API health already covers this, but check port directly)
    import socket

    try:
        with socket.create_connection(("localhost", 6379), timeout=3):
            ok("Redis puerto 6379 → accesible")
    except Exception as e:
        fail(f"Redis puerto 6379 → {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> int:
    width = 50
    print(f"\n{BOLD}{'═' * width}{RESET}")
    print(f"{BOLD}   IBEX35 RAG — Smoke Test{RESET}")
    print(f"{BOLD}{'═' * width}{RESET}")

    check_containers()
    check_api()
    check_metrics()
    check_prometheus()
    check_loki()
    check_grafana()
    check_services()
    check_langsmith()

    # Summary
    passed = sum(1 for ok, _ in _results if ok)
    total = len(_results)
    failed = total - passed

    print(f"\n{BOLD}{'═' * width}{RESET}")
    if failed == 0:
        print(f"{BOLD}{GREEN}  RESULTADO: {passed}/{total} checks OK  — TODO BIEN{RESET}")
    else:
        fails = [desc for p, desc in _results if not p]
        plural = "S" if failed > 1 else ""
        print(f"{BOLD}{RED}  RESULTADO: {passed}/{total} OK  — {failed} FALLO{plural}{RESET}")
        for desc in fails:
            print(f"    {RED}✗{RESET} {desc}")
    print(f"{BOLD}{'═' * width}{RESET}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
