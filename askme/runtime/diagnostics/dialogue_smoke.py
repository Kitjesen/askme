"""Real runtime dialogue plus memory retrieval smoke checks."""

from __future__ import annotations

import asyncio
import copy
import json
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import askme.interfaces.register_defaults  # noqa: F401 - register runtime backends
from askme.api.services.conversation_service import ConversationService
from askme.config import get_config
from askme.runtime.core.profiles import TEXT_PROFILE

DEFAULT_MESSAGE = "What is Thunder's current test identifier? Answer only the identifier."
DEFAULT_MEMORY_TEMPLATE = (
    "Thunder's current test identifier is {token}. "
    "This temporary record exists only for the AskMe runtime dialogue smoke test."
)


def run_dialogue_smoke_sync(**kwargs: Any) -> dict[str, Any]:
    """Run the async smoke from the synchronous CLI."""

    return asyncio.run(run_dialogue_smoke(**kwargs))


def run_dialogue_burst_sync(**kwargs: Any) -> dict[str, Any]:
    """Run the async burst evaluator from the synchronous CLI."""

    return asyncio.run(run_dialogue_burst(**kwargs))


async def run_dialogue_burst(
    *,
    fake_runs: int = 5,
    real_runs: int = 1,
    output_dir: str | Path = "",
    token_prefix: str = "",
    chat_timeout_s: float = 90.0,
    memory_timeout_s: float = 30.0,
    vector_min_similarity: float = 0.1,
    allow_reply_without_token: bool = False,
) -> dict[str, Any]:
    """Run repeated real-machine dialogue smokes and aggregate the evidence."""

    if fake_runs < 0 or real_runs < 0:
        raise ValueError("fake_runs and real_runs must be non-negative")

    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = _safe_token_prefix(token_prefix or f"ASKME-BURST-{stamp}")
    out_dir = Path(output_dir) if output_dir else Path("artifacts") / "runtime-dialogue-smoke" / "burst"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for index in range(1, fake_runs + 1):
        runs.append(
            await _run_burst_case(
                kind="fake",
                index=index,
                token=f"{prefix}-FAKE-{index:03d}",
                output_dir=out_dir / f"fake-{index:03d}",
                fake_llm=True,
                chat_timeout_s=chat_timeout_s,
                memory_timeout_s=memory_timeout_s,
                vector_min_similarity=vector_min_similarity,
                require_reply_token=not allow_reply_without_token,
            )
        )

    for index in range(1, real_runs + 1):
        runs.append(
            await _run_burst_case(
                kind="real",
                index=index,
                token=f"{prefix}-REAL-{index:03d}",
                output_dir=out_dir / f"real-{index:03d}",
                fake_llm=False,
                chat_timeout_s=chat_timeout_s,
                memory_timeout_s=memory_timeout_s,
                vector_min_similarity=vector_min_similarity,
                require_reply_token=not allow_reply_without_token,
            )
        )

    expected_runs = fake_runs + real_runs
    contract_checks = _burst_contract_checks(runs, expected_runs=expected_runs)
    status = "passed" if all(contract_checks.values()) else "failed"
    durations = [float(run.get("elapsed_ms") or 0.0) for run in runs]
    report = {
        "status": status,
        "failure_reason": "" if status == "passed" else _burst_failure_reason(contract_checks, runs),
        "token_prefix": prefix,
        "counts": {
            "expected": expected_runs,
            "total": len(runs),
            "passed": sum(1 for run in runs if run.get("status") == "passed"),
            "failed": sum(1 for run in runs if run.get("status") != "passed"),
            "fake": fake_runs,
            "real": real_runs,
        },
        "contract_checks": contract_checks,
        "timing_ms": _duration_stats(durations),
        "paths": {
            "output_dir": str(out_dir),
            "report": str(out_dir / "burst-report.json"),
        },
        "runs": runs,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
    }
    (out_dir / "burst-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


async def run_dialogue_smoke(
    *,
    message: str = DEFAULT_MESSAGE,
    memory_text: str = "",
    memory_query: str = "",
    output_dir: str | Path = "",
    data_dir: str | Path = "",
    token: str = "",
    chat_timeout_s: float = 90.0,
    memory_timeout_s: float = 30.0,
    vector_min_similarity: float = 0.1,
    fake_llm: bool = False,
    require_reply_token: bool = True,
) -> dict[str, Any]:
    """Exercise ConversationService -> TextLoop -> BrainPipeline with RAG.

    The smoke writes a temporary knowledge record into an isolated vector store,
    verifies direct retrieval, then sends a real runtime chat turn and checks
    the product chat payload contains the same evidence.
    """

    started = time.perf_counter()
    run_token = str(token or f"ASKME-LIVE-{secrets.token_hex(4).upper()}").strip()
    run_id = _safe_run_id(run_token)
    out_dir = Path(output_dir) if output_dir else Path("artifacts") / "runtime-dialogue-smoke" / run_id
    out_dir = out_dir.resolve()
    isolated_data_dir = Path(data_dir).resolve() if data_dir else (out_dir / "data").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    isolated_data_dir.mkdir(parents=True, exist_ok=True)

    memory_fact = memory_text.strip() or DEFAULT_MEMORY_TEMPLATE.format(token=run_token)
    query = memory_query.strip() or message
    cfg = _smoke_config(
        get_config(reload=True),
        data_dir=isolated_data_dir,
        memory_timeout_s=memory_timeout_s,
        vector_min_similarity=vector_min_similarity,
        fake_llm=fake_llm,
        token=run_token,
    )

    app = None
    import_payload: dict[str, Any] = {}
    direct_context = ""
    direct_health: dict[str, Any] = {}
    chat_payload: dict[str, Any] = {}
    service_diagnostics: dict[str, Any] = {}
    app_health: dict[str, Any] = {}
    failure_reason = ""

    try:
        text_blueprint = __import__(
            "askme.blueprints.presets.text", fromlist=["text"]
        ).text

        app = await text_blueprint.build(cfg)
        await app.start()
        memory_module = app.modules.get("memory")
        text_module = app.modules.get("text")
        memory_bridge = getattr(memory_module, "memory_bridge", None) if memory_module else None
        text_loop = getattr(text_module, "text_loop", None) if text_module else None
        if memory_bridge is None or text_loop is None:
            raise RuntimeError("text runtime did not expose memory_bridge and text_loop")

        seed_path = _write_memory_seed(out_dir, text=memory_fact, token=run_token)
        import_payload = await memory_module.import_payload(
            {
                "content": seed_path.read_text(encoding="utf-8"),
                "filename": seed_path.name,
                "source": "runtime-dialogue-smoke",
                "category": "customer_support",
                "quality_status": "public",
                "visibility": "external",
                "owner": "runtime-dialogue-smoke",
            }
        )

        direct_context = await memory_bridge.retrieve(query)
        direct_health = memory_bridge.health()

        service = ConversationService(
            chat_handler=text_loop.process_turn,
            memory_handler=memory_bridge,
            chat_timeout_s=chat_timeout_s,
            chat_slow_threshold_ms=2000.0,
        )
        chat_payload = await service.chat_payload_from_body(
            {
                "text": message,
                "conversation_session_id": f"dialogue-smoke-{run_id}",
                "runtime_policy": "disabled",
                "speak": False,
            },
            trace_id=f"dialogue-smoke-{run_id}",
        )
        service_diagnostics = service.diagnostics_snapshot()
        app_health = _compact_app_health(app.health())
    except Exception as exc:  # pragma: no cover - real diagnostic path
        failure_reason = f"{type(exc).__name__}: {exc}"
    finally:
        if app is not None:
            try:
                await app.stop()
            except Exception as exc:  # pragma: no cover - shutdown best effort
                stop_error = f"stop_error={type(exc).__name__}: {exc}"
                failure_reason = f"{failure_reason}; {stop_error}" if failure_reason else stop_error

    checks = _build_checks(
        token=run_token,
        import_payload=import_payload,
        direct_context=direct_context,
        direct_health=direct_health,
        chat_payload=chat_payload,
        failure_reason=failure_reason,
        require_reply_token=require_reply_token,
    )
    status = "passed" if all(checks.values()) and not failure_reason else "failed"
    report = {
        "status": status,
        "failure_reason": failure_reason or _first_failed_check(checks),
        "run_id": run_id,
        "token": run_token,
        "profile": TEXT_PROFILE.snapshot(),
        "message": message,
        "memory_query": query,
        "paths": {
            "output_dir": str(out_dir),
            "data_dir": str(isolated_data_dir),
            "vector_store": str(isolated_data_dir / "memory" / "vectors" / "store.json"),
            "seed_file": str(out_dir / "memory-seed.json"),
        },
        "config_overrides": {
            "memory.backend": "vector",
            "memory.customer_knowledge_backend": "vector",
            "memory.retrieve_timeout": memory_timeout_s,
            "memory.vector_min_similarity": vector_min_similarity,
            "brain.provider": "fake" if fake_llm else cfg.get("brain", {}).get("provider", ""),
        },
        "checks": checks,
        "import": import_payload,
        "memory_retrieval": {
            "context": direct_context,
            "health": _memory_health_summary(direct_health),
        },
        "chat": {
            "reply": chat_payload.get("reply", ""),
            "reply_preview": str(chat_payload.get("reply", ""))[:240],
            "evidence": chat_payload.get("evidence", []),
            "rag": chat_payload.get("rag", {}),
            "diagnostics": service_diagnostics,
        },
        "runtime_health": app_health,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
    }
    _write_report(out_dir, report)
    return report


def print_dialogue_smoke_summary(payload: dict[str, Any]) -> None:
    """Print a compact human-readable summary."""

    print(f"Runtime dialogue smoke: {payload.get('status', 'unknown')}")  # noqa: T201
    print(f"  token: {payload.get('token', '')}")  # noqa: T201
    for name, passed in (payload.get("checks") or {}).items():
        print(f"  {name}: {'ok' if passed else 'failed'}")  # noqa: T201
    failure = payload.get("failure_reason")
    if failure:
        print(f"  failure_reason: {failure}")  # noqa: T201
    reply = str((payload.get("chat") or {}).get("reply") or "")
    if reply:
        print(f"  reply: {reply[:160]}")  # noqa: T201
    print(f"  report: {(payload.get('paths') or {}).get('output_dir', '')}")  # noqa: T201


def print_dialogue_burst_summary(payload: dict[str, Any]) -> None:
    """Print a compact human-readable burst summary."""

    counts = payload.get("counts") or {}
    timing = payload.get("timing_ms") or {}
    print(f"Runtime dialogue burst: {payload.get('status', 'unknown')}")  # noqa: T201
    print(  # noqa: T201
        "  runs: "
        f"{counts.get('passed', 0)}/{counts.get('total', 0)} passed "
        f"(fake={counts.get('fake', 0)}, real={counts.get('real', 0)})"
    )
    print(  # noqa: T201
        "  timing_ms: "
        f"min={timing.get('min', 0)} p50={timing.get('p50', 0)} "
        f"p95={timing.get('p95', 0)} max={timing.get('max', 0)}"
    )
    for name, passed in (payload.get("contract_checks") or {}).items():
        print(f"  {name}: {'ok' if passed else 'failed'}")  # noqa: T201
    failure = payload.get("failure_reason")
    if failure:
        print(f"  failure_reason: {failure}")  # noqa: T201
    print(f"  report: {(payload.get('paths') or {}).get('report', '')}")  # noqa: T201


async def _run_burst_case(
    *,
    kind: str,
    index: int,
    token: str,
    output_dir: Path,
    fake_llm: bool,
    chat_timeout_s: float,
    memory_timeout_s: float,
    vector_min_similarity: float,
    require_reply_token: bool,
) -> dict[str, Any]:
    message = DEFAULT_MESSAGE
    memory_text = DEFAULT_MEMORY_TEMPLATE.format(token=token)
    try:
        report = await run_dialogue_smoke(
            message=message,
            memory_text=memory_text,
            memory_query=message,
            output_dir=output_dir,
            token=token,
            chat_timeout_s=chat_timeout_s,
            memory_timeout_s=memory_timeout_s,
            vector_min_similarity=vector_min_similarity,
            fake_llm=fake_llm,
            require_reply_token=require_reply_token,
        )
    except Exception as exc:  # pragma: no cover - burst aggregation fallback
        report = {
            "status": "failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "token": token,
            "checks": {},
            "paths": {"output_dir": str(output_dir)},
            "chat": {"reply_preview": ""},
            "elapsed_ms": 0.0,
        }

    chat = report.get("chat") or {}
    return {
        "kind": kind,
        "index": index,
        "status": report.get("status", "failed"),
        "token": report.get("token", token),
        "failure_reason": report.get("failure_reason", ""),
        "elapsed_ms": report.get("elapsed_ms", 0.0),
        "checks": report.get("checks", {}),
        "reply_preview": chat.get("reply_preview", chat.get("reply", "")),
        "output_dir": (report.get("paths") or {}).get("output_dir", str(output_dir)),
    }


def _burst_contract_checks(runs: list[dict[str, Any]], *, expected_runs: int) -> dict[str, bool]:
    return {
        "expected_run_count": len(runs) == expected_runs,
        "all_runs_passed": expected_runs > 0 and all(run.get("status") == "passed" for run in runs),
        "memory_context_all_passed": _all_run_check(runs, "memory_context_contains_token"),
        "chat_rag_all_passed": _all_run_check(runs, "chat_payload_has_rag"),
        "chat_evidence_all_passed": _all_run_check(runs, "chat_evidence_contains_token"),
        "chat_reply_token_all_passed": _all_run_check(runs, "chat_reply_contains_token"),
    }


def _all_run_check(runs: list[dict[str, Any]], check_name: str) -> bool:
    return bool(runs) and all(bool((run.get("checks") or {}).get(check_name)) for run in runs)


def _burst_failure_reason(contract_checks: dict[str, bool], runs: list[dict[str, Any]]) -> str:
    failed_contract = [name for name, passed in contract_checks.items() if not passed]
    failed_runs = [
        f"{run.get('kind')}#{run.get('index')}:{run.get('failure_reason') or run.get('status')}"
        for run in runs
        if run.get("status") != "passed"
    ]
    return "; ".join(failed_contract + failed_runs)


def _duration_stats(values: list[float]) -> dict[str, float]:
    clean = sorted(value for value in values if value >= 0)
    if not clean:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "avg": 0.0}
    return {
        "min": round(clean[0], 1),
        "p50": round(_percentile(clean, 50), 1),
        "p95": round(_percentile(clean, 95), 1),
        "max": round(clean[-1], 1),
        "avg": round(sum(clean) / len(clean), 1),
    }


def _percentile(sorted_values: list[float], percentile: int) -> float:
    if not sorted_values:
        return 0.0
    index = max(0, min(len(sorted_values) - 1, round((percentile / 100.0) * (len(sorted_values) - 1))))
    return sorted_values[index]


def _smoke_config(
    base_config: dict[str, Any],
    *,
    data_dir: Path,
    memory_timeout_s: float,
    vector_min_similarity: float,
    fake_llm: bool,
    token: str,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("app", {})["data_dir"] = str(data_dir)
    memory_cfg = cfg.setdefault("memory", {})
    memory_cfg.update(
        {
            "enabled": True,
            "backend": "vector",
            "customer_knowledge_backend": "vector",
            "robotmem_fallback_backend": "vector",
            "mempalace_fallback_backend": "vector",
            "retrieve_timeout": float(memory_timeout_s),
            "retrieve_cache_ttl_s": 0.0,
            "vector_min_similarity": float(vector_min_similarity),
            "rag_enforce_expiry": True,
        }
    )
    if fake_llm:
        brain_cfg = cfg.setdefault("brain", {})
        brain_cfg["provider"] = "fake"
        brain_cfg["api_key"] = "fake"
        brain_cfg["base_url"] = "http://fake.local/v1"
        brain_cfg["model"] = "fake-dialogue-smoke"
        brain_cfg["voice_model"] = "fake-dialogue-smoke"
        brain_cfg["provider_options"] = {
            **dict(brain_cfg.get("provider_options") or {}),
            "response_text": token,
        }
    return cfg


def _write_memory_seed(output_dir: Path, *, text: str, token: str) -> Path:
    seed_path = output_dir / "memory-seed.json"
    payload = {
        "records": [
            {
                "text": text,
                "source": "runtime-dialogue-smoke",
                "category": "customer_support",
                "approval_status": "published",
                "visibility": "external",
                "quality_status": "public",
                "record_id": f"dialogue_smoke_{_safe_run_id(token)}",
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        ]
    }
    seed_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return seed_path


def _build_checks(
    *,
    token: str,
    import_payload: dict[str, Any],
    direct_context: str,
    direct_health: dict[str, Any],
    chat_payload: dict[str, Any],
    failure_reason: str,
    require_reply_token: bool,
) -> dict[str, bool]:
    reply = str(chat_payload.get("reply") or "")
    evidence = chat_payload.get("evidence")
    rag = chat_payload.get("rag")
    return {
        "runtime_completed": not bool(failure_reason),
        "knowledge_imported": int(import_payload.get("imported") or 0) > 0
        and not bool(import_payload.get("errors")),
        "memory_backend_available": bool(direct_health.get("available")),
        "memory_context_contains_token": token in direct_context,
        "chat_reply_nonempty": bool(reply.strip())
        and not reply.lstrip().startswith("[")
        and "system error" not in reply.lower(),
        "chat_payload_has_rag": isinstance(rag, dict)
        and bool(rag.get("enabled"))
        and int(rag.get("last_retrieved_items") or 0) > 0,
        "chat_evidence_contains_token": _token_in_evidence(token, evidence),
        "chat_reply_contains_token": token in reply if require_reply_token else True,
    }


def _token_in_evidence(token: str, evidence: Any) -> bool:
    if not isinstance(evidence, list):
        return False
    for item in evidence:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "")
        metadata = item.get("metadata")
        metadata_text = json.dumps(metadata, ensure_ascii=False) if isinstance(metadata, dict) else ""
        if token in text or token in metadata_text:
            return True
    return False


def _memory_health_summary(health: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "enabled",
        "backend",
        "configured_backend",
        "available",
        "selected_backend_ready",
        "vector_ready",
        "vector_store_path",
        "vector_size",
        "last_backend",
        "last_retrieve_ms",
        "last_retrieved_items",
        "last_fallback_reason",
        "last_evidence",
        "last_dropped_evidence",
        "last_answer_policy",
    )
    return {key: health.get(key) for key in keys if key in health}


def _compact_app_health(health: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for name, module_health in health.items():
        if not isinstance(module_health, dict):
            compact[name] = module_health
            continue
        compact[name] = {
            key: module_health.get(key)
            for key in ("status", "model", "conversation_len", "available", "backend")
            if key in module_health
        }
        if name == "memory":
            compact[name]["rag"] = _memory_health_summary(module_health.get("rag", module_health))
    return compact


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_run_id(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value))
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned[:80] or f"run-{secrets.token_hex(4)}"


def _safe_token_prefix(value: str) -> str:
    cleaned = "".join(ch.upper() if ch.isalnum() else "-" for ch in str(value))
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned[:56] or f"ASKME-BURST-{secrets.token_hex(3).upper()}"


def _first_failed_check(checks: dict[str, bool]) -> str:
    for name, passed in checks.items():
        if not passed:
            return name
    return ""
