"""Benchmark indexed audit JSONL queries against full JSONL scans."""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from askme.audit.query import (
    AuditPaths,
    AuditQueryService,
    _matches,
    _normalize_field_action,
    _public_record,
    _read_jsonl,
    _time_boundary,
)


DEFAULT_QUICK_ITERATIONS = 60
DEFAULT_FULL_ITERATIONS = 240
DEFAULT_QUICK_CONCURRENCY = 8
DEFAULT_FULL_CONCURRENCY = 16
TAIL_LIMIT = 1000


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def summarize_load(
    name: str,
    *,
    attempts: int,
    concurrency: int,
    successful_latencies_ms: list[float],
    errors: int,
    timeouts: int,
    duration_s: float,
    scanned_records: list[int],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    successes = len(successful_latencies_ms)
    safe_attempts = max(1, attempts)
    safe_duration = max(duration_s, 0.000001)
    return {
        "name": name,
        "attempts": attempts,
        "successes": successes,
        "errors": errors,
        "timeouts": timeouts,
        "concurrency": concurrency,
        "duration_s": round(duration_s, 4),
        "throughput_per_s": round(successes / safe_duration, 3),
        "attempt_throughput_per_s": round(attempts / safe_duration, 3),
        "error_rate": round(errors / safe_attempts, 6),
        "timeout_rate": round(timeouts / safe_attempts, 6),
        "p50_ms": percentile(successful_latencies_ms, 0.50),
        "p95_ms": percentile(successful_latencies_ms, 0.95),
        "p99_ms": percentile(successful_latencies_ms, 0.99),
        "min_ms": round(min(successful_latencies_ms), 3) if successful_latencies_ms else 0.0,
        "max_ms": round(max(successful_latencies_ms), 3) if successful_latencies_ms else 0.0,
        "avg_scanned_records": round(sum(scanned_records) / max(1, len(scanned_records)), 3),
        "max_scanned_records": max(scanned_records) if scanned_records else 0,
        "extra": extra or {},
    }


async def run_load(
    name: str,
    operation: Any,
    *,
    attempts: int,
    concurrency: int,
    timeout_s: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    latencies: list[float] = []
    scanned_records: list[int] = []
    errors = 0
    timeouts = 0

    async def one(index: int) -> None:
        nonlocal errors, timeouts
        async with semaphore:
            started = time.perf_counter()
            try:
                scanned = await asyncio.wait_for(
                    asyncio.to_thread(operation, index),
                    timeout=timeout_s,
                )
            except TimeoutError:
                errors += 1
                timeouts += 1
            except Exception:
                errors += 1
            else:
                scanned_records.append(int(scanned))
                latencies.append((time.perf_counter() - started) * 1000.0)

    started = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(attempts)))
    duration_s = time.perf_counter() - started
    return summarize_load(
        name,
        attempts=attempts,
        concurrency=concurrency,
        successful_latencies_ms=latencies,
        errors=errors,
        timeouts=timeouts,
        duration_s=duration_s,
        scanned_records=scanned_records,
        extra=extra,
    )


def write_synthetic_field_audit(path: Path, *, records: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for index in range(records):
        in_tail = index >= records - TAIL_LIMIT
        targeted = in_tail and index % 50 == 0
        item = {
            "kind": "field_event_action",
            "created_at": index,
            "event_id": f"evt-{index}",
            "audit": {
                "at": index,
                "action": "close" if targeted else "acknowledge",
                "outcome": "accepted" if index % 3 else "denied",
                "operator_id": "security-target" if targeted else f"security-{index % 37}",
                "reason": "benchmark",
            },
        }
        lines.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scan_query(field_path: Path) -> int:
    records = [_normalize_field_action(item) for item in _read_jsonl(field_path, limit=TAIL_LIMIT)]
    since = _time_boundary("")
    until = _time_boundary("")
    filtered = [
        record
        for record in records
        if _matches(
            record,
            source="field",
            operator_id="security-target",
            action="close",
            outcome="accepted",
            tenant_id="",
            delivery_namespace="",
            customer_id="",
            project_id="",
            site_id="",
            managed_object_id="",
            q="",
            since=since,
            until=until,
        )
    ]
    filtered.sort(key=lambda item: item.get("sort_at") or 0.0, reverse=True)
    public = [_public_record(item) for item in filtered[:20]]
    if not public:
        raise RuntimeError("scan benchmark produced no records")
    return len(records)


def indexed_query(service: AuditQueryService) -> int:
    payload = service.query(
        limit=20,
        source="field",
        operator_id="security-target",
        action="close",
        outcome="accepted",
    )
    if not payload.get("records"):
        raise RuntimeError("indexed benchmark produced no records")
    return int(payload.get("query_engine", {}).get("scanned_records") or 0)


def optimization_evidence(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_p95 = float(before.get("p95_ms") or 0.0)
    after_p95 = float(after.get("p95_ms") or 0.0)
    before_scanned = float(before.get("avg_scanned_records") or 0.0)
    after_scanned = float(after.get("avg_scanned_records") or 0.0)
    return {
        "target": "audit_jsonl_exact_filter_index",
        "before": before,
        "after": after,
        "p95_delta_ms": round(before_p95 - after_p95, 3),
        "avg_scanned_records_delta": round(before_scanned - after_scanned, 3),
        "p95_improved": after_p95 < before_p95,
        "scanned_records_reduced": after_scanned < before_scanned,
    }


async def run_benchmarks(
    *,
    quick: bool,
    iterations: int | None,
    concurrency: int | None,
    records: int | None = None,
) -> dict[str, Any]:
    attempts = iterations or (DEFAULT_QUICK_ITERATIONS if quick else DEFAULT_FULL_ITERATIONS)
    worker_count = concurrency or (DEFAULT_QUICK_CONCURRENCY if quick else DEFAULT_FULL_CONCURRENCY)
    record_count = records or (4000 if quick else 20000)
    with tempfile.TemporaryDirectory(prefix="askme-audit-bench-") as tmp:
        root = Path(tmp)
        skill_path = root / "skill-audit.jsonl"
        field_path = root / "field-action-audit.jsonl"
        skill_path.write_text("", encoding="utf-8")
        write_synthetic_field_audit(field_path, records=record_count)
        service = AuditQueryService(
            paths=AuditPaths(skill_audit=skill_path, field_action_audit=field_path)
        )
        service.query(
            limit=20,
            source="field",
            operator_id="security-target",
            action="close",
            outcome="accepted",
        )
        before = await run_load(
            "audit_jsonl_scan",
            lambda index: scan_query(field_path),
            attempts=attempts,
            concurrency=worker_count,
            timeout_s=5.0,
            extra={"records_written": record_count, "tail_limit": TAIL_LIMIT},
        )
        after = await run_load(
            "audit_jsonl_indexed",
            lambda index: indexed_query(service),
            attempts=attempts,
            concurrency=worker_count,
            timeout_s=5.0,
            extra={"records_written": record_count, "tail_limit": TAIL_LIMIT},
        )
    return {
        "generated_at_unix": round(time.time(), 3),
        "quick": quick,
        "iterations": attempts,
        "concurrency": worker_count,
        "benchmarks": {
            "audit_jsonl_scan": before,
            "audit_jsonl_indexed": after,
        },
        "optimization_evidence": optimization_evidence(before, after),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use short local smoke benchmark sizing.")
    parser.add_argument("--iterations", type=int, default=None, help="Attempts per benchmark.")
    parser.add_argument("--concurrency", type=int, default=None, help="Concurrent workers per benchmark.")
    parser.add_argument("--records", type=int, default=None, help="Synthetic JSONL records to write.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = asyncio.run(
        run_benchmarks(
            quick=bool(args.quick),
            iterations=args.iterations,
            concurrency=args.concurrency,
            records=args.records,
        )
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
