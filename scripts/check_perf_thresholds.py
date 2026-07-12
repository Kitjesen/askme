"""Fail CI when benchmark metrics exceed configured thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    # Cloud LLM calls include network RTT + inference; 10s is a realistic gate
    "api_chat": {"p95_ms": 10000.0, "error_rate": 0.0, "timeout_rate": 0.0},
    # Voice turn = ASR + LLM + TTS pipeline; 15s gate for full end-to-end
    "runtime_voice_turn": {"p95_ms": 15000.0, "error_rate": 0.0, "timeout_rate": 0.0},
    # Tool execution includes I/O-bound operations (file read, subprocess); 200ms gate
    "tool_execution": {"p95_ms": 200.0, "error_rate": 0.0, "timeout_rate": 0.0},
    "memory_retrieve": {"p95_ms": 5.0, "error_rate": 0.0, "timeout_rate": 0.0},
}


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_override(raw: str) -> tuple[str, str, float]:
    try:
        lhs, value = raw.split("=", 1)
        benchmark, metric = lhs.split(".", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "threshold must look like benchmark.metric=value"
        ) from exc
    try:
        numeric = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"threshold value must be numeric: {raw}") from exc
    return benchmark, metric, numeric


def _thresholds_with_overrides(overrides: list[str]) -> dict[str, dict[str, float]]:
    thresholds = {name: dict(metrics) for name, metrics in DEFAULT_THRESHOLDS.items()}
    for raw in overrides:
        benchmark, metric, value = _parse_override(raw)
        thresholds.setdefault(benchmark, {})[metric] = value
    return thresholds


def check_thresholds(
    report: dict[str, Any],
    thresholds: dict[str, dict[str, float]],
) -> list[str]:
    failures: list[str] = []
    benchmarks = report.get("benchmarks") or {}
    for benchmark, metric_thresholds in thresholds.items():
        payload = benchmarks.get(benchmark)
        if not isinstance(payload, dict):
            failures.append(f"{benchmark}: missing benchmark payload")
            continue
        for metric, max_value in metric_thresholds.items():
            raw_value = payload.get(metric)
            if raw_value is None:
                failures.append(f"{benchmark}.{metric}: missing metric")
                continue
            value = float(raw_value)
            if value > max_value:
                failures.append(
                    f"{benchmark}.{metric}: {value:g} exceeds threshold {max_value:g}"
                )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/perf/core_paths_latest.json"),
        help="Benchmark JSON report path.",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="Override threshold, e.g. tool_execution.p95_ms=75.",
    )
    args = parser.parse_args(argv)

    report = _load_report(args.report)
    thresholds = _thresholds_with_overrides(args.threshold)
    failures = check_thresholds(report, thresholds)
    if failures:
        print("Performance threshold check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("Performance threshold check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
