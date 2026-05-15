from __future__ import annotations

import json

import pytest

from scripts import benchmark_audit_query, benchmark_core_paths, check_perf_thresholds


def _assert_load_summary(payload: dict) -> None:
    for key in (
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "error_rate",
        "timeout_rate",
        "throughput_per_s",
        "attempts",
        "successes",
    ):
        assert key in payload
    assert payload["attempts"] > 0
    assert payload["successes"] > 0
    assert payload["error_rate"] == 0
    assert payload["timeout_rate"] == 0


@pytest.mark.asyncio
async def test_run_benchmarks_emits_core_path_metrics() -> None:
    payload = await benchmark_core_paths.run_benchmarks(
        quick=True,
        iterations=20,
        concurrency=2,
    )

    assert set(payload["benchmarks"]) == {
        "api_chat",
        "runtime_voice_turn",
        "tool_execution",
        "memory_retrieve",
    }
    for summary in payload["benchmarks"].values():
        _assert_load_summary(summary)

    evidence = payload["optimization_evidence"]
    assert evidence["target"] == "memory_retrieve_exact_query_cache_and_inflight_coalescing"
    assert evidence["p95_improved"] is True
    assert evidence["after"]["extra"]["backend_calls"] < evidence["before"]["extra"]["backend_calls"]


def test_benchmark_cli_writes_json_report(tmp_path) -> None:
    output_path = tmp_path / "core_paths.json"

    exit_code = benchmark_core_paths.main([
        "--quick",
        "--iterations",
        "12",
        "--concurrency",
        "2",
        "--output",
        str(output_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["quick"] is True
    _assert_load_summary(payload["benchmarks"]["api_chat"])
    _assert_load_summary(payload["benchmarks"]["memory_retrieve"])


def test_perf_threshold_checker_fails_on_regression() -> None:
    report = {
        "benchmarks": {
            "tool_execution": {
                "p95_ms": 99.0,
                "error_rate": 0.0,
                "timeout_rate": 0.0,
            },
        },
    }
    thresholds = {
        "tool_execution": {
            "p95_ms": 50.0,
            "error_rate": 0.0,
            "timeout_rate": 0.0,
        },
    }

    failures = check_perf_thresholds.check_thresholds(report, thresholds)

    assert failures == ["tool_execution.p95_ms: 99 exceeds threshold 50"]


@pytest.mark.asyncio
async def test_audit_query_benchmark_proves_index_reduces_scans() -> None:
    payload = await benchmark_audit_query.run_benchmarks(
        quick=True,
        iterations=12,
        concurrency=2,
        records=1200,
    )

    before = payload["benchmarks"]["audit_jsonl_scan"]
    after = payload["benchmarks"]["audit_jsonl_indexed"]
    _assert_load_summary(before)
    _assert_load_summary(after)
    assert payload["optimization_evidence"]["target"] == "audit_jsonl_exact_filter_index"
    assert payload["optimization_evidence"]["scanned_records_reduced"] is True
    assert after["avg_scanned_records"] < before["avg_scanned_records"]
