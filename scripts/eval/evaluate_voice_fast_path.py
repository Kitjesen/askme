"""Evaluator for the deterministic low-latency voice path."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from askme.robot_interaction import IntentRouter
from askme.robot_interaction.routing_policy import DEFAULT_QUICK_REPLIES
from askme.voice.interaction import match_fast_voice_intent
from askme.voice.output.tts import TTSEngine

TESTS = (
    "tests/test_fast_voice_path.py",
    "tests/test_voice_loop.py",
    "tests/test_intent_router.py",
    "tests/test_tts.py",
    "tests/test_asr_manager.py",
)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "p50": round(statistics.median(values), 4) if values else 0.0,
        "p95": round(_percentile(values, 0.95), 4),
        "max": round(max(values), 4) if values else 0.0,
    }


def _run_tests() -> dict[str, object]:
    command = [sys.executable, "-m", "pytest", *TESTS, "-q"]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-2000:],
    }


def _benchmark(iterations: int) -> dict[str, object]:
    router = IntentRouter()
    route_inputs = ["\u4f60\u597d", "\u4f60\u662f\u8c01", "\u5f53\u524d\u4f4d\u7f6e"]
    route_ms: list[float] = []
    for index in range(iterations):
        started = time.perf_counter_ns()
        intent = router.route(route_inputs[index % len(route_inputs)])
        route_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        if not intent.fast_path:
            raise AssertionError(f"expected fast path for {intent.raw_text!r}")

    with tempfile.TemporaryDirectory(prefix="askme-fast-voice-") as tmp:
        engine = TTSEngine(
            {
                "backend": "edge",
                "sample_rate": 16000,
                "output_tail_silence_seconds": 0.0,
                "phrase_cache_dir": tmp,
            }
        )
        greeting = match_fast_voice_intent(
            "\u4f60\u597d",
            quick_replies=DEFAULT_QUICK_REPLIES,
        )
        assert greeting is not None and greeting.reply_text and greeting.cache_key
        storage_key = engine._phrase_cache_storage_key(
            greeting.reply_text,
            greeting.cache_key,
        )
        engine._phrase_cache.put(
            storage_key,
            np.ones(3200, dtype=np.float32) * 0.1,
            16000,
        )
        cache_ms: list[float] = []
        try:
            for _ in range(iterations):
                started = time.perf_counter_ns()
                hit = engine.queue_cached_phrase(
                    greeting.reply_text,
                    cache_key=greeting.cache_key,
                )
                cache_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
                if not hit:
                    raise AssertionError("primed phrase cache missed")
                with engine._buffer_lock:
                    engine.tts_buffer.clear()
        finally:
            engine.shutdown()

    action_phrases = (
        "\u5e26\u6211\u53bb\u5927\u5802",
        "\u524d\u8fdb\u4e24\u7c73",
        "\u4f60\u4f1a\u8bf4\u8bf7\u8ba9\u4e00\u4e0b\u5417",
        "\u4ecb\u7ecd\u4e00\u4e0b\u8fd9\u4e2a\u8bbe\u5907",
        "\u4e0d\u8981\u8bf4\u8bf7\u8ba9\u4e00\u4e0b",
    )
    false_fast_routes = sum(1 for text in action_phrases if router.route(text).fast_path)

    board = yaml.safe_load(Path("config.board.yaml").read_text(encoding="utf-8"))
    voice = board.get("voice", {})
    fast = voice.get("fast_path", {})
    tts = voice.get("tts", {})
    candidate_ms = float(fast.get("candidate_silence_ms", 300.0))
    stable_ms = float(fast.get("stable_partial_ms", 160.0))
    usb_settle_ms = 180.0
    leadin_ms = float(tts.get("usb_direct_speech_leadin_seconds", 0.25)) * 1000.0
    scheduler_budget_ms = 20.0
    route_stats = _summary(route_ms)
    cache_stats = _summary(cache_ms)
    projected_p50 = (
        max(candidate_ms, stable_ms)
        + float(route_stats["p50"])
        + float(cache_stats["p50"])
        + usb_settle_ms
        + leadin_ms
        + scheduler_budget_ms
    )
    projected_p95 = (
        max(candidate_ms, stable_ms)
        + float(route_stats["p95"])
        + float(cache_stats["p95"])
        + usb_settle_ms
        + leadin_ms
        + scheduler_budget_ms
    )
    return {
        "route_ms": route_stats,
        "cached_pcm_queue_ms": cache_stats,
        "false_fast_action_routes": false_fast_routes,
        "projected_speech_end_to_first_pcm_ms": {
            "p50": round(projected_p50, 3),
            "p95": round(projected_p95, 3),
            "measured_on_device": False,
            "provenance": {
                "candidate_silence_ms": candidate_ms,
                "stable_partial_ms": stable_ms,
                "usb_chunk_settle_ms": usb_settle_ms,
                "speech_leadin_ms": leadin_ms,
                "scheduler_budget_ms": scheduler_budget_ms,
                "config": "config.board.yaml",
                "note": "Projection; physical first sound requires Sunrise room-loop capture.",
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tests = _run_tests()
    benchmark: dict[str, object] = {}
    errors: list[str] = []
    if tests["passed"]:
        try:
            benchmark = _benchmark(300 if args.quick else 2000)
        except Exception as exc:
            errors.append(str(exc))
    else:
        errors.append("regression tests failed")

    projected = benchmark.get("projected_speech_end_to_first_pcm_ms", {})
    p95 = float(projected.get("p95", float("inf"))) if isinstance(projected, dict) else float("inf")
    false_routes = int(benchmark.get("false_fast_action_routes", -1))
    route = benchmark.get("route_ms", {})
    cache = benchmark.get("cached_pcm_queue_ms", {})
    passed = bool(
        tests["passed"]
        and not errors
        and false_routes == 0
        and isinstance(route, dict)
        and float(route.get("p95", float("inf"))) <= 5.0
        and isinstance(cache, dict)
        and float(cache.get("p95", float("inf"))) <= 15.0
        and p95 <= 900.0
    )
    payload = {
        "status": "passed" if passed else "failed",
        "tests": tests,
        "benchmark": benchmark,
        "errors": errors,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
