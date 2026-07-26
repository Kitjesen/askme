"""Evaluator for the deterministic low-latency voice path.

Process microbenchmarks and transport budgets are useful diagnostics, but only
physical-acoustic samples can satisfy the first-sound product gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from askme.robot_interaction import IntentRouter
from askme.robot_interaction.routing.fast_voice_intents import match_fast_voice_intent
from askme.robot_interaction.routing_policy import DEFAULT_QUICK_REPLIES
from askme.voice.output.tts import TTSEngine

TESTS = (
    "tests/test_fast_voice_path.py",
    "tests/test_voice_loop.py",
    "tests/test_intent_router.py",
    "tests/test_tts.py",
    "tests/test_asr_manager.py",
)
MIN_PHYSICAL_TRIALS = 20
FAST_PATH_PHYSICAL_P95_MS = 900.0
APLAY_COLD_PREROLL_MS = 1500.0
SCHEDULER_BUDGET_MS = 20.0


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


def _transport_components(
    tts_config: Mapping[str, object],
) -> tuple[str, dict[str, float] | None, str | None]:
    transport = str(tts_config.get("output_transport", "auto") or "auto").strip().lower()
    if transport == "aplay":
        return transport, {"aplay_cold_preroll_ms": APLAY_COLD_PREROLL_MS}, None
    if transport == "usb_direct":
        components = {
            "usb_direct_speech_leadin_ms": max(
                0.0,
                float(tts_config.get("usb_direct_speech_leadin_seconds", 0.0)) * 1000.0,
            )
        }
        if bool(tts_config.get("usb_direct_persistent_stream", False)):
            components["usb_direct_stream_start_grace_ms"] = max(
                0.0,
                float(tts_config.get("usb_direct_stream_start_grace_seconds", 0.0)) * 1000.0,
            )
        return transport, components, None
    if transport == "sounddevice":
        return transport, {}, None
    if transport == "auto":
        return transport, None, "runtime_output_transport_unresolved"
    return transport, None, "unsupported_output_transport"


def _build_projection(
    *,
    tts_config: Mapping[str, object],
    candidate_silence_ms: float,
    stable_partial_ms: float,
    route_stats: Mapping[str, object],
    cache_stats: Mapping[str, object],
    config_path: str = "config.board.yaml",
) -> dict[str, object]:
    """Build a transport-specific software budget without claiming physical audio."""

    transport, transport_components, unavailable_reason = _transport_components(tts_config)
    common: dict[str, object] = {
        "available": transport_components is not None,
        "output_transport": transport,
        "evidence_type": "software_projection",
        "measurement_scope": "computed_transport_budget",
        "measured_on_device": False,
        "physical_first_sound": False,
        "p50": None,
        "p95": None,
    }
    if transport_components is None:
        common.update(
            {
                "reason": unavailable_reason,
                "provenance": {
                    "config": config_path,
                    "output_transport": transport,
                    "note": "No projection is emitted until runtime transport is explicit.",
                },
            }
        )
        return common

    endpoint_ms = max(float(candidate_silence_ms), float(stable_partial_ms))
    transport_ms = sum(transport_components.values())
    route_p50 = float(route_stats.get("p50", 0.0))
    route_p95 = float(route_stats.get("p95", 0.0))
    cache_p50 = float(cache_stats.get("p50", 0.0))
    cache_p95 = float(cache_stats.get("p95", 0.0))
    common.update(
        {
            "p50": round(
                endpoint_ms + route_p50 + cache_p50 + transport_ms + SCHEDULER_BUDGET_MS,
                3,
            ),
            "p95": round(
                endpoint_ms + route_p95 + cache_p95 + transport_ms + SCHEDULER_BUDGET_MS,
                3,
            ),
            "projection_category": f"{transport}_software_budget",
            "provenance": {
                "candidate_silence_ms": float(candidate_silence_ms),
                "stable_partial_ms": float(stable_partial_ms),
                "transport_components_ms": transport_components,
                "scheduler_budget_ms": SCHEDULER_BUDGET_MS,
                "config": config_path,
                "output_transport": transport,
                "warm_state_assumption": "cold" if transport == "aplay" else "configured",
                "note": (
                    "Software projection only; physical semantic first sound requires "
                    "same-transport room-loop evidence. Cached PCM bypasses streamed aplay "
                    "network prebuffer, but a cold aplay start still pays its 1.5s pre-roll."
                ),
            },
        }
    )
    return common


def _benchmark(
    iterations: int, *, config_path: Path = Path("config.board.yaml")
) -> dict[str, object]:
    router = IntentRouter()
    route_inputs = ["你好", "你是谁", "当前位置"]
    route_ms: list[float] = []
    for index in range(iterations):
        started = time.perf_counter_ns()
        intent = router.route(route_inputs[index % len(route_inputs)])
        route_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        if not intent.fast_path:
            raise AssertionError(f"expected fast path for {intent.raw_text!r}")

    cache_hits = 0
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
            "你好",
            quick_replies=DEFAULT_QUICK_REPLIES,
        )
        assert greeting is not None and greeting.reply_text and greeting.cache_key
        storage_key = engine._phrase_cache_storage_key(
            greeting.reply_text,
            greeting.cache_key,
        )
        cache_primed = engine._phrase_cache.put(
            storage_key,
            np.ones(3200, dtype=np.float32) * 0.1,
            16000,
        )
        if not cache_primed:
            engine.shutdown()
            raise AssertionError("failed to prime phrase cache")
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
                cache_hits += 1
                with engine._buffer_lock:
                    engine.tts_buffer.clear()
        finally:
            engine.shutdown()

    action_phrases = (
        "带我去大堂",
        "前进两米",
        "你会说请让一下吗",
        "介绍一下这个设备",
        "不要说请让一下",
    )
    false_fast_routes = sum(1 for text in action_phrases if router.route(text).fast_path)

    board = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(board, Mapping):
        raise ValueError(f"invalid config mapping: {config_path}")
    voice = board.get("voice", {})
    if not isinstance(voice, Mapping):
        raise ValueError(f"invalid voice config mapping: {config_path}")
    fast = voice.get("fast_path", {})
    tts = voice.get("tts", {})
    if not isinstance(fast, Mapping) or not isinstance(tts, Mapping):
        raise ValueError(f"invalid fast_path/tts config mapping: {config_path}")
    candidate_ms = float(fast.get("candidate_silence_ms", 300.0))
    stable_ms = float(fast.get("stable_partial_ms", 160.0))
    route_stats = _summary(route_ms)
    cache_stats = _summary(cache_ms)
    projection = _build_projection(
        tts_config=tts,
        candidate_silence_ms=candidate_ms,
        stable_partial_ms=stable_ms,
        route_stats=route_stats,
        cache_stats=cache_stats,
        config_path=str(config_path),
    )
    return {
        "route_ms": route_stats,
        "cached_pcm_queue_ms": cache_stats,
        "cache_evidence": {
            "ready": cache_hits == iterations and iterations > 0,
            "primed": cache_primed,
            "hits": cache_hits,
            "expected_hits": iterations,
            "scope": "synthetic_process_cache",
        },
        "false_fast_action_routes": false_fast_routes,
        "projected_speech_end_to_first_pcm_ms": projection,
    }


def _physical_trials(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
    raw_trials: object = payload.get("trials", payload.get("response_trials", []))
    if isinstance(raw_trials, Mapping):
        raw_trials = raw_trials.get("assistant_response", [])
    if not isinstance(raw_trials, list):
        return []
    return [trial for trial in raw_trials if isinstance(trial, Mapping)]


def _summarize_physical_evidence(
    payload: Mapping[str, object] | None,
    *,
    expected_transport: str,
) -> dict[str, object]:
    if payload is None:
        return {"ready": False, "reason": "physical_evidence_missing", "count": 0}
    metadata = payload.get("metadata", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    evidence_transport = (
        str(payload.get("output_transport", metadata.get("output_transport", "")) or "")
        .strip()
        .lower()
    )
    if not evidence_transport:
        return {
            "ready": False,
            "reason": "physical_evidence_transport_missing",
            "count": 0,
        }
    if evidence_transport != str(expected_transport or "").strip().lower():
        return {
            "ready": False,
            "reason": "physical_evidence_transport_mismatch",
            "count": 0,
            "output_transport": evidence_transport,
        }

    latencies: list[float] = []
    for trial in _physical_trials(payload):
        route = str(trial.get("route", trial.get("path", "")) or "").strip().lower()
        if trial.get("fast_path") is not True and route != "fast_path":
            continue
        if trial.get("heard") is not True:
            continue
        if trial.get("evidence_kind") != "physical_acoustic":
            continue
        if trial.get("audio_class") != "semantic":
            continue
        latency = trial.get("speech_end_to_first_semantic_audio_ms")
        if isinstance(latency, bool) or not isinstance(latency, (int, float)):
            continue
        value = float(latency)
        if not np.isfinite(value) or value < 0.0:
            continue
        latencies.append(value)

    summary: dict[str, object] = {
        "ready": False,
        "reason": "physical_fast_path_samples_insufficient",
        "count": len(latencies),
        "required_count": MIN_PHYSICAL_TRIALS,
        "output_transport": evidence_transport,
        "evidence_type": "physical_acoustic_measured",
    }
    if latencies:
        summary.update(
            {
                "p50": round(statistics.median(latencies), 3),
                "p95": round(_percentile(latencies, 0.95), 3),
                "max": round(max(latencies), 3),
            }
        )
    if len(latencies) < MIN_PHYSICAL_TRIALS:
        return summary
    if float(summary["p95"]) > FAST_PATH_PHYSICAL_P95_MS:
        summary["reason"] = "physical_fast_path_p95_exceeds_900ms"
        return summary
    summary["ready"] = True
    summary["reason"] = "physical_fast_path_evidence_ready"
    return summary


def _load_physical_evidence(path: Path | None, *, expected_transport: str) -> dict[str, object]:
    if path is None:
        return _summarize_physical_evidence(None, expected_transport=expected_transport)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ready": False,
            "reason": "physical_evidence_invalid",
            "count": 0,
            "error": str(exc),
        }
    if not isinstance(payload, Mapping):
        return {"ready": False, "reason": "physical_evidence_invalid", "count": 0}
    return _summarize_physical_evidence(payload, expected_transport=expected_transport)


def _acceptance_failures(
    *,
    tests_passed: bool,
    benchmark: Mapping[str, object],
    physical_evidence: Mapping[str, object],
) -> list[str]:
    failures: list[str] = []
    if not tests_passed:
        failures.append("regression_tests_failed")

    if int(benchmark.get("false_fast_action_routes", -1)) != 0:
        failures.append("unsafe_fast_action_route_detected")
    route = benchmark.get("route_ms", {})
    route = route if isinstance(route, Mapping) else {}
    if float(route.get("p95", float("inf"))) > 5.0:
        failures.append("fast_route_p95_exceeds_5ms")
    cache = benchmark.get("cached_pcm_queue_ms", {})
    cache = cache if isinstance(cache, Mapping) else {}
    if float(cache.get("p95", float("inf"))) > 15.0:
        failures.append("cached_pcm_queue_p95_exceeds_15ms")
    cache_evidence = benchmark.get("cache_evidence", {})
    cache_evidence = cache_evidence if isinstance(cache_evidence, Mapping) else {}
    if cache_evidence.get("ready") is not True:
        failures.append("cached_phrase_evidence_missing")

    projection = benchmark.get("projected_speech_end_to_first_pcm_ms", {})
    projection = projection if isinstance(projection, Mapping) else {}
    if projection.get("available") is not True:
        failures.append(str(projection.get("reason") or "transport_projection_unavailable"))
    if projection.get("measured_on_device") is not False:
        failures.append("software_projection_mislabeled_as_device_measurement")
    if projection.get("physical_first_sound") is not False:
        failures.append("software_projection_mislabeled_as_physical_sound")

    if physical_evidence.get("ready") is not True:
        failures.append(str(physical_evidence.get("reason") or "physical_evidence_missing"))
    return list(dict.fromkeys(failures))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--config", default="config.board.yaml")
    parser.add_argument("--physical-evidence")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    tests = _run_tests()
    benchmark: dict[str, object] = {}
    execution_errors: list[str] = []
    if tests["passed"]:
        try:
            benchmark = _benchmark(
                300 if args.quick else 2000,
                config_path=Path(args.config),
            )
        except Exception as exc:
            execution_errors.append(str(exc))
    else:
        execution_errors.append("regression tests failed")

    projection = benchmark.get("projected_speech_end_to_first_pcm_ms", {})
    projection = projection if isinstance(projection, Mapping) else {}
    expected_transport = str(projection.get("output_transport", "") or "")
    physical_evidence = _load_physical_evidence(
        Path(args.physical_evidence) if args.physical_evidence else None,
        expected_transport=expected_transport,
    )
    gate_failures = _acceptance_failures(
        tests_passed=bool(tests["passed"]),
        benchmark=benchmark,
        physical_evidence=physical_evidence,
    )
    errors = [*execution_errors, *gate_failures]
    errors = list(dict.fromkeys(errors))
    passed = not errors
    payload = {
        "status": "passed" if passed else "failed",
        "generated_at": datetime.now(UTC).isoformat(),
        "tests": tests,
        "benchmark": benchmark,
        "physical_evidence": physical_evidence,
        "acceptance": {
            "passed": passed,
            "fail_closed": True,
            "required_physical_trials": MIN_PHYSICAL_TRIALS,
            "physical_p95_target_ms": FAST_PATH_PHYSICAL_P95_MS,
            "failures": errors,
        },
        "errors": errors,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
