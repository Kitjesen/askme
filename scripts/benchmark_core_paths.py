"""Benchmark Askme core latency paths.

The benchmark is intentionally local and deterministic: it exercises the FastAPI
ASGI app, tool registry, and MemoryBridge retrieval mechanics without requiring
external LLM, robot, or vector services.
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from askme.health_server import create_health_app
from askme.memory.bridge import MemoryBridge
from askme.tools.tool_registry import BaseTool, ToolRegistry


DEFAULT_QUICK_ITERATIONS = 40
DEFAULT_FULL_ITERATIONS = 200
DEFAULT_QUICK_CONCURRENCY = 8
DEFAULT_FULL_CONCURRENCY = 16


class _RuntimeVoiceHandler:
    async def voice_turn_payload(self, text: str, **kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(0.001)
        return {
            "handled": True,
            "recognized_text": text,
            "action": "benchmark",
            "kwargs": kwargs,
        }


class _BenchmarkTool(BaseTool):
    name = "benchmark_sleep"
    description = "Local benchmark tool."
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "normal"

    def execute(self, **kwargs: Any) -> str:
        time.sleep(0.001)
        return "ok"


class _BenchmarkMemoryBridge(MemoryBridge):
    def __init__(self, *, cache_ttl_s: float, backend_delay_s: float = 0.003) -> None:
        cfg = {
            "memory": {
                "enabled": True,
                "backend": "vector",
                "embed_model": "benchmark",
                "retrieve_timeout": 1.0,
                "retrieve_cache_ttl_s": cache_ttl_s,
                "retrieve_cache_max_entries": 256,
            },
            "app": {"data_dir": "data"},
            "brain": {"api_key": "", "base_url": "", "model": "benchmark"},
        }
        super().__init__(config=cfg)
        self.backend_delay_s = backend_delay_s
        self.backend_calls = 0

    async def _retrieve_with_fallbacks(self, text: str) -> str:
        self.backend_calls += 1
        self._last_backend = "benchmark"
        self._last_fallback_reason = ""
        await asyncio.sleep(self.backend_delay_s)
        items = [{
            "text": f"benchmark memory for {text}",
            "backend": "benchmark",
            "source": "benchmark",
            "category": "benchmark",
            "score": 1.0,
            "metadata": {"approval_status": "published"},
        }]
        self._set_evidence(items, backend="benchmark")
        return self._format_evidence(items)


def _benchmark_config(concurrency: int) -> dict[str, Any]:
    return {
        "brain": {"api_key": "benchmark", "base_url": "http://benchmark"},
        "conversation": {
            "chat_timeout_s": 2.0,
            "chat_max_concurrency": max(1, concurrency),
            "chat_slow_threshold_ms": 1000.0,
            "chat_diagnostics_history_limit": 20,
            "runtime_voice_turn_timeout_s": 2.0,
        },
        "memory": {"enabled": False},
        "tools": {},
    }


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
    value = ordered[lower] * (1 - weight) + ordered[upper] * weight
    return round(value, 3)


def summarize_load(
    name: str,
    *,
    attempts: int,
    concurrency: int,
    successful_latencies_ms: list[float],
    errors: int,
    timeouts: int,
    duration_s: float,
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
    errors = 0
    timeouts = 0

    async def one(index: int) -> None:
        nonlocal errors, timeouts
        async with semaphore:
            started = time.perf_counter()
            try:
                await asyncio.wait_for(operation(index), timeout=timeout_s)
            except TimeoutError:
                errors += 1
                timeouts += 1
            except Exception:
                errors += 1
            else:
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
        extra=extra,
    )


async def benchmark_chat(*, attempts: int, concurrency: int) -> dict[str, Any]:
    async def chat_handler(text: str, *, speak: bool = False) -> dict[str, Any]:
        await asyncio.sleep(0.001)
        return {"reply": f"ack:{text}", "spoken": speak}

    with patch("askme.health_server.get_config", return_value=_benchmark_config(concurrency)):
        app = create_health_app(
            lambda: {"status": "ok"},
            chat_handler=chat_handler,
            runtime_handler=_RuntimeVoiceHandler(),
            field_operations_handler=object(),
            space_handler=object(),
        )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://benchmark") as client:
        async def operation(index: int) -> None:
            response = await client.post(
                "/api/chat",
                json={"text": f"benchmark chat {index}"},
                headers={"X-Request-Id": f"bench-chat-{index}"},
            )
            if response.status_code == 504:
                raise TimeoutError(response.text)
            if response.status_code >= 400:
                raise RuntimeError(f"chat status {response.status_code}")

        return await run_load(
            "api_chat",
            operation,
            attempts=attempts,
            concurrency=concurrency,
            timeout_s=2.0,
        )


async def benchmark_runtime_voice_turn(*, attempts: int, concurrency: int) -> dict[str, Any]:
    with patch("askme.health_server.get_config", return_value=_benchmark_config(concurrency)):
        app = create_health_app(
            lambda: {"status": "ok"},
            chat_handler=lambda text, speak=False: {"reply": text, "spoken": speak},
            runtime_handler=_RuntimeVoiceHandler(),
            field_operations_handler=object(),
            space_handler=object(),
        )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://benchmark") as client:
        async def operation(index: int) -> None:
            response = await client.post(
                "/api/runtime/voice-turn",
                json={
                    "text": f"benchmark voice {index}",
                    "transcript_id": f"voice-{index}",
                    "is_final": True,
                },
            )
            if response.status_code == 504:
                raise TimeoutError(response.text)
            if response.status_code >= 400:
                raise RuntimeError(f"voice-turn status {response.status_code}")

        return await run_load(
            "runtime_voice_turn",
            operation,
            attempts=attempts,
            concurrency=concurrency,
            timeout_s=2.0,
        )


async def benchmark_tool_execution(*, attempts: int, concurrency: int) -> dict[str, Any]:
    registry = ToolRegistry(config={
        "default_timeout": 1.0,
        "timeout_cooldown": 0.0,
        "executor_max_workers": max(1, concurrency),
        "require_confirmation_levels": ["dangerous", "critical"],
    })
    registry.register(_BenchmarkTool())

    async def operation(index: int) -> None:
        result = await asyncio.to_thread(registry.execute, "benchmark_sleep", "{}")
        if result != "ok":
            raise RuntimeError(result)

    try:
        result = await run_load(
            "tool_execution",
            operation,
            attempts=attempts,
            concurrency=concurrency,
            timeout_s=2.0,
        )
        result["extra"] = registry.diagnostics()
        return result
    finally:
        registry.shutdown(wait=True, cancel_futures=True)


async def benchmark_memory_retrieve(
    *,
    attempts: int,
    concurrency: int,
    cache_ttl_s: float,
    name: str = "memory_retrieve",
) -> dict[str, Any]:
    bridge = _BenchmarkMemoryBridge(cache_ttl_s=cache_ttl_s)
    warmed_cache = False
    if cache_ttl_s > 0:
        await bridge.retrieve("same field location")
        warmed_cache = True

    async def operation(index: int) -> None:
        result = await bridge.retrieve("same field location")
        if not result:
            raise RuntimeError("empty retrieve")

    result = await run_load(
        name,
        operation,
        attempts=attempts,
        concurrency=concurrency,
        timeout_s=2.0,
        extra={
            "cache_ttl_s": cache_ttl_s,
            "warmed_cache": warmed_cache,
        },
    )
    result["extra"]["backend_calls"] = bridge.backend_calls
    result["extra"]["health"] = bridge.health()
    return result


def optimization_evidence(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_p95 = float(before.get("p95_ms") or 0.0)
    after_p95 = float(after.get("p95_ms") or 0.0)
    before_timeout_rate = float(before.get("timeout_rate") or 0.0)
    after_timeout_rate = float(after.get("timeout_rate") or 0.0)
    return {
        "target": "memory_retrieve_exact_query_cache_and_inflight_coalescing",
        "before": before,
        "after": after,
        "p95_delta_ms": round(before_p95 - after_p95, 3),
        "timeout_rate_delta": round(before_timeout_rate - after_timeout_rate, 6),
        "p95_improved": after_p95 < before_p95,
        "timeouts_reduced": after_timeout_rate < before_timeout_rate,
    }


async def run_benchmarks(*, quick: bool, iterations: int | None, concurrency: int | None) -> dict[str, Any]:
    attempts = iterations or (DEFAULT_QUICK_ITERATIONS if quick else DEFAULT_FULL_ITERATIONS)
    worker_count = concurrency or (DEFAULT_QUICK_CONCURRENCY if quick else DEFAULT_FULL_CONCURRENCY)
    memory_before = await benchmark_memory_retrieve(
        attempts=attempts,
        concurrency=worker_count,
        cache_ttl_s=0.0,
        name="memory_retrieve_no_cache",
    )
    memory_after = await benchmark_memory_retrieve(
        attempts=attempts,
        concurrency=worker_count,
        cache_ttl_s=30.0,
        name="memory_retrieve",
    )
    return {
        "generated_at_unix": round(time.time(), 3),
        "quick": quick,
        "iterations": attempts,
        "concurrency": worker_count,
        "benchmarks": {
            "api_chat": await benchmark_chat(attempts=attempts, concurrency=worker_count),
            "runtime_voice_turn": await benchmark_runtime_voice_turn(
                attempts=attempts,
                concurrency=worker_count,
            ),
            "tool_execution": await benchmark_tool_execution(
                attempts=attempts,
                concurrency=worker_count,
            ),
            "memory_retrieve": memory_after,
        },
        "optimization_evidence": optimization_evidence(memory_before, memory_after),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use short local smoke benchmark sizing.")
    parser.add_argument("--iterations", type=int, default=None, help="Attempts per benchmark.")
    parser.add_argument("--concurrency", type=int, default=None, help="Concurrent workers per benchmark.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING)
    args = build_parser().parse_args(argv)
    payload = asyncio.run(
        run_benchmarks(
            quick=bool(args.quick),
            iterations=args.iterations,
            concurrency=args.concurrency,
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
