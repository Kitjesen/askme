"""Benchmark semantic TTFT for the LiteLLM voice model alias.

This script is intentionally LiteLLM-only. It fails closed unless the scoped
LiteLLM virtual key is provided through ``LITELLM_VIRTUAL_KEY`` and never records
prompt text, response text, response chunks, Authorization headers, or secrets in
the JSON evidence file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openai import AsyncOpenAI

DEFAULT_MODEL_ALIAS = "voice-fast"
DEFAULT_DEV_SAMPLES = 10
RELEASE_MIN_SAMPLES = 50
DEFAULT_MAX_TOKENS = 80

PROMPTS: tuple[dict[str, Any], ...] = (
    {
        "id": "time_short_zh",
        "messages": [{"role": "user", "content": "用一句自然中文回答：现在需要先做什么？"}],
    },
    {
        "id": "robot_ack_zh",
        "messages": [{"role": "user", "content": "用户说去仓库A，请用一句话确认任务。"}],
    },
    {
        "id": "brief_help_zh",
        "messages": [{"role": "user", "content": "用不超过十五个字说明你可以帮我做什么。"}],
    },
)


class BenchConfigError(RuntimeError):
    """Raised when benchmark inputs would make the evidence invalid."""


@dataclass(frozen=True)
class BenchConfig:
    base_url: str
    api_key: str
    model: str = DEFAULT_MODEL_ALIAS
    samples: int = DEFAULT_DEV_SAMPLES
    mode: str = "dev"
    seed: int | None = None
    timeout_s: float = 30.0
    max_tokens: int = DEFAULT_MAX_TOKENS


@dataclass(frozen=True)
class SampleCase:
    sample_id: str
    stratum: str
    prompt_id: str
    prompt_index: int


@dataclass(frozen=True)
class SampleResult:
    sample_id: str
    stratum: str
    prompt_id: str
    status: str
    model_alias: str
    semantic_ttft_ms: float | None
    total_ms: float
    semantic_chunks: int
    error_type: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "stratum": self.stratum,
            "prompt_id": self.prompt_id,
            "status": self.status,
            "model_alias": self.model_alias,
            "semantic_ttft_ms": self.semantic_ttft_ms,
            "total_ms": round(self.total_ms, 3),
            "semantic_chunks": self.semantic_chunks,
            "error_type": self.error_type,
        }


ClientFactory = Callable[[str, str, float], Any]


def load_config_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    model: str = DEFAULT_MODEL_ALIAS,
    samples: int = DEFAULT_DEV_SAMPLES,
    mode: str = "dev",
    seed: int | None = None,
    timeout_s: float = 30.0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> BenchConfig:
    env = os.environ if environ is None else environ
    base_url = str(env.get("LITELLM_BASE_URL") or "").strip().rstrip("/")
    api_key = str(env.get("LITELLM_VIRTUAL_KEY") or "").strip()
    legacy_key = str(env.get("LLM_API_KEY") or "").strip()

    if not base_url:
        raise BenchConfigError("LITELLM_BASE_URL is required")
    if not api_key:
        raise BenchConfigError("LITELLM_VIRTUAL_KEY is required")
    if legacy_key and legacy_key == api_key:
        raise BenchConfigError("LLM_API_KEY must not be used as the LiteLLM benchmark key")
    if mode not in {"dev", "release"}:
        raise BenchConfigError("mode must be dev or release")
    if samples < 1:
        raise BenchConfigError("samples must be positive")
    if mode == "release" and samples < RELEASE_MIN_SAMPLES:
        raise BenchConfigError(f"release mode requires at least {RELEASE_MIN_SAMPLES} samples")
    if not model.strip():
        raise BenchConfigError("model alias is required")
    if max_tokens < 1:
        raise BenchConfigError("max_tokens must be positive")

    return BenchConfig(
        base_url=base_url,
        api_key=api_key,
        model=model.strip(),
        samples=samples,
        mode=mode,
        seed=seed,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
    )


def plan_samples(samples: int, *, seed: int | None = None) -> list[SampleCase]:
    if samples < 1:
        raise BenchConfigError("samples must be positive")

    cold_count = samples // 2
    warm_count = samples - cold_count
    strata = ["cold"] * cold_count + ["warm"] * warm_count
    rng = random.Random(seed)
    rng.shuffle(strata)

    cases: list[SampleCase] = []
    for index, stratum in enumerate(strata):
        prompt_index = index % len(PROMPTS)
        cases.append(
            SampleCase(
                sample_id=f"sample-{index + 1:03d}",
                stratum=stratum,
                prompt_id=str(PROMPTS[prompt_index]["id"]),
                prompt_index=prompt_index,
            )
        )
    return cases


def _chunk_content(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""
    first = choices[0]
    delta = getattr(first, "delta", None)
    content = getattr(delta, "content", None) if delta is not None else None
    if content is None:
        content = getattr(first, "text", None)
    return content if isinstance(content, str) else ""


def _is_semantic_payload(text: str) -> bool:
    return bool(text and text.strip())


async def _close_client(client: Any) -> None:
    close = getattr(client, "close", None) or getattr(client, "aclose", None)
    if close is None:
        return
    result = close()
    if hasattr(result, "__await__"):
        await result


def _create_openai_client(base_url: str, api_key: str, timeout_s: float) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)


async def measure_semantic_ttft(
    client: Any,
    *,
    model: str,
    messages: Sequence[Mapping[str, str]],
    sample: SampleCase,
    max_tokens: int,
) -> SampleResult:
    """Measure first non-empty semantic payload, ignoring empty keepalive chunks."""
    started = time.perf_counter()
    semantic_ttft_ms: float | None = None
    semantic_chunks = 0

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=list(messages),
            stream=True,
            temperature=0.2,
            max_tokens=max_tokens,
        )
        async for chunk in stream:
            content = _chunk_content(chunk)
            if not _is_semantic_payload(content):
                continue
            semantic_chunks += 1
            if semantic_ttft_ms is None:
                semantic_ttft_ms = (time.perf_counter() - started) * 1000

        total_ms = (time.perf_counter() - started) * 1000
        status = "ok" if semantic_ttft_ms is not None else "no_semantic_payload"
        return SampleResult(
            sample_id=sample.sample_id,
            stratum=sample.stratum,
            prompt_id=sample.prompt_id,
            status=status,
            model_alias=model,
            semantic_ttft_ms=round(semantic_ttft_ms, 3) if semantic_ttft_ms is not None else None,
            total_ms=total_ms,
            semantic_chunks=semantic_chunks,
        )
    except Exception as exc:  # noqa: BLE001 - evidence stores type only, never message.
        total_ms = (time.perf_counter() - started) * 1000
        return SampleResult(
            sample_id=sample.sample_id,
            stratum=sample.stratum,
            prompt_id=sample.prompt_id,
            status="error",
            model_alias=model,
            semantic_ttft_ms=None,
            total_ms=total_ms,
            semantic_chunks=0,
            error_type=type(exc).__name__,
        )


def _warm_primer_failure_result(
    case: SampleCase,
    *,
    model: str,
    primer_result: SampleResult,
) -> SampleResult:
    error_type = primer_result.error_type or primer_result.status
    return SampleResult(
        sample_id=case.sample_id,
        stratum=case.stratum,
        prompt_id=case.prompt_id,
        status="warm_primer_failed",
        model_alias=model,
        semantic_ttft_ms=None,
        total_ms=0.0,
        semantic_chunks=0,
        error_type=error_type,
    )


async def _prime_warm_client(client: Any, config: BenchConfig) -> SampleResult:
    primer = SampleCase(
        sample_id="warm-primer",
        stratum="warm",
        prompt_id="warm_primer",
        prompt_index=0,
    )
    return await measure_semantic_ttft(
        client,
        model=config.model,
        messages=PROMPTS[0]["messages"],
        sample=primer,
        max_tokens=config.max_tokens,
    )


async def run_benchmark(
    config: BenchConfig,
    *,
    client_factory: ClientFactory = _create_openai_client,
) -> dict[str, Any]:
    cases = plan_samples(config.samples, seed=config.seed)
    warm_client: Any | None = None
    warm_primer_result: SampleResult | None = None
    results: list[SampleResult] = []

    try:
        for case in cases:
            prompt = PROMPTS[case.prompt_index]
            if case.stratum == "warm":
                if warm_client is None:
                    warm_client = client_factory(
                        config.base_url,
                        config.api_key,
                        config.timeout_s,
                    )
                    warm_primer_result = await _prime_warm_client(warm_client, config)

                if warm_primer_result is None or warm_primer_result.status != "ok":
                    failed_primer = warm_primer_result or SampleResult(
                        "warm-primer",
                        "warm",
                        "warm_primer",
                        "error",
                        config.model,
                        None,
                        0.0,
                        0,
                        "missing_warm_primer",
                    )
                    results.append(
                        _warm_primer_failure_result(
                            case,
                            model=config.model,
                            primer_result=failed_primer,
                        )
                    )
                    continue

                client = warm_client
            else:
                client = client_factory(config.base_url, config.api_key, config.timeout_s)

            result = await measure_semantic_ttft(
                client,
                model=config.model,
                messages=prompt["messages"],
                sample=case,
                max_tokens=config.max_tokens,
            )
            results.append(result)

            if case.stratum == "cold":
                await _close_client(client)
    finally:
        if warm_client is not None:
            await _close_client(warm_client)

    return build_evidence(config, results)


def percentile(values: Sequence[float], percentile_value: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (percentile_value / 100) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def _percentile_summary(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def build_evidence(config: BenchConfig, results: Sequence[SampleResult]) -> dict[str, Any]:
    successful = [r.semantic_ttft_ms for r in results if r.semantic_ttft_ms is not None]
    cold_successful = [
        r.semantic_ttft_ms
        for r in results
        if r.stratum == "cold" and r.semantic_ttft_ms is not None
    ]
    warm_successful = [
        r.semantic_ttft_ms
        for r in results
        if r.stratum == "warm" and r.semantic_ttft_ms is not None
    ]
    parsed_url = urlparse(config.base_url)
    host = parsed_url.netloc or parsed_url.path

    return {
        "schema_version": "askme.ttft_bench.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": "litellm_semantic_ttft",
        "product_gate_usable": False,
        "mode": config.mode,
        "model_alias": config.model,
        "litellm_endpoint_host": host,
        "sample_count_requested": config.samples,
        "sample_count_ok": len(successful),
        "strata": {
            "cold": sum(1 for result in results if result.stratum == "cold"),
            "warm": sum(1 for result in results if result.stratum == "warm"),
        },
        "summary": {
            "semantic_ttft_ms": {
                "aggregate": _percentile_summary(successful),
                "cold": _percentile_summary(cold_successful),
                "warm": _percentile_summary(warm_successful),
            }
        },
        "samples": [result.to_json() for result in results],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ALIAS, help="LiteLLM capability alias")
    parser.add_argument("--samples", type=int, default=DEFAULT_DEV_SAMPLES)
    parser.add_argument("--mode", choices=("dev", "release"), default="dev")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--output", type=Path, default=None, help="Write JSON evidence to this path"
    )
    return parser


async def async_main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = load_config_from_env(
            model=args.model,
            samples=args.samples,
            mode=args.mode,
            seed=args.seed,
            timeout_s=args.timeout,
            max_tokens=args.max_tokens,
        )
        evidence = await run_benchmark(config)
    except BenchConfigError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    payload = json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"[OK] wrote TTFT evidence: {args.output}")
    else:
        print(payload, end="")
    return 0


def main() -> int:
    return asyncio.run(async_main())


if __name__ == "__main__":
    raise SystemExit(main())
