"""Measure the configured voice model through Askme's real LLM path."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any

from askme.config import get_config
from askme.llm.core.client import LLMClient
from askme.pipeline.core.persona import persona_from_brain_config

QUESTIONS = [
    "洗手间在哪里？",
    "今天园区有什么活动？",
    "你叫什么名字？",
]

_STRONG_ENDINGS = "。！？!?"
_GENERIC_ACKNOWLEDGEMENTS = {"好的", "收到", "明白", "可以", "没问题"}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "navigate_to",
            "description": "引导用户前往园区内的指定点位",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "目标点位名称"},
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_patrol",
            "description": "发起一次园区巡检任务",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return round(ordered[index], 1)


async def run_once(
    client: LLMClient,
    voice_model: str,
    question: str,
    max_tokens: int,
    *,
    system_prompt: str,
    prompt_seed: list[dict[str, str]],
    user_prefix: str,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        *prompt_seed,
        {
            "role": "user",
            "content": f"{user_prefix}\n{question}" if user_prefix else question,
        },
    ]
    started = time.perf_counter()
    first_chunk_s = None
    first_content_s = None
    first_clause_s = None
    first_semantic_clause_s = None
    first_clause = ""
    scan_from = 0
    text_parts: list[str] = []
    tool_names: list[str] = []
    async for chunk in client.chat_stream(
        messages,
        tools=TOOLS,
        tool_choice="auto",
        model=voice_model,
        max_tokens=max_tokens,
    ):
        elapsed = time.perf_counter() - started
        if first_chunk_s is None:
            first_chunk_s = elapsed
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "content", None):
            if first_content_s is None:
                first_content_s = elapsed
            text_parts.append(delta.content)
            combined = "".join(text_parts)
            for index in range(scan_from, len(combined)):
                if combined[index] not in _STRONG_ENDINGS:
                    continue
                clause = combined[scan_from : index + 1].strip()
                scan_from = index + 1
                normalized = clause.rstrip(_STRONG_ENDINGS).strip()
                if first_clause_s is None:
                    first_clause_s = elapsed
                    first_clause = clause
                if (
                    first_semantic_clause_s is None
                    and normalized not in _GENERIC_ACKNOWLEDGEMENTS
                ):
                    first_semantic_clause_s = elapsed
        for tc in getattr(delta, "tool_calls", None) or []:
            if tc.function and tc.function.name:
                tool_names.append(tc.function.name)
    total = time.perf_counter() - started
    text = "".join(text_parts)
    silent_response = text.strip().upper() == "[SILENT]"
    if first_semantic_clause_s is None and text.strip() and not silent_response:
        normalized_text = text.strip().rstrip(_STRONG_ENDINGS).strip()
        if normalized_text not in _GENERIC_ACKNOWLEDGEMENTS:
            first_semantic_clause_s = total
    return {
        "first_chunk_ms": round((first_chunk_s or total) * 1000, 1),
        "first_content_ms": (
            round(first_content_s * 1000, 1) if first_content_s is not None else None
        ),
        "first_clause_ms": (
            round(first_clause_s * 1000, 1) if first_clause_s is not None else None
        ),
        "first_semantic_clause_ms": (
            round(first_semantic_clause_s * 1000, 1)
            if first_semantic_clause_s is not None
            else None
        ),
        "total_ms": round(total * 1000, 1),
        "first_clause": first_clause,
        "first_clause_chars": len(first_clause.rstrip(_STRONG_ENDINGS)),
        "first_clause_within_10_chars": bool(first_clause)
        and len(first_clause.rstrip(_STRONG_ENDINGS)) <= 10,
        "generic_acknowledgement": (
            first_clause.rstrip(_STRONG_ENDINGS).strip()
            in _GENERIC_ACKNOWLEDGEMENTS
        ),
        "silent_response": silent_response,
        "text": text[:160],
        "tool_calls": tool_names,
    }


async def main(*, samples: int, output: Path) -> None:
    cfg = get_config(reload=True)
    brain = cfg.get("brain", {})
    voice_model = brain.get("voice_model")
    max_tokens = int(brain.get("voice_max_tokens", 50))
    persona = persona_from_brain_config(brain)
    system_prompt = str(brain.get("system_prompt") or "").strip()
    if not system_prompt:
        system_prompt = persona.build_system_prompt()
    raw_prompt_seed = brain.get("prompt_seed")
    prompt_seed = (
        [
            {"role": str(item["role"]), "content": str(item["content"])}
            for item in raw_prompt_seed
            if isinstance(item, dict) and "role" in item and "content" in item
        ]
        if isinstance(raw_prompt_seed, list) and raw_prompt_seed
        else persona.build_prompt_seed()
    )
    user_prefix = str(brain.get("user_prefix") or "").strip()
    if not user_prefix:
        user_prefix = persona.build_user_prefix()
    client = LLMClient()
    try:
        await run_once(
            client,
            voice_model,
            "你好",
            max_tokens,
            system_prompt=system_prompt,
            prompt_seed=prompt_seed,
            user_prefix=user_prefix,
        )  # warm-up; excluded from measured samples
        questions = [QUESTIONS[index % len(QUESTIONS)] for index in range(samples)]
        runs = [
            await run_once(
                client,
                voice_model,
                question,
                max_tokens,
                system_prompt=system_prompt,
                prompt_seed=prompt_seed,
                user_prefix=user_prefix,
            )
            for question in questions
        ]
        # tool-call correctness check
        tool_run = await run_once(
            client,
            voice_model,
            "带我去服务中心",
            max_tokens,
            system_prompt=system_prompt,
            prompt_seed=prompt_seed,
            user_prefix=user_prefix,
        )
        metric_names = (
            "first_content_ms",
            "first_clause_ms",
            "first_semantic_clause_ms",
            "total_ms",
        )
        summary: dict[str, Any] = {
            metric: {
                "p50": _percentile(
                    [float(run[metric]) for run in runs if run[metric] is not None],
                    0.50,
                ),
                "p95": _percentile(
                    [float(run[metric]) for run in runs if run[metric] is not None],
                    0.95,
                ),
            }
            for metric in metric_names
        }
        spoken_clause_runs = [run for run in runs if run["first_clause"]]
        summary["spoken_clause_count"] = len(spoken_clause_runs)
        summary["first_clause_within_10_chars_rate"] = (
            round(
                sum(
                    bool(run["first_clause_within_10_chars"])
                    for run in spoken_clause_runs
                )
                / len(spoken_clause_runs),
                3,
            )
            if spoken_clause_runs
            else None
        )
        summary["generic_acknowledgement_rate"] = round(
            sum(bool(run["generic_acknowledgement"]) for run in runs) / len(runs),
            3,
        )
        summary["silent_response_rate"] = round(
            sum(bool(run["silent_response"]) for run in runs) / len(runs),
            3,
        )
        result = {
            "evidence_type": "measured",
            "sample_count": len(runs),
            "voice_model": voice_model,
            "summary": summary,
            "runs": runs,
            "tool_run": tool_run,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
    finally:
        raw = getattr(client, "raw_client", None)
        close = getattr(raw, "close", None)
        if callable(close):
            r = close()
            if asyncio.iscoroutine(r):
                await r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/voice_model_verify.json"),
    )
    arguments = parser.parse_args()
    asyncio.run(main(samples=max(1, arguments.samples), output=arguments.output))
