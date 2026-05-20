"""Online voice-provider smoke checks for MiniMax and DashScope."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np

from askme.config import get_config
from askme.llm.core.client import LLMClient
from askme.voice.input.cloud_asr import CloudASR
from askme.voice.output.tts import TTSEngine


async def _check_minimax_llm() -> dict[str, Any]:
    started = time.perf_counter()
    try:
        client = LLMClient()
        text = await client.chat(
            [
                {"role": "system", "content": "Only reply with OK."},
                {"role": "user", "content": "online connectivity smoke test"},
            ],
            temperature=0.0,
        )
        return {
            "status": "ok",
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "text_preview": (text or "")[:32],
        }
    except Exception as exc:  # pragma: no cover - network/runtime path
        return {
            "status": "error",
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "error": str(exc)[:300],
        }


async def _check_minimax_tts(tts_cfg: dict[str, Any], *, text: str) -> dict[str, Any]:
    started = time.perf_counter()
    engine: TTSEngine | None = None
    try:
        engine = TTSEngine(dict(tts_cfg))
        generation = getattr(engine, "_generation", 0) + 1
        engine._generation = generation
        ok = await engine._generate_minimax_transport(text, generation)
        with engine._buffer_lock:
            samples = int(sum(len(chunk) for chunk in engine.tts_buffer))
        return {
            "status": "ok" if ok and samples > 0 else "degraded",
            "ok": bool(ok),
            "samples": samples,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "voice_id": getattr(engine, "_minimax_voice_id", ""),
            "model": getattr(engine, "_minimax_tts_model", ""),
            "transport": getattr(engine, "_minimax_tts_transport", ""),
            "audio_format": getattr(engine, "_minimax_audio_format", ""),
        }
    except Exception as exc:  # pragma: no cover - network/runtime path
        voice_id = str(tts_cfg.get("minimax_voice_id", ""))
        model = str(tts_cfg.get("minimax_tts_model", ""))
        return {
            "status": "error",
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "voice_id": voice_id,
            "model": model,
            "error": str(exc)[:300],
        }
    finally:
        if engine is not None:
            engine.shutdown()


def _check_dashscope_asr(cloud_cfg: dict[str, Any], *, silence_seconds: float) -> dict[str, Any]:
    started = time.perf_counter()
    cloud = CloudASR(cloud_cfg)
    try:
        if not cloud.available:
            return {
                "status": "error",
                "available": False,
                "error": "DashScope ASR is not enabled or api_key is missing",
            }
        if not cloud.start_session():
            return {
                "status": "error",
                "available": True,
                "error": getattr(cloud, "_last_session_error", "start_session failed")[:300],
            }
        rate = int(cloud_cfg.get("sample_rate", 16000) or 16000)
        silence = np.zeros(max(1, int(rate * silence_seconds)), dtype="<i2")
        cloud.feed(silence.tobytes())
        transcript = cloud.finish_session(timeout=3.0)
        return {
            "status": "ok",
            "available": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "transcript_preview": transcript[:32],
        }
    except Exception as exc:  # pragma: no cover - network/runtime path
        try:
            cloud.cancel_session()
        except Exception:
            pass
        return {
            "status": "error",
            "available": True,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "error": str(exc)[:300],
        }


async def run_voice_online_smoke(
    *,
    text: str = "你好，请问需要指路吗？服务中心在前方右转。",
    silence_seconds: float = 0.2,
) -> dict[str, Any]:
    """Run real network checks without exposing provider secrets."""
    cfg = get_config(reload=True)
    voice_cfg = cfg.get("voice", {}) if isinstance(cfg, dict) else {}
    tts_cfg = voice_cfg.get("tts", {}) if isinstance(voice_cfg, dict) else {}
    cloud_cfg = voice_cfg.get("cloud_asr", {}) if isinstance(voice_cfg, dict) else {}
    checks = {
        "minimax_llm": await _check_minimax_llm(),
        "minimax_tts": await _check_minimax_tts(tts_cfg, text=text),
        "dashscope_asr": _check_dashscope_asr(
            cloud_cfg,
            silence_seconds=silence_seconds,
        ),
    }
    keys_present = {
        "minimax_llm": bool(str(cfg.get("brain", {}).get("api_key", "")).strip()),
        "minimax_tts": bool(str(tts_cfg.get("minimax_api_key", "")).strip()),
        "dashscope_asr": bool(str(cloud_cfg.get("api_key", "")).strip()),
    }
    status = "ok" if all(check.get("status") == "ok" for check in checks.values()) else "degraded"
    return {
        "status": status,
        "text": text,
        "keys_present": keys_present,
        "checks": checks,
    }


def run_voice_online_smoke_sync(**kwargs: Any) -> dict[str, Any]:
    return asyncio.run(run_voice_online_smoke(**kwargs))


def print_voice_online_smoke_summary(payload: dict[str, Any]) -> None:
    print(f"Voice online smoke: {payload.get('status', 'unknown')}")  # noqa: T201
    for name, check in payload.get("checks", {}).items():
        line = f"  {name}: {check.get('status')}"
        if check.get("latency_ms") is not None:
            line += f" {check.get('latency_ms')}ms"
        if check.get("samples") is not None:
            line += f" samples={check.get('samples')}"
        if check.get("error"):
            line += f" error={check.get('error')}"
        print(line)  # noqa: T201
