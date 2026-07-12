"""Online checks for the configured LLM and optional cloud voice providers."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np

from askme.config import get_config
from askme.llm.core.client import LLMClient
from askme.voice.input.cloud_asr import CloudASR, cloud_asr_credentials_present
from askme.voice.output.tts import TTSEngine


async def _check_llm() -> dict[str, Any]:
    started = time.perf_counter()
    client: LLMClient | None = None
    try:
        client = LLMClient()
        provider = client.provider_status()
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
            "provider": provider.get("provider", ""),
            "model": provider.get("model", ""),
        }
    except Exception as exc:  # pragma: no cover - network/runtime path
        return {
            "status": "error",
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "error": str(exc)[:300],
        }
    finally:
        raw_client = getattr(client, "raw_client", None) if client is not None else None
        close = getattr(raw_client, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result


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


def _check_cloud_asr(cloud_cfg: dict[str, Any], *, silence_seconds: float) -> dict[str, Any]:
    started = time.perf_counter()
    cloud = CloudASR(cloud_cfg)
    try:
        if not cloud.available:
            return {
                "status": "error",
                "available": False,
                "error": "Cloud ASR is not enabled or credentials are incomplete",
            }
        if not cloud.start_session():
            snapshot = cloud.status_snapshot()
            return {
                "status": "error",
                "available": True,
                "provider": snapshot.get("provider", ""),
                "error": str(
                    snapshot.get("last_error")
                    or getattr(cloud, "_last_session_error", "start_session failed")
                )[:300],
            }
        rate = int(cloud_cfg.get("sample_rate", 16000) or 16000)
        silence = np.zeros(max(1, int(rate * silence_seconds)), dtype="<i2")
        cloud.feed(silence.tobytes())
        transcript = cloud.finish_session(timeout=3.0)
        snapshot = cloud.status_snapshot()
        return {
            "status": "ok",
            "available": True,
            "provider": snapshot.get("provider", ""),
            "resource_id": snapshot.get("resource_id", ""),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "transcript_preview": transcript[:32],
            "log_id": snapshot.get("log_id", ""),
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
        "llm": await _check_llm(),
        "minimax_tts": await _check_minimax_tts(tts_cfg, text=text),
        "cloud_asr": _check_cloud_asr(
            cloud_cfg,
            silence_seconds=silence_seconds,
        ),
    }
    keys_present = {
        "llm": bool(str(cfg.get("brain", {}).get("api_key", "")).strip()),
        "minimax_tts": bool(str(tts_cfg.get("minimax_api_key", "")).strip()),
        "cloud_asr": cloud_asr_credentials_present(cloud_cfg),
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
    logger.info(f"Voice online smoke: {payload.get('status', 'unknown')}")
    for name, check in payload.get("checks", {}).items():
        line = f"  {name}: {check.get('status')}"
        if check.get("latency_ms") is not None:
            line += f" {check.get('latency_ms')}ms"
        if check.get("samples") is not None:
            line += f" samples={check.get('samples')}"
        if check.get("error"):
            line += f" error={check.get('error')}"
        logger.info(line)
