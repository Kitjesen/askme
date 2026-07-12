"""Runtime microphone gate calibration helpers."""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_RUNTIME_URL = "http://127.0.0.1:8765"


def collect_runtime_mic_calibration(
    *,
    server: str = DEFAULT_RUNTIME_URL,
    duration_s: float = 5.0,
    interval_s: float = 0.5,
    timeout_s: float = 5.0,
    min_signal_peak: int = 100,
) -> dict[str, Any]:
    """Poll runtime `/health` and summarize recent microphone input levels."""
    duration_s = max(0.0, float(duration_s))
    interval_s = max(0.1, float(interval_s))
    min_signal_peak = max(0, int(min_signal_peak))
    samples: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    latest_health: dict[str, Any] = {}

    started = time.time()
    deadline = started + duration_s
    while True:
        try:
            latest_health = _fetch_health(server, timeout_s=timeout_s)
            input_status = _extract_input_status(latest_health)
            samples.append(_sample_from_input_status(input_status))
        except Exception as exc:
            errors.append(str(exc))

        if time.time() >= deadline:
            break
        time.sleep(min(interval_s, max(0.0, deadline - time.time())))

    latest_input = _extract_input_status(latest_health) if latest_health else {}
    observed_peak_values = [
        int(sample["last_peak"])
        for sample in samples
        if isinstance(sample.get("last_peak"), int)
    ]
    observed_peak_values.extend(
        int(sample["peak_max_10s"])
        for sample in samples
        if isinstance(sample.get("peak_max_10s"), int)
    )
    observed_peak_max = max(observed_peak_values, default=0)
    observed_peak_p95 = _percentile(observed_peak_values, 95)
    noise_gate_peak = _safe_int(latest_input.get("noise_gate_peak"), 0)
    recommended_gate = _recommended_noise_gate(observed_peak_max, noise_gate_peak)

    mic_open = bool(latest_input.get("mic_open", False))
    if not mic_open:
        errors.append("mic_not_open")
    if not samples:
        errors.append("no_health_samples")
    if noise_gate_peak > 0 and 0 < observed_peak_max < noise_gate_peak:
        warnings.append(f"observed_peak_below_noise_gate:{observed_peak_max}<{noise_gate_peak}")
    if observed_peak_max < min_signal_peak:
        warnings.append(f"observed_peak_below_min_signal:{observed_peak_max}<{min_signal_peak}")

    status = "ok" if not errors and not warnings else "degraded"
    return {
        "status": status,
        "target": "runtime-mic-calibration",
        "server": _normalise_server_url(server),
        "duration_s": round(time.time() - started, 2),
        "interval_s": interval_s,
        "sample_count": len(samples),
        "errors": _dedupe(errors),
        "warnings": _dedupe(warnings),
        "summary": {
            "run_id": latest_input.get("run_id"),
            "mic_open": mic_open,
            "transport": latest_input.get("transport"),
            "device": latest_input.get("device"),
            "sample_rate": latest_input.get("sample_rate"),
            "native_rate": latest_input.get("native_rate"),
            "channels": latest_input.get("channels"),
            "channel_select": latest_input.get("channel_select"),
            "noise_gate_peak": noise_gate_peak,
            "observed_peak_max": observed_peak_max,
            "observed_peak_p95": observed_peak_p95,
            "runtime_peak_p95_10s": latest_input.get("peak_p95_10s"),
            "runtime_rms_p95_10s": latest_input.get("rms_p95_10s"),
            "gate_state": latest_input.get("gate_state"),
            "vad_state": latest_input.get("vad_state"),
            "asr_timeouts": latest_input.get("asr_timeouts"),
            "last_failure_reason": latest_input.get("last_failure_reason"),
            "recommended_noise_gate_peak": recommended_gate,
            "recommendation": _recommendation_text(
                observed_peak_max=observed_peak_max,
                noise_gate_peak=noise_gate_peak,
                min_signal_peak=min_signal_peak,
                mic_open=mic_open,
            ),
        },
        "samples": samples,
    }


def print_mic_calibration_summary(payload: dict[str, Any]) -> None:
    """Print a compact human-readable calibration summary."""
    summary = payload.get("summary", {})
    logger.info(f"runtime-mic-calibration: {payload.get('status', 'unknown')}")
    logger.info(f"  server: {payload.get('server', '')}")
    logger.info(f"  samples: {payload.get('sample_count', 0)}")
    logger.info(
        "  mic: "
        f"open={summary.get('mic_open')} "
        f"transport={summary.get('transport')} "
        f"device={summary.get('device')}"
    )
    logger.info(
        "  peak: "
        f"max={summary.get('observed_peak_max')} "
        f"p95={summary.get('observed_peak_p95')} "
        f"gate={summary.get('noise_gate_peak')} "
        f"recommended_gate={summary.get('recommended_noise_gate_peak')}"
    )
    logger.info(
        "  state: "
        f"gate={summary.get('gate_state')} "
        f"vad={summary.get('vad_state')} "
        f"asr_timeouts={summary.get('asr_timeouts')}"
    )
    recommendation = summary.get("recommendation")
    if recommendation:
        logger.info(f"  recommendation: {recommendation}")
    for warning in payload.get("warnings", []):
        logger.warning(f"  warning: {warning}")
    for error in payload.get("errors", []):
        logger.error(f"  error: {error}")


def write_mic_calibration_json(payload: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _fetch_health(server: str, *, timeout_s: float) -> dict[str, Any]:
    url = f"{_normalise_server_url(server)}/health"
    with urllib.request.urlopen(url, timeout=max(float(timeout_s), 0.1)) as response:
        body = response.read().decode("utf-8")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise RuntimeError("health endpoint returned non-object JSON")
    return payload


def _extract_input_status(health: dict[str, Any]) -> dict[str, Any]:
    voice_status = health.get("voice_pipeline_status", {})
    if not isinstance(voice_status, dict):
        return {}
    input_status = voice_status.get("input", {})
    return input_status if isinstance(input_status, dict) else {}


def _sample_from_input_status(input_status: dict[str, Any]) -> dict[str, Any]:
    return {
        "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "last_peak": _safe_int(input_status.get("last_peak"), 0),
        "peak_max_10s": _safe_int(input_status.get("peak_max_10s"), 0),
        "peak_p95_10s": input_status.get("peak_p95_10s"),
        "last_rms": input_status.get("last_rms"),
        "rms_p95_10s": input_status.get("rms_p95_10s"),
        "gate_state": input_status.get("gate_state"),
        "vad_state": input_status.get("vad_state"),
        "asr_timeouts": input_status.get("asr_timeouts"),
    }


def _normalise_server_url(server: str) -> str:
    return str(server or DEFAULT_RUNTIME_URL).rstrip("/")


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _percentile(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    index = (max(0.0, min(percentile, 100.0)) / 100.0) * (len(ordered) - 1)
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    frac = index - lo
    return round(ordered[lo] * (1.0 - frac) + ordered[hi] * frac, 2)


def _recommended_noise_gate(observed_peak_max: int, noise_gate_peak: int) -> int:
    if observed_peak_max <= 0:
        return 0
    if noise_gate_peak <= 0:
        return 0
    if observed_peak_max < noise_gate_peak:
        return max(0, min(noise_gate_peak - 1, int(observed_peak_max * 0.6)))
    return noise_gate_peak


def _recommendation_text(
    *,
    observed_peak_max: int,
    noise_gate_peak: int,
    min_signal_peak: int,
    mic_open: bool,
) -> str:
    if not mic_open:
        return "microphone stream is not open; start a voice runtime first"
    if observed_peak_max < min_signal_peak:
        return "input signal is very low; check mic device/channel/gain before tuning ASR"
    if noise_gate_peak > 0 and observed_peak_max < noise_gate_peak:
        return "observed signal is below noise_gate_peak; lower gate only after confirming this was speech"
    return "input signal is above the configured gate"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result
