"""Prometheus text exposition for askme runtime snapshots."""

from __future__ import annotations

import math
from typing import Any


def render_prometheus_metrics(snapshot: dict[str, Any]) -> str:
    """Render the runtime snapshot as Prometheus text exposition."""

    voice_status = snapshot.get("voice_pipeline_status", {})
    voice_input = voice_status.get("input", {}) if isinstance(voice_status, dict) else {}
    active_skills = snapshot.get("active_skills", [])
    ota_status = snapshot.get("ota_bridge_status") or snapshot.get("ota_bridge") or {}

    lines: list[str] = []
    append_metric(lines, "askme_up", "Whether the askme process is running", "gauge", 1)
    append_metric(
        lines,
        "askme_service_info",
        "Static askme service metadata",
        "gauge",
        1,
        labels={
            "service": snapshot.get("service") or snapshot.get("service_name", "askme"),
            "version": snapshot.get("version") or snapshot.get("service_version", "unknown"),
        },
    )
    append_metric(
        lines,
        "askme_model_info",
        "Configured primary LLM model",
        "gauge",
        1,
        labels={"model_name": snapshot.get("model_name", "unknown")},
    )
    append_metric(
        lines,
        "askme_health_status",
        "Overall askme health status (1=ok, 0=degraded)",
        "gauge",
        snapshot.get("status") == "ok",
    )
    append_metric(
        lines,
        "askme_uptime_seconds",
        "Process uptime in seconds",
        "gauge",
        snapshot.get("uptime_seconds"),
    )
    append_metric(
        lines,
        "askme_conversations_total",
        "Total conversation turns recorded",
        "counter",
        snapshot.get("total_conversations"),
    )
    append_metric(
        lines,
        "askme_last_llm_latency_ms",
        "Latency of the most recent LLM call in milliseconds",
        "gauge",
        snapshot.get("last_llm_latency_ms"),
    )
    llm_snap = snapshot.get("llm", {})
    append_metric(
        lines,
        "askme_llm_latency_p50_ms",
        "LLM call latency p50 over last 100 calls (ms)",
        "gauge",
        llm_snap.get("p50_latency_ms"),
    )
    append_metric(
        lines,
        "askme_llm_latency_p95_ms",
        "LLM call latency p95 over last 100 calls (ms)",
        "gauge",
        llm_snap.get("p95_latency_ms"),
    )
    append_metric(
        lines,
        "askme_llm_latency_p99_ms",
        "LLM call latency p99 over last 100 calls (ms)",
        "gauge",
        llm_snap.get("p99_latency_ms"),
    )
    conversation_runtime = snapshot.get("conversation_runtime", {})
    if isinstance(conversation_runtime, dict):
        append_metric(
            lines,
            "askme_chat_in_flight",
            "Current in-flight HTTP chat requests",
            "gauge",
            conversation_runtime.get("in_flight"),
        )
        append_metric(
            lines,
            "askme_chat_turns_total",
            "Total HTTP chat turns handled by this process",
            "counter",
            conversation_runtime.get("total_turns"),
        )
        append_metric(
            lines,
            "askme_chat_failures_total",
            "Total HTTP chat turns that ended in an error or timeout",
            "counter",
            conversation_runtime.get("failures"),
        )
        append_metric(
            lines,
            "askme_chat_timeouts_total",
            "Total HTTP chat turns that exceeded the configured timeout",
            "counter",
            conversation_runtime.get("timeouts"),
        )
        append_metric(
            lines,
            "askme_chat_overloads_total",
            "Total HTTP chat turns rejected by the concurrency limiter",
            "counter",
            conversation_runtime.get("overloads"),
        )
        append_metric(
            lines,
            "askme_chat_slow_turns_total",
            "Total HTTP chat turns above the configured slow threshold",
            "counter",
            conversation_runtime.get("slow_turns_total"),
        )
        append_metric(
            lines,
            "askme_chat_last_turn_latency_ms",
            "Total latency of the most recent HTTP chat turn in milliseconds",
            "gauge",
            conversation_runtime.get("last_turn_latency_ms"),
        )
        append_metric(
            lines,
            "askme_chat_last_handler_ms",
            "Handler latency of the most recent HTTP chat turn in milliseconds",
            "gauge",
            conversation_runtime.get("last_handler_ms"),
        )
    append_metric(
        lines,
        "askme_active_skills",
        "Number of currently enabled skills",
        "gauge",
        snapshot.get(
            "active_skill_count",
            len(active_skills) if isinstance(active_skills, list) else 0,
        ),
    )

    for skill_name in active_skills if isinstance(active_skills, list) else []:
        append_metric(
            lines,
            "askme_active_skill_info",
            "Enabled skill metadata",
            "gauge",
            1,
            labels={"skill": skill_name},
        )

    append_metric(
        lines,
        "askme_voice_pipeline_ok",
        "Whether the voice pipeline is currently healthy",
        "gauge",
        voice_status.get("pipeline_ok"),
    )
    append_metric(
        lines,
        "askme_voice_mode_enabled",
        "Whether askme is running in voice mode",
        "gauge",
        voice_status.get("mode") == "voice",
    )
    append_metric(
        lines,
        "askme_voice_input_ready",
        "Whether ASR and VAD are available for voice input",
        "gauge",
        voice_status.get("input_ready"),
    )
    append_metric(
        lines,
        "askme_voice_output_ready",
        "Whether TTS output is available",
        "gauge",
        voice_status.get("output_ready"),
    )
    append_metric(
        lines,
        "askme_voice_asr_available",
        "Whether the ASR engine is available",
        "gauge",
        voice_status.get("asr_available"),
    )
    append_metric(
        lines,
        "askme_voice_vad_available",
        "Whether the VAD engine is available",
        "gauge",
        voice_status.get("vad_available"),
    )
    append_metric(
        lines,
        "askme_voice_kws_available",
        "Whether the wake-word detector is available",
        "gauge",
        voice_status.get("kws_available"),
    )
    append_metric(
        lines,
        "askme_voice_tts_busy",
        "Whether TTS is currently playing or queued",
        "gauge",
        voice_status.get("tts_busy"),
    )
    append_metric(
        lines,
        "askme_voice_last_input_chars",
        "Character length of the most recent recognized voice input",
        "gauge",
        voice_status.get("last_input_chars"),
    )
    append_metric(
        lines,
        "askme_voice_input_last_peak",
        "Most recent observed microphone peak amplitude",
        "gauge",
        voice_input.get("last_peak") if isinstance(voice_input, dict) else None,
    )
    append_metric(
        lines,
        "askme_voice_input_peak_max_10s",
        "Maximum observed microphone peak amplitude over the recent window",
        "gauge",
        voice_input.get("peak_max_10s") if isinstance(voice_input, dict) else None,
    )
    append_metric(
        lines,
        "askme_voice_input_peak_p95_10s",
        "P95 observed microphone peak amplitude over the recent window",
        "gauge",
        voice_input.get("peak_p95_10s") if isinstance(voice_input, dict) else None,
    )
    append_metric(
        lines,
        "askme_voice_input_last_rms",
        "Most recent observed microphone RMS amplitude",
        "gauge",
        voice_input.get("last_rms") if isinstance(voice_input, dict) else None,
    )
    append_metric(
        lines,
        "askme_voice_input_rms_p95_10s",
        "P95 observed microphone RMS amplitude over the recent window",
        "gauge",
        voice_input.get("rms_p95_10s") if isinstance(voice_input, dict) else None,
    )
    append_metric(
        lines,
        "askme_voice_input_asr_timeouts",
        "Count of ASR listen timeouts observed by the audio agent",
        "counter",
        voice_input.get("asr_timeouts") if isinstance(voice_input, dict) else None,
    )
    append_metric(
        lines,
        "askme_voice_input_sample_count_10s",
        "Number of microphone frames in the recent input diagnostics window",
        "gauge",
        voice_input.get("sample_count_10s") if isinstance(voice_input, dict) else None,
    )

    append_metric(
        lines,
        "askme_ota_bridge_enabled",
        "Whether OTA bridge reporting is enabled",
        "gauge",
        ota_status.get("enabled"),
    )
    append_metric(
        lines,
        "askme_ota_bridge_registered",
        "Whether the OTA bridge currently has valid registration",
        "gauge",
        ota_status.get("registered"),
    )
    append_metric(
        lines,
        "askme_ota_bridge_info",
        "Static OTA bridge metadata",
        "gauge",
        1,
        labels={
            "channel": ota_status.get("channel", ""),
            "device_id": ota_status.get("device_id", ""),
            "product": ota_status.get("product", ""),
            "state": ota_status.get("state", ""),
        },
    )

    return "".join(lines)


def append_metric(
    lines: list[str],
    name: str,
    help_text: str,
    metric_type: str,
    value: Any,
    *,
    labels: dict[str, Any] | None = None,
) -> None:
    """Append one Prometheus metric including HELP and TYPE lines."""

    lines.append(f"# HELP {name} {help_text}\n")
    lines.append(f"# TYPE {name} {metric_type}\n")
    lines.append(f"{name}{format_labels(labels)} {format_metric_value(value)}\n")


def format_labels(labels: dict[str, Any] | None) -> str:
    """Format Prometheus labels in stable key order."""

    if not labels:
        return ""

    parts = [
        f'{key}="{escape_label_value(value)}"'
        for key, value in sorted(labels.items())
    ]
    return "{" + ",".join(parts) + "}"


def escape_label_value(value: Any) -> str:
    """Escape a Prometheus label value."""

    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def format_metric_value(value: Any) -> str:
    """Format a value using Prometheus numeric text rules."""

    if value is None:
        return "NaN"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NaN"
        return f"{value:.6f}".rstrip("0").rstrip(".")

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NaN"
    if not math.isfinite(numeric):
        return "NaN"
    return f"{numeric:.6f}".rstrip("0").rstrip(".")


__all__ = [
    "append_metric",
    "escape_label_value",
    "format_labels",
    "format_metric_value",
    "render_prometheus_metrics",
]
