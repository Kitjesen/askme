"""Voice diagnostics, readiness, and smoke-check package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "check_minimax_hybrid_voice_brain": (
        "askme.voice.diagnostics.minimax_hybrid",
        "check_minimax_hybrid_voice_brain",
    ),
    "collect_runtime_mic_calibration": (
        "askme.voice.diagnostics.mic_calibration",
        "collect_runtime_mic_calibration",
    ),
    "collect_s100p_readiness_bundle": (
        "askme.voice.diagnostics.s100p_readiness_bundle",
        "collect_s100p_readiness_bundle",
    ),
    "print_audio_devices_summary": (
        "askme.voice.diagnostics.audio_devices",
        "print_audio_devices_summary",
    ),
    "print_mic_calibration_summary": (
        "askme.voice.diagnostics.mic_calibration",
        "print_mic_calibration_summary",
    ),
    "print_s100p_readiness_bundle_summary": (
        "askme.voice.diagnostics.s100p_readiness_bundle",
        "print_s100p_readiness_bundle_summary",
    ),
    "print_sunrise_audio_doctor_summary": (
        "askme.voice.diagnostics.sunrise_audio_doctor",
        "print_sunrise_audio_doctor_summary",
    ),
    "print_sunrise_voice_readiness_summary": (
        "askme.voice.diagnostics.sunrise_readiness",
        "print_sunrise_voice_readiness_summary",
    ),
    "print_voice_health_summary": (
        "askme.voice.diagnostics.health_check",
        "print_voice_health_summary",
    ),
    "print_voice_online_smoke_summary": (
        "askme.voice.diagnostics.online_smoke",
        "print_voice_online_smoke_summary",
    ),
    "print_windows_beep_loopback_summary": (
        "askme.voice.diagnostics.audio_devices",
        "print_windows_beep_loopback_summary",
    ),
    "query_audio_devices": ("askme.voice.diagnostics.audio_devices", "query_audio_devices"),
    "resolve_minimax_hybrid_voice_brain": (
        "askme.voice.diagnostics.minimax_hybrid",
        "resolve_minimax_hybrid_voice_brain",
    ),
    "run_audio_loopback": ("askme.voice.diagnostics.audio_devices", "run_audio_loopback"),
    "run_audio_route_scan": ("askme.voice.diagnostics.audio_devices", "run_audio_route_scan"),
    "run_windows_beep_loopback": (
        "askme.voice.diagnostics.audio_devices",
        "run_windows_beep_loopback",
    ),
    "run_sunrise_audio_doctor": (
        "askme.voice.diagnostics.sunrise_audio_doctor",
        "run_sunrise_audio_doctor",
    ),
    "run_sunrise_voice_readiness": (
        "askme.voice.diagnostics.sunrise_readiness",
        "run_sunrise_voice_readiness",
    ),
    "run_voice_health": ("askme.voice.diagnostics.health_check", "run_voice_health"),
    "run_voice_online_smoke_sync": (
        "askme.voice.diagnostics.online_smoke",
        "run_voice_online_smoke_sync",
    ),
    "write_mic_calibration_json": (
        "askme.voice.diagnostics.mic_calibration",
        "write_mic_calibration_json",
    ),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
