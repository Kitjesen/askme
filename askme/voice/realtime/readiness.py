"""Offline readiness checks for the optional realtime S2S lane."""

from __future__ import annotations

from typing import Any

from askme.voice.realtime.config import (
    RealtimeVoiceMode,
    resolve_realtime_voice_config,
)


def check_realtime_voice(
    config: dict[str, Any] | None,
    *,
    deps: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Validate provider configuration without opening a network connection."""

    resolved = resolve_realtime_voice_config(config)
    dependency_state = deps or {}
    if not resolved.enabled or resolved.mode is RealtimeVoiceMode.SPLIT:
        return {
            "status": "skipped",
            "ok": True,
            "enabled": False,
            "provider": resolved.provider,
            "mode": resolved.mode.value,
            "fallback": resolved.fallback,
            "robot_control_allowed": False,
            "checks": {"websocket_client": bool(dependency_state.get("websocket_client"))},
            "errors": [],
            "warnings": [],
            "config": resolved.status_snapshot(),
        }

    errors = list(resolved.validation_errors())
    websocket_ok = bool(dependency_state.get("websocket_client", False))
    if not websocket_ok:
        errors.append("Realtime S2S dependency missing: websocket-client")
    warnings = [
        "Realtime S2S is an optional general-chat lane; robot tasks stay on the cascade"
    ]
    return {
        "status": "ok" if not errors else "degraded",
        "ok": not errors,
        "enabled": True,
        "provider": resolved.provider,
        "mode": resolved.mode.value,
        "fallback": resolved.fallback,
        "robot_control_allowed": False,
        "checks": {
            "websocket_client": websocket_ok,
            "credentials_configured": resolved.credentials_configured,
            "input_pcm_16khz": resolved.input_sample_rate == 16_000,
            "output_pcm_s16le_24khz": (
                resolved.output_format == "pcm_s16le"
                and resolved.output_sample_rate == 24_000
            ),
        },
        "errors": errors,
        "warnings": warnings,
        "config": resolved.status_snapshot(),
    }
