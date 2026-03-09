"""Runtime health snapshot helpers for HTTP and OTA observability."""

from __future__ import annotations

from typing import Any, Callable


class RuntimeHealthSnapshot:
    """Build compact and detailed health payloads from live askme state."""

    def __init__(
        self,
        *,
        app_name: str,
        app_version: str,
        brain_config: dict[str, Any],
        voice_mode: bool,
        robot_mode: bool,
        metrics_provider: Callable[[], dict[str, Any]],
        active_skill_names_provider: Callable[[], list[str]],
        voice_status_provider: Callable[[], dict[str, Any]],
        ota_status_provider: Callable[[], dict[str, Any]],
    ) -> None:
        self._app_name = app_name
        self._app_version = app_version
        self._brain_config = brain_config
        self._voice_mode = voice_mode
        self._robot_mode = robot_mode
        self._metrics_provider = metrics_provider
        self._active_skill_names_provider = active_skill_names_provider
        self._voice_status_provider = voice_status_provider
        self._ota_status_provider = ota_status_provider

    def health_snapshot(self) -> dict[str, Any]:
        """Return the compact health document for `/health`."""
        metrics = self.metrics_snapshot()
        return {
            "status": metrics["status"],
            "degraded_reasons": metrics["degraded_reasons"],
            "service_name": metrics["service_name"],
            "service_version": metrics["service_version"],
            "uptime_seconds": metrics["uptime_seconds"],
            "model_name": metrics["model_name"],
            "last_llm_latency_ms": metrics["last_llm_latency_ms"],
            "total_conversations": metrics["total_conversations"],
            "active_skills": metrics["active_skills"],
            "active_skill_count": metrics["active_skill_count"],
            "voice_pipeline_status": metrics["voice_pipeline_status"],
            "ota_bridge_status": metrics["ota_bridge_status"],
        }

    def metrics_snapshot(self) -> dict[str, Any]:
        """Return the detailed runtime document for `/metrics`."""
        metrics = self._metrics_provider()
        llm_metrics = dict(metrics.get("llm", {}))
        skill_metrics = dict(metrics.get("skills", {}))
        active_skills = _normalise_skill_names(self._active_skill_names_provider())
        voice_pipeline = merge_voice_pipeline_status(
            self._voice_status_provider(),
            metrics.get("voice_pipeline", {}),
        )
        ota_bridge = dict(self._ota_status_provider())
        configured_model = _clean_optional(self._brain_config.get("model"))
        voice_model = _clean_optional(self._brain_config.get("voice_model"))
        last_model = _clean_optional(llm_metrics.get("last_model"))
        active_model = last_model or configured_model or voice_model or "unknown"
        status, degraded_reasons = self._overall_status(
            voice_pipeline=voice_pipeline,
            ota_bridge=ota_bridge,
        )
        service_version = self._app_version or "unknown"
        total_conversations = int(metrics.get("conversation_count", 0) or 0)

        return {
            "status": status,
            "degraded_reasons": degraded_reasons,
            "service_name": self._app_name,
            "service_version": service_version,
            "voice_mode": self._voice_mode,
            "robot_mode": self._robot_mode,
            "uptime_seconds": metrics.get("uptime_seconds", 0.0),
            "configured_model_name": configured_model,
            "voice_model_name": voice_model,
            "model_name": active_model,
            "last_llm_latency_ms": llm_metrics.get("last_latency_ms"),
            "total_conversations": total_conversations,
            "llm": llm_metrics,
            "active_skills": active_skills,
            "active_skill_count": len(active_skills),
            "skills": {
                **skill_metrics,
                "active": active_skills,
                "active_count": len(active_skills),
            },
            "voice_pipeline_status": voice_pipeline,
            "ota_bridge_status": ota_bridge,
        }

    @staticmethod
    def _overall_status(
        *,
        voice_pipeline: dict[str, Any],
        ota_bridge: dict[str, Any],
    ) -> tuple[str, list[str]]:
        degraded_reasons: list[str] = []

        if not voice_pipeline.get("pipeline_ok", False):
            degraded_reasons.append("voice_pipeline_unavailable")

        if ota_bridge.get("enabled") and ota_bridge.get("state") in {
            "auth_error",
            "degraded",
            "stopped",
        }:
            degraded_reasons.append(f"ota_bridge_{ota_bridge.get('state', 'unknown')}")

        if degraded_reasons:
            return ("degraded", degraded_reasons)

        return ("ok", [])


def merge_voice_pipeline_status(
    live_status: dict[str, Any] | None,
    metrics_status: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge live audio readiness with the recent OTA voice metrics."""
    merged = dict(metrics_status or {})
    merged.update(live_status or {})

    mode = str(merged.get("mode", "text") or "text")
    enabled = bool(merged.get("enabled", mode == "voice"))
    asr_available = bool(merged.get("asr_available", False))
    vad_available = bool(merged.get("vad_available", False))
    kws_available = bool(merged.get("kws_available", False))
    tts_backend = _clean_optional(merged.get("tts_backend"))
    output_ready = bool(merged.get("output_ready", tts_backend is not None))
    input_ready = bool(
        merged.get(
            "input_ready",
            enabled and asr_available and vad_available,
        )
    )

    merged["mode"] = mode
    merged["enabled"] = enabled
    merged["input_ready"] = input_ready
    merged["output_ready"] = output_ready
    merged["asr_available"] = asr_available
    merged["vad_available"] = vad_available
    merged["kws_available"] = kws_available
    merged["wake_word_enabled"] = bool(
        merged.get("wake_word_enabled", enabled and kws_available)
    )
    merged["tts_backend"] = tts_backend
    merged["tts_busy"] = bool(merged.get("tts_busy", False))
    merged["last_input_chars"] = int(merged.get("last_input_chars", 0) or 0)
    merged["pipeline_ok"] = bool(
        merged.get("pipeline_ok", output_ready and (not enabled or input_ready))
    )

    return merged


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalise_skill_names(skill_names: list[str]) -> list[str]:
    cleaned = {
        str(skill_name).strip()
        for skill_name in skill_names
        if str(skill_name).strip()
    }
    return sorted(cleaned)
