"""Configuration model for optional realtime speech-to-speech providers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

DEFAULT_VOLCENGINE_REALTIME_ENDPOINT = (
    "wss://openspeech.bytedance.com/api/v3/realtime/dialogue"
)
DEFAULT_VOLCENGINE_REALTIME_RESOURCE_ID = "volc.speech.dialog"
DEFAULT_VOLCENGINE_REALTIME_APP_KEY = "PlgvMymc7f3tQnJ6"
SUPPORTED_VOLCENGINE_REALTIME_MODEL = "1.2.1.1"
MIN_END_SMOOTH_WINDOW_MS = 500
MAX_END_SMOOTH_WINDOW_MS = 50_000


class RealtimeVoiceMode(StrEnum):
    """Rollout stages that preserve the existing cascade as fallback."""

    SPLIT = "split"
    SHADOW = "shadow"
    GENERAL_CHAT = "general_chat"


@dataclass(frozen=True)
class RealtimeVoiceConfig:
    enabled: bool = False
    mode: RealtimeVoiceMode = RealtimeVoiceMode.SPLIT
    provider: str = "volcengine_s2s"
    fallback: str = "cascade"
    endpoint: str = DEFAULT_VOLCENGINE_REALTIME_ENDPOINT
    app_id: str = ""
    access_token: str = ""
    resource_id: str = DEFAULT_VOLCENGINE_REALTIME_RESOURCE_ID
    app_key: str = DEFAULT_VOLCENGINE_REALTIME_APP_KEY
    model: str = SUPPORTED_VOLCENGINE_REALTIME_MODEL
    speaker: str = "zh_male_yunzhou_jupiter_bigtts"
    bot_name: str = "小算"
    system_role: str = ""
    speaking_style: str = "简洁、自然、口语化；不要声称已经执行机器人动作。"
    input_mode: str = "audio"
    input_sample_rate: int = 16_000
    output_sample_rate: int = 24_000
    output_format: str = "pcm_s16le"
    chunk_ms: int = 20
    end_smooth_window_ms: int = 800
    connect_timeout_s: float = 4.0
    close_timeout_s: float = 1.0
    audio_queue_ms: int = 400
    event_queue_size: int = 256
    pending_output_ms: int = 2_000
    max_reconnect_attempts: int = 1
    circuit_failure_threshold: int = 3
    circuit_reset_seconds: float = 15.0

    @property
    def credentials_configured(self) -> bool:
        return bool(self.app_id and self.access_token)

    @property
    def available(self) -> bool:
        return self.enabled and not self.validation_errors()

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if self.enabled and not self.credentials_configured:
            errors.append(
                "voice.realtime requires app_id and access_token when enabled"
            )
        if self.provider != "volcengine_s2s":
            errors.append("voice.realtime.provider must be volcengine_s2s")
        if self.endpoint != DEFAULT_VOLCENGINE_REALTIME_ENDPOINT:
            errors.append(
                "voice.realtime.endpoint must use the official WSS endpoint"
            )
        if self.fallback != "cascade":
            errors.append("voice.realtime.fallback must stay cascade")
        if self.input_sample_rate != 16_000:
            errors.append(
                "voice.realtime.input_sample_rate must be 16000 for PCM input"
            )
        if self.output_sample_rate != 24_000:
            errors.append(
                "voice.realtime.output_sample_rate must be 24000 for provider PCM output"
            )
        if self.output_format != "pcm_s16le":
            errors.append(
                "voice.realtime.output_format must be pcm_s16le for the local robot player"
            )
        if self.chunk_ms != 20:
            errors.append("voice.realtime.chunk_ms must be 20")
        if self.model != SUPPORTED_VOLCENGINE_REALTIME_MODEL:
            errors.append("voice.realtime.model must be 1.2.1.1 (O2.0)")
        if not (
            MIN_END_SMOOTH_WINDOW_MS
            <= self.end_smooth_window_ms
            <= MAX_END_SMOOTH_WINDOW_MS
        ):
            errors.append(
                "voice.realtime.end_smooth_window_ms must be between 500 and 50000"
            )
        return errors

    def status_snapshot(self) -> dict[str, Any]:
        errors = self.validation_errors()
        return {
            "enabled": self.enabled,
            "mode": self.mode.value,
            "provider": self.provider,
            "fallback": self.fallback,
            "endpoint": self.endpoint,
            "resource_id": self.resource_id,
            "model": self.model,
            "speaker": self.speaker,
            "credentials_configured": self.credentials_configured,
            "available": self.enabled and not errors,
            "input_sample_rate": self.input_sample_rate,
            "output_sample_rate": self.output_sample_rate,
            "output_format": self.output_format,
            "chunk_ms": self.chunk_ms,
            "end_smooth_window_ms": self.end_smooth_window_ms,
            "errors": errors,
        }


def resolve_realtime_voice_config(config: dict[str, Any] | None) -> RealtimeVoiceConfig:
    """Resolve ``voice.realtime`` without performing network or device I/O."""

    root = config if isinstance(config, dict) else {}
    voice = root.get("voice", root)
    if not isinstance(voice, dict):
        voice = {}
    raw = voice.get("realtime", {})
    if not isinstance(raw, dict):
        raw = {}

    enabled = bool(raw.get("enabled", False))
    mode_raw = str(raw.get("mode", "split")).strip().lower() or "split"
    try:
        mode = RealtimeVoiceMode(mode_raw)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in RealtimeVoiceMode)
        raise ValueError(
            f"voice.realtime.mode must be one of: {allowed}; got {mode_raw!r}"
        ) from exc

    provider_raw = str(raw.get("provider", "volcengine_s2s")).strip().lower()
    provider = {
        "volcengine": "volcengine_s2s",
        "doubao": "volcengine_s2s",
        "doubao_s2s": "volcengine_s2s",
        "volcengine_realtime": "volcengine_s2s",
    }.get(provider_raw, provider_raw or "volcengine_s2s")

    return RealtimeVoiceConfig(
        enabled=enabled,
        mode=mode,
        provider=provider,
        fallback=str(raw.get("fallback", "cascade")).strip().lower() or "cascade",
        endpoint=str(raw.get("endpoint", DEFAULT_VOLCENGINE_REALTIME_ENDPOINT)).strip(),
        app_id=str(raw.get("app_id", "")).strip(),
        access_token=str(raw.get("access_token", raw.get("access_key", ""))).strip(),
        resource_id=str(
            raw.get("resource_id", DEFAULT_VOLCENGINE_REALTIME_RESOURCE_ID)
        ).strip(),
        app_key=str(raw.get("app_key", DEFAULT_VOLCENGINE_REALTIME_APP_KEY)).strip(),
        model=(
            str(raw.get("model", SUPPORTED_VOLCENGINE_REALTIME_MODEL)).strip()
            or SUPPORTED_VOLCENGINE_REALTIME_MODEL
        ),
        speaker=str(
            raw.get("speaker", "zh_male_yunzhou_jupiter_bigtts")
        ).strip(),
        bot_name=str(raw.get("bot_name", "小算")).strip() or "小算",
        system_role=str(raw.get("system_role", "")).strip(),
        speaking_style=str(
            raw.get(
                "speaking_style",
                "简洁、自然、口语化；不要声称已经执行机器人动作。",
            )
        ).strip(),
        input_mode=str(raw.get("input_mode", "audio")).strip() or "audio",
        input_sample_rate=int(raw.get("input_sample_rate", 16_000)),
        output_sample_rate=int(raw.get("output_sample_rate", 24_000)),
        output_format=str(raw.get("output_format", "pcm_s16le")).strip().lower(),
        chunk_ms=int(raw.get("chunk_ms", 20)),
        end_smooth_window_ms=int(raw.get("end_smooth_window_ms", 800)),
        connect_timeout_s=float(raw.get("connect_timeout_s", 4.0)),
        close_timeout_s=float(raw.get("close_timeout_s", 1.0)),
        audio_queue_ms=int(raw.get("audio_queue_ms", 400)),
        event_queue_size=int(raw.get("event_queue_size", 256)),
        pending_output_ms=int(raw.get("pending_output_ms", 2_000)),
        max_reconnect_attempts=int(raw.get("max_reconnect_attempts", 1)),
        circuit_failure_threshold=int(raw.get("circuit_failure_threshold", 3)),
        circuit_reset_seconds=float(raw.get("circuit_reset_seconds", 15.0)),
    )
