"""Construction boundary for optional realtime dialogue providers."""

from __future__ import annotations

import logging
from typing import Any

from askme.voice.core.realtime_contracts import RealtimeDialogueSession
from askme.voice.realtime.config import (
    RealtimeVoiceMode,
    resolve_realtime_voice_config,
)

logger = logging.getLogger(__name__)


def build_realtime_dialogue(
    config: dict[str, Any] | None,
    *,
    connection_factory: Any | None = None,
) -> RealtimeDialogueSession | None:
    """Build an idle provider or return ``None`` for the cascade fallback."""

    resolved = resolve_realtime_voice_config(config)
    if not resolved.enabled or resolved.mode is RealtimeVoiceMode.SPLIT:
        return None
    errors = resolved.validation_errors()
    if errors:
        logger.warning(
            "Realtime voice disabled; using cascade fallback: %s",
            "; ".join(errors),
        )
        return None
    if resolved.provider == "volcengine_s2s":
        from askme.voice.realtime.volcengine import VolcengineRealtimeDialogue

        return VolcengineRealtimeDialogue(
            resolved,
            connection_factory=connection_factory,
        )
    if resolved.provider == "qwen3_5_omni":
        from askme.voice.realtime.qwen import (
            QwenRealtimeConfig,
            QwenRealtimeDialogue,
        )

        qwen_config = QwenRealtimeConfig(
            enabled=resolved.enabled,
            mode=resolved.mode,
            fallback=resolved.fallback,
            endpoint=resolved.endpoint,
            api_key=resolved.api_key,
            model=resolved.model,
            voice=resolved.speaker,
            bot_name=resolved.bot_name,
            system_role=resolved.system_role,
            speaking_style=resolved.speaking_style,
            input_sample_rate=resolved.input_sample_rate,
            output_sample_rate=resolved.output_sample_rate,
            output_format=resolved.output_format,
            chunk_ms=resolved.chunk_ms,
            vad_silence_duration_ms=resolved.end_smooth_window_ms,
            connect_timeout_s=resolved.connect_timeout_s,
            close_timeout_s=resolved.close_timeout_s,
            audio_queue_ms=resolved.audio_queue_ms,
            event_queue_size=resolved.event_queue_size,
            max_reconnect_attempts=resolved.max_reconnect_attempts,
        )
        return QwenRealtimeDialogue(
            qwen_config,
            connection_factory=connection_factory,
        )
    if resolved.provider == "volcengine_duplex":
        from askme.voice.realtime.volcengine_duplex import (
            VolcengineDuplexConfig,
            VolcengineDuplexDialogue,
        )

        duplex_config = VolcengineDuplexConfig(
            enabled=resolved.enabled,
            api_key=resolved.api_key,
            endpoint=resolved.endpoint,
            model=resolved.model,
            speaker=resolved.speaker,
            bot_name=resolved.bot_name,
            system_role=resolved.system_role,
            speaking_style=resolved.speaking_style,
            input_sample_rate=resolved.input_sample_rate,
            output_sample_rate=resolved.output_sample_rate,
            output_format=resolved.output_format,
            chunk_ms=resolved.chunk_ms,
            connect_timeout_s=resolved.connect_timeout_s,
            close_timeout_s=resolved.close_timeout_s,
            audio_queue_ms=resolved.audio_queue_ms,
            event_queue_size=resolved.event_queue_size,
        )
        return VolcengineDuplexDialogue(
            duplex_config,
            connection_factory=connection_factory,
        )
    logger.warning(
        "Realtime provider %s is unsupported; using cascade fallback",
        resolved.provider,
    )
    return None
