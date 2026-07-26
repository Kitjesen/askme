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
    logger.warning(
        "Realtime provider %s is unsupported; using cascade fallback",
        resolved.provider,
    )
    return None
