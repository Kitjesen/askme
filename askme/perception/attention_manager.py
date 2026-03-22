"""AttentionManager — decides what scene changes are worth acting on.

Prevents alert fatigue by enforcing per-event-type cooldowns and
importance thresholds. Separates "worth a TTS alert" from
"worth spending VLM compute on".
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from askme.schemas.events import ChangeEvent, ChangeEventType

logger = logging.getLogger(__name__)

# Default cooldown seconds per event type — prevents spam for repeated events
_DEFAULT_COOLDOWNS: dict[ChangeEventType, float] = {
    ChangeEventType.PERSON_APPEARED: 10.0,
    ChangeEventType.PERSON_LEFT: 15.0,
    ChangeEventType.OBJECT_APPEARED: 30.0,
    ChangeEventType.OBJECT_DISAPPEARED: 30.0,
    ChangeEventType.COUNT_CHANGED: 20.0,
}

# Importance threshold: events below this never trigger alerts
_DEFAULT_ALERT_THRESHOLD: float = 0.5

# Importance threshold for escalating to VLM scene analysis
_DEFAULT_INVESTIGATE_THRESHOLD: float = 0.7


@dataclass
class AttentionConfig:
    """Configures cooldowns and thresholds for AttentionManager."""

    cooldowns: dict[ChangeEventType, float] = field(
        default_factory=lambda: dict(_DEFAULT_COOLDOWNS)
    )
    alert_threshold: float = _DEFAULT_ALERT_THRESHOLD
    investigate_threshold: float = _DEFAULT_INVESTIGATE_THRESHOLD

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AttentionConfig:
        cfg = (config or {}).get("proactive", {}).get("attention", {})
        cooldowns = dict(_DEFAULT_COOLDOWNS)
        for key, val in cfg.get("cooldowns", {}).items():
            try:
                et = ChangeEventType(key)
                cooldowns[et] = float(val)
            except ValueError:
                logger.warning("[AttentionManager] Unknown event type in cooldowns: %s", key)
        return cls(
            cooldowns=cooldowns,
            alert_threshold=float(cfg.get("alert_threshold", _DEFAULT_ALERT_THRESHOLD)),
            investigate_threshold=float(
                cfg.get("investigate_threshold", _DEFAULT_INVESTIGATE_THRESHOLD)
            ),
        )


class AttentionManager:
    """Decides which events warrant TTS alerts or VLM investigation.

    Uses per-event-type cooldowns and importance thresholds to
    suppress low-value noise while surfacing high-priority changes.

    Usage::

        manager = AttentionManager()
        if manager.should_alert(event):
            tts.speak(event.description_zh())
        if manager.should_investigate(event):
            vlm.describe_scene()
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._cfg = AttentionConfig.from_config(config or {})
        # event_type → last alert timestamp
        self._last_alert: dict[ChangeEventType, float] = {}
        # event_type → last investigate timestamp
        self._last_investigate: dict[ChangeEventType, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_alert(self, event: ChangeEvent) -> bool:
        """Return True if this event warrants a TTS alert.

        Criteria:
        - importance >= alert_threshold
        - cooldown for this event type has expired
        """
        if event.importance < self._cfg.alert_threshold:
            logger.debug(
                "[AttentionManager] Alert suppressed (importance %.2f < %.2f): %s",
                event.importance, self._cfg.alert_threshold, event.event_type,
            )
            return False

        now = time.time()
        last = self._last_alert.get(event.event_type, 0.0)
        cooldown = self._cfg.cooldowns.get(event.event_type, 30.0)

        if now - last < cooldown:
            logger.debug(
                "[AttentionManager] Alert on cooldown (%.1fs remaining): %s",
                cooldown - (now - last), event.event_type,
            )
            return False

        self._last_alert[event.event_type] = now
        logger.info(
            "[AttentionManager] Alert triggered: %s (importance=%.2f)",
            event.event_type, event.importance,
        )
        return True

    def should_investigate(self, event: ChangeEvent) -> bool:
        """Return True if this event warrants VLM scene analysis.

        Criteria:
        - importance >= investigate_threshold
        - investigate cooldown (2x alert cooldown) has expired
        """
        if event.importance < self._cfg.investigate_threshold:
            return False

        now = time.time()
        last = self._last_investigate.get(event.event_type, 0.0)
        cooldown = self._cfg.cooldowns.get(event.event_type, 30.0) * 2

        if now - last < cooldown:
            logger.debug(
                "[AttentionManager] Investigate on cooldown (%.1fs remaining): %s",
                cooldown - (now - last), event.event_type,
            )
            return False

        self._last_investigate[event.event_type] = now
        logger.info(
            "[AttentionManager] Investigate triggered: %s (importance=%.2f)",
            event.event_type, event.importance,
        )
        return True

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_cooldown(self, event_type: ChangeEventType) -> None:
        """Force-clear cooldown for an event type (e.g. after state change)."""
        self._last_alert.pop(event_type, None)
        self._last_investigate.pop(event_type, None)

    def reset_all_cooldowns(self) -> None:
        """Clear all cooldowns."""
        self._last_alert.clear()
        self._last_investigate.clear()

    def cooldown_remaining(self, event_type: ChangeEventType) -> float:
        """Return seconds until next alert is allowed (0.0 if ready)."""
        now = time.time()
        last = self._last_alert.get(event_type, 0.0)
        cooldown = self._cfg.cooldowns.get(event_type, 30.0)
        return max(0.0, cooldown - (now - last))

    def status(self) -> dict[str, Any]:
        """Return current cooldown status for diagnostics."""
        now = time.time()
        return {
            "alert_threshold": self._cfg.alert_threshold,
            "investigate_threshold": self._cfg.investigate_threshold,
            "cooldowns_remaining": {
                et.value: round(max(0.0, cd - (now - self._last_alert.get(et, 0.0))), 1)
                for et, cd in self._cfg.cooldowns.items()
            },
        }
