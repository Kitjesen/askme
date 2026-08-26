"""Candidate-silence controller for safe deterministic voice intents."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from askme.robot_interaction.routing.fast_voice_intents import (
    FastVoiceIntent,
    FastVoiceIntentKind,
    match_fast_voice_intent,
)


class FastEndpointAction(Enum):
    WAIT = "wait"
    COMMIT = "commit"


@dataclass(frozen=True)
class FastEndpointDecision:
    action: FastEndpointAction
    intent: FastVoiceIntent | None = None
    silence_ms: float = 0.0
    stable_text_ms: float = 0.0


class FastEndpointController:
    """Admit early endpoints only after quiet and transcript stability."""

    def __init__(
        self,
        *,
        quick_replies: Mapping[str, str],
        enabled: bool = False,
        candidate_silence_ms: float = 300.0,
        estop_candidate_silence_ms: float = 150.0,
        stable_partial_ms: float = 160.0,
    ) -> None:
        self._quick_replies = quick_replies
        self._enabled = bool(enabled)
        self._candidate_silence_ms = max(100.0, float(candidate_silence_ms))
        self._estop_candidate_silence_ms = min(
            self._candidate_silence_ms,
            max(100.0, float(estop_candidate_silence_ms)),
        )
        self._stable_partial_ms = max(0.0, float(stable_partial_ms))
        self.reset()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def reset(self) -> None:
        self._silence_started_at: float | None = None
        self._stable_text_started_at: float | None = None
        self._last_candidate_identity: tuple[str, FastVoiceIntentKind, str] | None = None
        self._candidate_observations = 0
        self._committed = False

    def observe(
        self,
        *,
        partial_text: str,
        quiet: bool,
        now: float | None = None,
    ) -> FastEndpointDecision:
        current = time.monotonic() if now is None else float(now)
        if not self._enabled or self._committed:
            return FastEndpointDecision(FastEndpointAction.WAIT)

        intent = match_fast_voice_intent(
            partial_text,
            quick_replies=self._quick_replies,
        )
        if intent is None:
            self._last_candidate_identity = None
            self._candidate_observations = 0
            self._stable_text_started_at = None
            self._silence_started_at = None
            return FastEndpointDecision(FastEndpointAction.WAIT)

        candidate_identity = (
            intent.intent_id,
            intent.kind,
            intent.normalized_text,
        )
        if candidate_identity != self._last_candidate_identity:
            self._last_candidate_identity = candidate_identity
            self._candidate_observations = 0
            self._stable_text_started_at = current
            self._silence_started_at = None
        self._candidate_observations += 1

        if not quiet:
            self._silence_started_at = None
            return FastEndpointDecision(FastEndpointAction.WAIT, intent=intent)

        if self._silence_started_at is None:
            self._silence_started_at = current

        silence_ms = (current - self._silence_started_at) * 1000.0
        stable_since = (
            self._stable_text_started_at
            if self._stable_text_started_at is not None
            else current
        )
        stable_text_ms = (current - stable_since) * 1000.0
        required_silence_ms = (
            self._estop_candidate_silence_ms
            if intent.kind is FastVoiceIntentKind.ESTOP
            else self._candidate_silence_ms
        )
        if (
            silence_ms < required_silence_ms
            or stable_text_ms < self._stable_partial_ms
            or (
                intent.kind is FastVoiceIntentKind.ESTOP
                and self._candidate_observations < 2
            )
        ):
            return FastEndpointDecision(
                FastEndpointAction.WAIT,
                intent=intent,
                silence_ms=silence_ms,
                stable_text_ms=stable_text_ms,
            )

        self._committed = True
        return FastEndpointDecision(
            FastEndpointAction.COMMIT,
            intent=intent,
            silence_ms=silence_ms,
            stable_text_ms=stable_text_ms,
        )


__all__ = [
    "FastEndpointAction",
    "FastEndpointController",
    "FastEndpointDecision",
]
