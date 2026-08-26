"""Candidate-silence controller for safe deterministic voice intents."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from askme.voice.interaction import (
    FastVoiceIntent,
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
        stable_partial_ms: float = 160.0,
    ) -> None:
        self._quick_replies = quick_replies
        self._enabled = bool(enabled)
        self._candidate_silence_ms = max(100.0, float(candidate_silence_ms))
        self._stable_partial_ms = max(0.0, float(stable_partial_ms))
        self.reset()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def reset(self) -> None:
        self._silence_started_at: float | None = None
        self._stable_text_started_at: float | None = None
        self._last_text = ""
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
            self._last_text = ""
            self._stable_text_started_at = None
            if not quiet:
                self._silence_started_at = None
            return FastEndpointDecision(FastEndpointAction.WAIT)

        if intent.normalized_text != self._last_text:
            self._last_text = intent.normalized_text
            self._stable_text_started_at = current

        if not quiet:
            self._silence_started_at = None
            return FastEndpointDecision(FastEndpointAction.WAIT, intent=intent)

        if self._silence_started_at is None:
            self._silence_started_at = current

        silence_ms = (current - self._silence_started_at) * 1000.0
        stable_since = self._stable_text_started_at or current
        stable_text_ms = (current - stable_since) * 1000.0
        if (
            silence_ms < self._candidate_silence_ms
            or stable_text_ms < self._stable_partial_ms
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
