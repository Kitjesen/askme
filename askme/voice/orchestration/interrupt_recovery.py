"""Generation-safe playback recovery while a possible interruption is validated.

This module owns no conversation or cancellation truth.  It only coordinates a
lossless playback hold while ASR decides whether detected speech is actionable.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Protocol

_SAFE_REASON = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,63}$")


class PlaybackHoldPort(Protocol):
    """Minimal output capability needed by interruption recovery."""

    def pause_playback(self, *, timeout_s: float) -> object | None: ...

    def resume_playback(self, token: object) -> bool: ...

    def abort_playback_hold(self, token: object) -> bool: ...


class InterruptionRecoveryState(StrEnum):
    """Observable validation state; an active token is reported separately."""

    IDLE = "idle"
    DETECTED = "detected"
    VALIDATING = "validating"


class InterruptionRecoveryCoordinator:
    """Linearize hold, validation, recovery, and commit for one audio agent."""

    def __init__(
        self,
        playback: PlaybackHoldPort,
        *,
        pause_timeout_s: float = 0.05,
        hold_timeout_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(pause_timeout_s) or pause_timeout_s < 0:
            raise ValueError("pause_timeout_s must be finite and non-negative")
        if not math.isfinite(hold_timeout_s) or hold_timeout_s <= 0:
            raise ValueError("hold_timeout_s must be finite and positive")
        self._playback = playback
        self._pause_timeout_s = float(pause_timeout_s)
        self._hold_timeout_s = float(hold_timeout_s)
        self._clock = clock
        self._lock = threading.RLock()
        self._state = InterruptionRecoveryState.IDLE
        self._token: object | None = None
        self._hold_started_at: float | None = None
        self._hold_supported: bool | None = None
        self._detections = 0
        self._confirmations = 0
        self._commits = 0
        self._recoveries = 0
        self._hold_timeouts = 0
        self._resume_failures = 0
        self._abort_failures = 0
        self._last_reason = "startup"

    @property
    def state(self) -> InterruptionRecoveryState:
        with self._lock:
            return self._state

    def begin_detection(self) -> bool:
        """Attempt one lossless hold; duplicate VAD notifications are idempotent."""

        with self._lock:
            if self._state is not InterruptionRecoveryState.IDLE:
                return self._token is not None
            self._detections += 1
            self._state = InterruptionRecoveryState.DETECTED
            self._last_reason = "speech_detected"
            try:
                token = self._playback.pause_playback(
                    timeout_s=self._pause_timeout_s,
                )
            except Exception:
                token = None
                self._last_reason = "pause_failed"
            self._token = token
            self._hold_supported = token is not None
            self._hold_started_at = self._safe_now() if token is not None else None
            return token is not None

    def confirm(self) -> None:
        """Enter ASR validation without cancelling playback or pipeline work."""

        with self._lock:
            if self._state is InterruptionRecoveryState.IDLE:
                self.begin_detection()
            if self._state is InterruptionRecoveryState.VALIDATING:
                return
            self._state = InterruptionRecoveryState.VALIDATING
            self._confirmations += 1
            self._last_reason = "speech_confirmed"

    def recover(self, reason: str) -> bool:
        """Resume the exact held generation after rejected/false speech."""

        with self._lock:
            if self._state is InterruptionRecoveryState.IDLE:
                return False
            token = self._clear_active_state(reason)
            self._recoveries += 1
            if token is not None:
                try:
                    resumed = bool(self._playback.resume_playback(token))
                except Exception:
                    resumed = False
                if not resumed:
                    self._resume_failures += 1
            return True

    def commit(self, reason: str) -> bool:
        """Discard the hold only after ASR has admitted a real interruption."""

        with self._lock:
            if self._state is InterruptionRecoveryState.IDLE:
                return False
            token = self._clear_active_state(reason)
            self._commits += 1
            if token is not None:
                try:
                    aborted = bool(self._playback.abort_playback_hold(token))
                except Exception:
                    aborted = False
                if not aborted:
                    self._abort_failures += 1
            return True

    def expire_hold(self) -> bool:
        """Bound silence while leaving the ASR validation decision open."""

        with self._lock:
            token = self._token
            started_at = self._hold_started_at
            if token is None or started_at is None:
                return False
            if self._safe_now() - started_at < self._hold_timeout_s:
                return False
            self._token = None
            self._hold_started_at = None
            self._hold_timeouts += 1
            self._last_reason = "hold_timeout"
            try:
                resumed = bool(self._playback.resume_playback(token))
            except Exception:
                resumed = False
            if not resumed:
                self._resume_failures += 1
            return True

    def close(self) -> bool:
        """Abort a live hold during lifecycle teardown without counting a commit."""

        with self._lock:
            if self._state is InterruptionRecoveryState.IDLE:
                return False
            token = self._clear_active_state("closed")
            if token is not None:
                try:
                    aborted = bool(self._playback.abort_playback_hold(token))
                except Exception:
                    aborted = False
                if not aborted:
                    self._abort_failures += 1
            return True

    def status_snapshot(self) -> dict[str, object]:
        """Return bounded, privacy-safe operational evidence."""

        with self._lock:
            return {
                "state": self._state.value,
                "hold_active": self._token is not None,
                "hold_supported": self._hold_supported,
                "detections": self._detections,
                "confirmations": self._confirmations,
                "commits": self._commits,
                "recoveries": self._recoveries,
                "hold_timeouts": self._hold_timeouts,
                "resume_failures": self._resume_failures,
                "abort_failures": self._abort_failures,
                "last_reason": self._last_reason,
            }

    def _clear_active_state(self, reason: str) -> object | None:
        token = self._token
        self._token = None
        self._hold_started_at = None
        self._state = InterruptionRecoveryState.IDLE
        self._last_reason = _safe_reason(reason)
        return token

    def _safe_now(self) -> float:
        try:
            value = float(self._clock())
        except (TypeError, ValueError, OverflowError):
            return 0.0
        return value if math.isfinite(value) else 0.0


def _safe_reason(reason: str) -> str:
    candidate = str(reason or "").strip().lower()
    return candidate if _SAFE_REASON.fullmatch(candidate) else "unspecified"
