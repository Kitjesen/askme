"""Turn-scoped cancellation independent from the sticky safety E-STOP."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar
from uuid import uuid4

from askme.pipeline.core.protocols import CancellationToken

_T = TypeVar("_T")


class AtomicCancellationToken:
    """Thread-safe cancellation token with an atomic work-handoff seam."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._event = threading.Event()

    def is_set(self) -> bool:
        with self._lock:
            return self._event.is_set()

    def set(self) -> None:
        with self._lock:
            self._event.set()

    def try_run(self, callback: Callable[[], _T]) -> tuple[bool, _T | None]:
        """Run a synchronous handoff iff cancellation has not linearized."""

        with self._lock:
            if self._event.is_set():
                return False, None
            return True, callback()


@dataclass(frozen=True, slots=True)
class TurnLease:
    """Ownership token for one voice turn."""

    turn_id: str
    epoch: int
    cancel_event: CancellationToken
    started_ns: int
    _controller: TurnCancellationController = field(repr=False, compare=False)

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def is_set(self) -> bool:
        return self.cancel_event.is_set()

    def set(self) -> None:
        self.cancel_event.set()

    def try_run(self, callback: Callable[[], _T]) -> tuple[bool, _T | None]:
        """Atomically settle history or hand work off while this lease owns turn."""

        return self._controller.try_run(self, callback)


class TurnCancellationController:
    """Own the single active voice turn and monotonically increasing epoch.

    The controller never owns or clears the E-STOP token. Callers can cancel
    an answer on barge-in without altering the robot's sticky safety state.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._epoch = 0
        self._active: TurnLease | None = None
        self._last_cancel_reason: str | None = None

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    @property
    def active(self) -> TurnLease | None:
        with self._lock:
            return self._active

    @property
    def last_cancel_reason(self) -> str | None:
        with self._lock:
            return self._last_cancel_reason

    def begin(
        self,
        turn_id: str | None = None,
        *,
        cancel_event: CancellationToken | None = None,
    ) -> TurnLease:
        """Start one turn, superseding any unfinished prior turn."""

        normalized_turn_id = str(turn_id or "").strip() or uuid4().hex
        with self._lock:
            if self._active is not None and not self._active.cancelled:
                self._active.cancel_event.set()
                self._last_cancel_reason = "superseded"
            self._epoch += 1
            lease = TurnLease(
                turn_id=normalized_turn_id,
                epoch=self._epoch,
                cancel_event=(cancel_event if cancel_event is not None else threading.Event()),
                started_ns=time.monotonic_ns(),
                _controller=self,
            )
            self._active = lease
            return lease

    def try_run(
        self,
        lease: TurnLease,
        callback: Callable[[], _T],
    ) -> tuple[bool, _T | None]:
        """Linearize one synchronous handoff against active-turn cancellation."""

        with self._lock:
            if self._active is not lease or lease.cancelled:
                return False, None
            atomic_runner = getattr(lease.cancel_event, "try_run", None)
            if callable(atomic_runner):
                return atomic_runner(callback)
            return True, callback()

    def cancel_active(self, *, reason: str) -> bool:
        """Cancel only the active turn and invalidate its media epoch."""

        with self._lock:
            lease = self._active
            if lease is None or lease.cancelled:
                return False
            lease.cancel_event.set()
            self._last_cancel_reason = str(reason or "cancelled")
            self._epoch += 1
            return True

    def finish(self, lease: TurnLease) -> None:
        """Release ownership only when *lease* is still the current turn."""

        with self._lock:
            if self._active is lease:
                self._active = None
