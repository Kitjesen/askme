"""Turn-scoped cancellation independent from the sticky safety E-STOP."""

from __future__ import annotations

import asyncio
import contextvars
import itertools
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar
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


@dataclass(eq=False, slots=True)
class TurnLease:
    """Ownership and cancellation token for one conversational turn."""

    turn_id: str
    epoch: int
    cancel_event: CancellationToken
    started_ns: int
    _controller: TurnCancellationController = field(repr=False, compare=False)
    sequence: int = 0
    owner: str = "generic"
    loop: asyncio.AbstractEventLoop | None = field(default=None, repr=False)
    task: asyncio.Task[Any] | None = field(default=None, repr=False)
    _reason: str | None = None
    _context_token: contextvars.Token[TurnLease | None] | None = field(
        default=None,
        repr=False,
    )

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def is_set(self) -> bool:
        return self.cancel_event.is_set()

    def set(self) -> None:
        self.cancel_event.set()

    def try_run(self, callback: Callable[[], _T]) -> tuple[bool, _T | None]:
        """Atomically settle history or hand work off while this lease owns turn."""

        return self._controller.try_run(self, callback)


class _CombinedCancellationToken:
    """Stable sticky-E-STOP plus context-local turn cancellation view."""

    def __init__(self, controller: TurnCancellationController) -> None:
        self._controller = controller

    def is_set(self) -> bool:
        return self._controller.is_cancelled_for_current_context()


class TurnCancellationController:
    """Own cancellation linearization and cooperative task interruption.

    ``begin``/``cancel_active`` are the canonical single-voice-turn API.  The
    ``begin_turn`` compatibility API additionally binds the lease to the
    current asyncio task so older text/runtime callers can wake a blocked
    await without latching the emergency-stop token.
    """

    def __init__(self, emergency_token: CancellationToken | None = None) -> None:
        self._emergency_token = emergency_token
        self._lock = threading.RLock()
        self._epoch = 0
        self._sequence = itertools.count(1)
        self._active: TurnLease | None = None
        self._leases: dict[int, TurnLease] = {}
        self._last_cancel_reason: str | None = None
        self._current_lease: contextvars.ContextVar[TurnLease | None] = (
            contextvars.ContextVar(
                f"askme_turn_cancel_lease_{id(self)}",
                default=None,
            )
        )
        self._token = _CombinedCancellationToken(self)

    @property
    def token(self) -> _CombinedCancellationToken:
        return self._token

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
        """Start one canonical voice turn, superseding an unfinished prior turn."""

        normalized_turn_id = str(turn_id or "").strip() or uuid4().hex
        with self._lock:
            if self._active is not None and not self._active.cancelled:
                self._active.cancel_event.set()
                self._active._reason = "superseded"
                self._last_cancel_reason = "superseded"
            self._epoch += 1
            lease = TurnLease(
                turn_id=normalized_turn_id,
                epoch=self._epoch,
                cancel_event=(
                    cancel_event if cancel_event is not None else AtomicCancellationToken()
                ),
                started_ns=time.monotonic_ns(),
                _controller=self,
            )
            self._active = lease
            return lease

    def begin_turn(self, owner: str = "generic") -> TurnLease:
        """Bind a cooperative cancellation lease to the current asyncio task."""

        loop = asyncio.get_running_loop()
        task = asyncio.current_task(loop=loop)
        if task is None:
            raise RuntimeError("begin_turn must run inside an asyncio task")
        with self._lock:
            sequence = next(self._sequence)
            lease = TurnLease(
                turn_id=uuid4().hex,
                epoch=self._epoch,
                cancel_event=AtomicCancellationToken(),
                started_ns=time.monotonic_ns(),
                _controller=self,
                sequence=sequence,
                owner=self._normalize_owner(owner),
                loop=loop,
                task=task,
            )
            self._leases[sequence] = lease
        lease._context_token = self._current_lease.set(lease)
        return lease

    def try_run(
        self,
        lease: TurnLease,
        callback: Callable[[], _T],
    ) -> tuple[bool, _T | None]:
        """Linearize one synchronous handoff against active-turn cancellation."""

        with self._lock:
            canonical = self._active is lease
            compatible = bool(
                lease.sequence and self._leases.get(lease.sequence) is lease
            )
            if (not canonical and not compatible) or lease.cancelled:
                return False, None
            atomic_runner = getattr(lease.cancel_event, "try_run", None)
            if callable(atomic_runner):
                return atomic_runner(callback)
            return True, callback()

    def cancel_active(self, *, reason: str) -> bool:
        """Cancel only the canonical active turn and invalidate its media epoch."""

        with self._lock:
            lease = self._active
            if lease is None or lease.cancelled:
                return False
            lease._reason = str(reason or "cancelled")
            lease.cancel_event.set()
            self._last_cancel_reason = lease._reason
            self._epoch += 1
            return True

    def cancel_current_turn(
        self,
        reason: str = "barge_in",
        *,
        owner: str | None = None,
    ) -> bool:
        """Cancel the latest compatible task lease, optionally by owner."""

        normalized_reason = str(reason or "barge_in").strip() or "barge_in"
        with self._lock:
            lease = self._latest_lease_locked(owner)
            if lease is None:
                return False
            lease._reason = normalized_reason
            lease.cancel_event.set()
            self._last_cancel_reason = normalized_reason
        if lease.loop is not None:
            try:
                lease.loop.call_soon_threadsafe(self._cancel_if_active, lease)
            except RuntimeError:
                pass
        return True

    def finish(self, lease: TurnLease) -> None:
        """Release canonical ownership only when *lease* is still active."""

        with self._lock:
            if self._active is lease:
                self._active = None

    def end_turn(self, lease: TurnLease) -> None:
        """Release a compatibility task lease without altering sticky E-STOP."""

        with self._lock:
            self._leases.pop(lease.sequence, None)
        context_token = lease._context_token
        lease._context_token = None
        if context_token is not None:
            self._current_lease.reset(context_token)

    def is_cancelled_for_current_context(self) -> bool:
        if self._emergency_token is not None and self._emergency_token.is_set():
            return True
        lease = self._current_lease.get()
        return lease.cancelled if lease is not None else False

    def _cancel_if_active(self, lease: TurnLease) -> None:
        with self._lock:
            active = self._leases.get(lease.sequence) is lease
        if active and lease.task is not None and not lease.task.done():
            lease.task.cancel(lease.reason or "barge_in")

    def _latest_lease_locked(self, owner: str | None) -> TurnLease | None:
        if owner is None:
            return next(reversed(self._leases.values()), None)
        normalized_owner = self._normalize_owner(owner)
        return next(
            (
                lease
                for lease in reversed(self._leases.values())
                if lease.owner == normalized_owner
            ),
            None,
        )

    @staticmethod
    def _normalize_owner(owner: str) -> str:
        return str(owner or "generic").strip().lower() or "generic"
