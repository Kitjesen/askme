"""Thread-safe cancellation control for one conversational turn.

The controller keeps emergency stop and conversational interruption separate:
the injected emergency token is sticky until an operator resets it, while every
``begin_turn`` call creates a fresh lease that is discarded by ``end_turn``.
Callers pass the stable ``token`` view to pipeline modules; context-local lease
lookup makes concurrent turns observe only their own interruption.
"""

from __future__ import annotations

import asyncio
import contextvars
import itertools
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass(eq=False, slots=True)
class TurnLease:
    """Handle for one invocation of the conversational pipeline."""

    sequence: int
    owner: str
    loop: asyncio.AbstractEventLoop
    task: asyncio.Task[Any]
    _cancelled: threading.Event = field(default_factory=threading.Event)
    _reason: str | None = None
    _context_token: contextvars.Token[TurnLease | None] | None = None

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason


class _CombinedCancellationToken:
    """Stable token view shared by StreamProcessor and TurnExecutor."""

    __slots__ = ("_controller",)

    def __init__(self, controller: TurnCancellationController) -> None:
        self._controller = controller

    def is_set(self) -> bool:
        return self._controller.is_cancelled_for_current_context()


class TurnCancellationController:
    """Own global E-STOP state and short-lived per-turn cancellation leases.

    ``cancel_current_turn`` is deliberately synchronous so microphone/audio
    worker threads can call it directly. It sets a thread-safe lease flag and
    schedules ``Task.cancel`` through ``loop.call_soon_threadsafe`` to wake an
    in-flight await immediately.
    """

    def __init__(self, emergency_token: asyncio.Event) -> None:
        self._emergency_token = emergency_token
        self._lock = threading.RLock()
        self._sequence = itertools.count(1)
        self._leases: dict[int, TurnLease] = {}
        self._current_lease: contextvars.ContextVar[TurnLease | None] = (
            contextvars.ContextVar(
                f"askme_turn_cancel_lease_{id(self)}",
                default=None,
            )
        )
        self._token = _CombinedCancellationToken(self)

    @property
    def token(self) -> _CombinedCancellationToken:
        """Return the stable cancellation view supplied to pipeline modules."""

        return self._token

    def begin_turn(self, owner: str = "generic") -> TurnLease:
        """Create and bind a fresh cancellation lease to the current task."""

        loop = asyncio.get_running_loop()
        task = asyncio.current_task(loop=loop)
        if task is None:
            raise RuntimeError("begin_turn must run inside an asyncio task")

        with self._lock:
            lease = TurnLease(
                sequence=next(self._sequence),
                owner=self._normalize_owner(owner),
                loop=loop,
                task=task,
            )
            self._leases[lease.sequence] = lease
        lease._context_token = self._current_lease.set(lease)
        return lease

    def end_turn(self, lease: TurnLease) -> None:
        """Release *lease* without changing sticky emergency-stop state."""

        with self._lock:
            self._leases.pop(lease.sequence, None)
        context_token = lease._context_token
        lease._context_token = None
        if context_token is not None:
            self._current_lease.reset(context_token)

    def cancel_current_turn(
        self,
        reason: str = "barge_in",
        *,
        owner: str | None = None,
    ) -> bool:
        """Cancel the latest active turn, optionally scoped to one owner.

        Returns ``False`` when no turn is active. The emergency token is never
        modified, so a conversational interruption cannot latch E-STOP. An
        omitted owner preserves the legacy behavior of selecting the latest
        lease across all channels.
        """

        normalized_reason = str(reason or "barge_in").strip() or "barge_in"
        with self._lock:
            lease = self._latest_lease_locked(owner=owner)
            if lease is None:
                return False
            lease._reason = normalized_reason
            lease._cancelled.set()

        try:
            lease.loop.call_soon_threadsafe(self._cancel_if_active, lease)
        except RuntimeError:
            # A concurrently closing event loop has already made the turn
            # impossible to continue; the lease flag still records cancellation.
            pass
        return True

    def is_cancelled_for_current_context(self) -> bool:
        """Return E-STOP or the calling context's own turn cancellation state."""

        if self._emergency_token.is_set():
            return True
        lease = self._current_lease.get()
        return lease.cancelled if lease is not None else False

    def _cancel_if_active(self, lease: TurnLease) -> None:
        """Event-loop callback that avoids cancelling a task after lease release."""

        with self._lock:
            active = self._leases.get(lease.sequence) is lease
        if active and not lease.task.done():
            lease.task.cancel(lease.reason or "barge_in")

    def _latest_lease_locked(self, owner: str | None = None) -> TurnLease | None:
        if not self._leases:
            return None
        if owner is None:
            return next(reversed(self._leases.values()))
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
