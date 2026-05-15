"""Execution control primitives for tool scheduling.

This module is intentionally independent from concrete tool classes.  The
registry owns policy and argument validation; the scheduler owns bounded
priority execution, cancellation-before-start, and lightweight diagnostics.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any


class ToolQueueFullError(RuntimeError):
    """Raised when a tool execution cannot be queued."""


@dataclass(order=True)
class ScheduledWork:
    """One queued callable with stable priority ordering."""

    priority: int
    sequence: int
    work_id: str = field(compare=False)
    fn: Callable[[], Any] = field(compare=False)
    future: Future = field(compare=False)
    submitted_at: float = field(compare=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(order=True)
class _StopWork:
    priority: int = field(default=10**9)
    sequence: int = field(default=10**9)


class ToolExecutionScheduler:
    """Bounded priority queue backed by worker threads."""

    def __init__(
        self,
        *,
        max_workers: int,
        max_queue_size: int,
        thread_name_prefix: str = "askme-tool",
        on_start: Callable[[ScheduledWork], None] | None = None,
        on_finish: Callable[[ScheduledWork, str, Any, BaseException | None, float], None]
        | None = None,
    ) -> None:
        self._max_workers = max(1, int(max_workers))
        self._queue: queue.PriorityQueue[ScheduledWork | _StopWork] = (
            queue.PriorityQueue(maxsize=max(1, int(max_queue_size)))
        )
        self._thread_name_prefix = thread_name_prefix
        self._on_start = on_start
        self._on_finish = on_finish
        self._lock = threading.RLock()
        self._closed = False
        self._sequence = 0
        self._futures_by_id: dict[str, Future] = {}
        self._running: set[str] = set()
        self._completed_count = 0
        self._threads = [
            threading.Thread(
                target=self._worker,
                name=f"{self._thread_name_prefix}-{index}",
                daemon=True,
            )
            for index in range(self._max_workers)
        ]
        for thread in self._threads:
            thread.start()

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def submit(
        self,
        fn: Callable[[], Any],
        *,
        priority: int,
        work_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Future:
        with self._lock:
            if self._closed:
                raise RuntimeError("tool execution scheduler is shut down")
            future: Future = Future()
            self._sequence += 1
            work = ScheduledWork(
                priority=int(priority),
                sequence=self._sequence,
                work_id=work_id,
                fn=fn,
                future=future,
                submitted_at=time.monotonic(),
                metadata=dict(metadata or {}),
            )
            try:
                self._queue.put_nowait(work)
            except queue.Full as exc:
                raise ToolQueueFullError("tool execution queue is full") from exc
            self._futures_by_id[work_id] = future
            return future

    def cancel(self, work_id: str) -> bool:
        with self._lock:
            future = self._futures_by_id.get(work_id)
        if future is None:
            return False
        return future.cancel()

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            futures = list(self._futures_by_id.values())
        if cancel_futures:
            for future in futures:
                future.cancel()
            self._drain_cancelled_queue()
        for _ in self._threads:
            while True:
                try:
                    self._queue.put(_StopWork(), timeout=0.1)
                    break
                except queue.Full:
                    if not wait:
                        break
        if wait:
            for thread in self._threads:
                thread.join(timeout=2.0)

    def diagnostics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "max_workers": self._max_workers,
                "queued": self._queue.qsize(),
                "running": len(self._running),
                "tracked": len(self._futures_by_id),
                "completed": self._completed_count,
                "closed": self._closed,
            }

    def _drain_cancelled_queue(self) -> None:
        while True:
            try:
                work = self._queue.get_nowait()
            except queue.Empty:
                return
            try:
                if isinstance(work, ScheduledWork):
                    work.future.cancel()
                    self._finish_work(work, "cancelled", None, None, time.perf_counter())
            finally:
                self._queue.task_done()

    def _worker(self) -> None:
        while True:
            work = self._queue.get()
            try:
                if isinstance(work, _StopWork):
                    return
                self._run_work(work)
            finally:
                self._queue.task_done()

    def _run_work(self, work: ScheduledWork) -> None:
        started = time.perf_counter()
        if not work.future.set_running_or_notify_cancel():
            self._finish_work(work, "cancelled", None, None, started)
            return
        with self._lock:
            self._running.add(work.work_id)
        if self._on_start is not None:
            self._on_start(work)
        try:
            result = work.fn()
        except BaseException as exc:  # noqa: BLE001 - surfaced through Future
            if not work.future.done():
                work.future.set_exception(exc)
            self._finish_work(work, "failed", None, exc, started)
        else:
            if not work.future.done():
                work.future.set_result(result)
            self._finish_work(work, "succeeded", result, None, started)

    def _finish_work(
        self,
        work: ScheduledWork,
        status: str,
        result: Any,
        error: BaseException | None,
        started: float,
    ) -> None:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        with self._lock:
            self._running.discard(work.work_id)
            self._futures_by_id.pop(work.work_id, None)
            self._completed_count += 1
        if self._on_finish is not None:
            self._on_finish(work, status, result, error, elapsed_ms)


class WindowRateLimiter:
    """Simple per-key fixed-window limiter."""

    def __init__(self, *, window_seconds: float = 60.0) -> None:
        self._window_seconds = max(0.001, float(window_seconds))
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.RLock()

    def check_and_consume(self, key: str, limit: float) -> tuple[bool, float]:
        if limit <= 0:
            return True, 0.0
        now = time.monotonic()
        with self._lock:
            events = self._events.setdefault(key, deque())
            cutoff = now - self._window_seconds
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= int(limit):
                retry_after = self._window_seconds - (now - events[0])
                return False, max(0.0, retry_after)
            events.append(now)
            return True, 0.0

    def diagnostics(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            active = {}
            for key, events in self._events.items():
                cutoff = now - self._window_seconds
                active[key] = sum(1 for item in events if item > cutoff)
            return {"window_seconds": self._window_seconds, "active": active}


class CircuitBreaker:
    """Per-tool consecutive-failure circuit breaker."""

    def __init__(self, *, failure_threshold: int, cooldown_seconds: float) -> None:
        self._failure_threshold = max(0, int(failure_threshold))
        self._cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, float] = {}
        self._lock = threading.RLock()

    def remaining_open_seconds(self, key: str) -> float:
        with self._lock:
            until = self._open_until.get(key)
            if until is None:
                return 0.0
            remaining = until - time.monotonic()
            if remaining <= 0:
                self._open_until.pop(key, None)
                self._failures.pop(key, None)
                return 0.0
            return remaining

    def record_success(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)
            self._open_until.pop(key, None)

    def record_failure(self, key: str) -> None:
        if self._failure_threshold <= 0 or self._cooldown_seconds <= 0:
            return
        with self._lock:
            failures = self._failures.get(key, 0) + 1
            self._failures[key] = failures
            if failures >= self._failure_threshold:
                self._open_until[key] = time.monotonic() + self._cooldown_seconds

    def diagnostics(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            open_remaining = {
                key: round(until - now, 3)
                for key, until in self._open_until.items()
                if until > now
            }
            return {
                "failure_threshold": self._failure_threshold,
                "cooldown_seconds": self._cooldown_seconds,
                "failure_counts": dict(self._failures),
                "open": open_remaining,
            }
