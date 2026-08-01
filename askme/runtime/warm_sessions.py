"""Warm provider session maintenance for runtime-managed external services.

The manager owns scheduling and observability only. Provider-specific modules
adapt LLM/TTS clients behind ``WarmSessionTarget`` so the runtime can keep a
session warm across robot uptime without coupling the scheduler to a vendor SDK.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WarmSessionResult:
    """Result returned by a provider-specific warm-session adapter."""

    ok: bool
    status: str
    reason: str = ""
    elapsed_ms: float | None = None
    reused: bool = False
    neutral: bool = False
    provider_session_key: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class WarmSessionTarget(Protocol):
    """Provider adapter boundary consumed by WarmSessionManager."""

    async def ensure_warm(self, *, force_refresh: bool) -> WarmSessionResult:
        """Open or refresh the provider session."""

    def cancel(self) -> None:
        """Best-effort cancellation hook used during runtime shutdown."""


@dataclass(frozen=True)
class WarmSessionPolicy:
    """Scheduling policy for one warm provider session."""

    startup_delay_seconds: float = 0.0
    refresh_interval_seconds: float = 60.0
    timeout_seconds: float = 10.0
    initial_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 60.0
    busy_retry_seconds: float = 1.0
    jitter_ratio: float = 0.1
    max_attempts_per_hour: int = 60

    def __post_init__(self) -> None:
        if self.startup_delay_seconds < 0:
            raise ValueError("startup_delay_seconds must be >= 0")
        if self.refresh_interval_seconds <= 0:
            raise ValueError("refresh_interval_seconds must be > 0")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.initial_backoff_seconds <= 0:
            raise ValueError("initial_backoff_seconds must be > 0")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("max_backoff_seconds must be >= initial_backoff_seconds")
        if self.busy_retry_seconds <= 0:
            raise ValueError("busy_retry_seconds must be > 0")
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be between 0 and 1")
        if self.max_attempts_per_hour <= 0:
            raise ValueError("max_attempts_per_hour must be > 0")


@dataclass(frozen=True)
class WarmSessionBinding:
    """One named warm-session target plus its policy."""

    name: str
    target: WarmSessionTarget
    policy: WarmSessionPolicy = field(default_factory=WarmSessionPolicy)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("warm session binding name must not be empty")


@dataclass
class _WarmSessionState:
    status: str = "idle"
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    skips: int = 0
    consecutive_failures: int = 0
    last_status: str | None = None
    last_reason: str | None = None
    last_latency_ms: float | None = None
    last_provider_session_key: str | None = None
    active_worker_age_seconds: float | None = None
    active_worker_age_recorded_at: float | None = None
    stuck_busy_threshold_seconds: float | None = None
    last_success_at: float | None = None
    last_attempt_at: float | None = None
    next_attempt_at: float | None = None
    attempt_window_started_at: float = field(default_factory=time.monotonic)
    attempt_window_count: int = 0


class WarmSessionManager:
    """Maintain warm provider sessions for the lifetime of a running runtime.

    A manager may run from robot boot until shutdown. Individual physical
    provider sessions are refreshed or replaced according to policy, so the
    contract is "always maintained" rather than "one socket lives forever".
    """

    def __init__(
        self,
        bindings: list[WarmSessionBinding] | tuple[WarmSessionBinding, ...],
        *,
        shutdown_timeout_seconds: float = 0.5,
    ) -> None:
        names = [binding.name for binding in bindings]
        if len(names) != len(set(names)):
            raise ValueError("warm session binding names must be unique")
        if shutdown_timeout_seconds <= 0:
            raise ValueError("shutdown_timeout_seconds must be > 0")
        self._bindings = tuple(bindings)
        self._states = {binding.name: _WarmSessionState() for binding in self._bindings}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._started = False
        self._stop_event = asyncio.Event()
        self._refresh_events: dict[str, asyncio.Event] = {}
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._lifecycle_epoch = 0
        self._lifecycle_lock = asyncio.Lock()
        self._shutdown_timeout_seconds = float(shutdown_timeout_seconds)

    async def start(self) -> None:
        async with self._lifecycle_lock:
            if self._started:
                return
            self._lifecycle_epoch += 1
            lifecycle_epoch = self._lifecycle_epoch
            stop_event = asyncio.Event()
            refresh_events = {binding.name: asyncio.Event() for binding in self._bindings}
            now = time.monotonic()
            for binding in self._bindings:
                state = self._states[binding.name]
                state.status = "scheduled"
                state.consecutive_failures = 0
                state.last_status = None
                state.last_reason = None
                state.last_latency_ms = None
                state.last_provider_session_key = None
                state.active_worker_age_seconds = None
                state.active_worker_age_recorded_at = None
                state.stuck_busy_threshold_seconds = None
                state.last_success_at = None
                state.next_attempt_at = now + binding.policy.startup_delay_seconds
            self._stop_event = stop_event
            self._refresh_events = refresh_events
            self._event_loop = asyncio.get_running_loop()
            self._started = True
            for binding in self._bindings:
                self._tasks[binding.name] = asyncio.create_task(
                    self._run_binding(
                        binding,
                        lifecycle_epoch=lifecycle_epoch,
                        stop_event=stop_event,
                        refresh_event=refresh_events[binding.name],
                    ),
                    name=f"warm-session-{binding.name}",
                )

    def request_refresh(self, name: str) -> bool:
        """Wake one binding so its current provider target is refreshed promptly.

        The callback may originate from an audio worker thread after a pending
        TTS hot switch becomes active, so the event is always signalled through
        the manager's owning event loop.
        """

        refresh_event = self._refresh_events.get(name)
        loop = self._event_loop
        if not self._started or refresh_event is None or loop is None or loop.is_closed():
            return False

        def _signal() -> None:
            if not self._started or self._refresh_events.get(name) is not refresh_event:
                return
            self._states[name].next_attempt_at = time.monotonic()
            refresh_event.set()

        try:
            loop.call_soon_threadsafe(_signal)
        except RuntimeError:
            return False
        return True

    @staticmethod
    def _cancel_target(binding: WarmSessionBinding) -> None:
        cancel = getattr(binding.target, "cancel", None)
        if not callable(cancel):
            return
        try:
            cancel()
        except Exception as exc:
            logger.debug("WarmSessionManager: cancel %s failed: %s", binding.name, exc)

    async def stop(self) -> None:
        async with self._lifecycle_lock:
            if not self._started and not self._tasks:
                return
            self._started = False
            self._stop_event.set()
            tasks = tuple(self._tasks.values())
            for task in tasks:
                task.cancel()
            if tasks:
                done, pending = await asyncio.wait(
                    tasks,
                    timeout=self._shutdown_timeout_seconds,
                )
                for task in done:
                    self._consume_task_result(task)
                for task in pending:
                    task.add_done_callback(self._consume_task_result)
                if pending:
                    logger.warning(
                        "WarmSessionManager: %d task(s) exceeded %.2fs shutdown timeout",
                        len(pending),
                        self._shutdown_timeout_seconds,
                    )
            self._tasks.clear()
            self._refresh_events.clear()
            self._event_loop = None

    @staticmethod
    def _consume_task_result(task: asyncio.Task[Any]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.debug("WarmSessionManager: detached task failed: %s", exc)

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        schedulers_running = (
            self._started
            and bool(self._bindings)
            and all(
                (task := self._tasks.get(binding.name)) is not None and not task.done()
                for binding in self._bindings
            )
        )
        if schedulers_running:
            manager_status = "running"
        elif self._started:
            manager_status = "degraded"
        else:
            manager_status = "stopped"
        return {
            "status": manager_status,
            "running": schedulers_running,
            "targets": {
                binding.name: self._snapshot_state(self._states[binding.name], now, binding.policy)
                for binding in self._bindings
            },
        }

    async def _run_binding(
        self,
        binding: WarmSessionBinding,
        *,
        lifecycle_epoch: int,
        stop_event: asyncio.Event,
        refresh_event: asyncio.Event,
    ) -> None:
        state = self._states[binding.name]
        try:
            await self._sleep(
                binding.policy.startup_delay_seconds,
                stop_event,
                refresh_event,
            )
            force_refresh = True
            while self._lifecycle_epoch == lifecycle_epoch and not stop_event.is_set():
                # A refresh request received before this attempt is satisfied by
                # the attempt itself. Requests received while it is running stay
                # set and trigger one immediate follow-up for the new target.
                refresh_event.clear()
                await self._attempt_warm(
                    binding,
                    state,
                    force_refresh=force_refresh,
                    lifecycle_epoch=lifecycle_epoch,
                    stop_event=stop_event,
                )
                if self._lifecycle_epoch != lifecycle_epoch or stop_event.is_set():
                    break
                force_refresh = True
                delay = self._next_delay(binding.policy, state)
                state.next_attempt_at = time.monotonic() + delay
                await self._sleep(delay, stop_event, refresh_event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - guard long-lived task
            state.status = "error"
            state.last_reason = type(exc).__name__
            logger.warning("WarmSessionManager: %s loop stopped: %s", binding.name, exc)

    async def _attempt_warm(
        self,
        binding: WarmSessionBinding,
        state: _WarmSessionState,
        *,
        force_refresh: bool,
        lifecycle_epoch: int,
        stop_event: asyncio.Event,
    ) -> None:
        now = time.monotonic()
        if now - state.attempt_window_started_at >= 3600:
            state.attempt_window_started_at = now
            state.attempt_window_count = 0
        if state.attempt_window_count >= binding.policy.max_attempts_per_hour:
            state.skips += 1
            state.status = "throttled"
            state.last_status = "throttled"
            state.last_reason = "attempt_budget_exhausted"
            return

        state.attempts += 1
        state.attempt_window_count += 1
        state.last_attempt_at = now
        state.status = "warming"
        started = time.perf_counter()
        try:
            result = await self._ensure_warm_with_timeout(
                binding,
                force_refresh=force_refresh,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._lifecycle_epoch != lifecycle_epoch or stop_event.is_set():
                return
            state.failures += 1
            state.consecutive_failures += 1
            state.status = "degraded"
            state.last_status = "error"
            state.last_reason = type(exc).__name__
            state.last_latency_ms = (time.perf_counter() - started) * 1000
            logger.debug("WarmSessionManager: %s warm failed: %s", binding.name, exc)
            return

        if self._lifecycle_epoch != lifecycle_epoch or stop_event.is_set():
            return
        measured_latency_ms = (time.perf_counter() - started) * 1000
        state.last_latency_ms = (
            measured_latency_ms if result.elapsed_ms is None else float(result.elapsed_ms)
        )
        state.last_status = result.status
        state.last_reason = result.reason
        state.last_provider_session_key = result.provider_session_key
        state.active_worker_age_seconds = self._optional_float(
            result.details.get("prewarm_worker_age_seconds")
        )
        state.active_worker_age_recorded_at = (
            time.monotonic() if state.active_worker_age_seconds is not None else None
        )
        state.stuck_busy_threshold_seconds = (
            binding.policy.timeout_seconds if state.active_worker_age_seconds is not None else None
        )
        if result.ok:
            state.successes += 1
            state.consecutive_failures = 0
            state.last_success_at = time.monotonic()
            state.status = "warm"
            state.active_worker_age_seconds = None
            state.active_worker_age_recorded_at = None
            state.stuck_busy_threshold_seconds = None
        elif (
            result.status == "busy"
            and state.active_worker_age_seconds is not None
            and state.active_worker_age_seconds >= binding.policy.timeout_seconds
        ):
            state.failures += 1
            state.consecutive_failures += 1
            state.status = "degraded"
            state.last_status = "stuck_busy"
            state.last_reason = result.reason or "prewarm_already_running"
        elif result.neutral or result.status in {"busy", "skipped"}:
            # Neutral local admission outcomes did not consume provider traffic.
            # Do not let sustained real-user activity exhaust the cloud-attempt
            # budget and suppress warming for the rest of the hour.
            state.attempt_window_count = max(0, state.attempt_window_count - 1)
            state.skips += 1
            state.status = result.status if result.status in {"busy", "skipped"} else "skipped"
        else:
            state.failures += 1
            state.consecutive_failures += 1
            state.status = "degraded"

    async def _ensure_warm_with_timeout(
        self,
        binding: WarmSessionBinding,
        *,
        force_refresh: bool,
    ) -> WarmSessionResult:
        """Cancel provider work before cancelling its adapter coroutine."""

        attempt = asyncio.create_task(
            binding.target.ensure_warm(force_refresh=force_refresh),
            name=f"warm-session-attempt-{binding.name}",
        )
        try:
            done, _pending = await asyncio.wait(
                (attempt,),
                timeout=binding.policy.timeout_seconds,
            )
            if attempt not in done:
                self._cancel_target(binding)
                attempt.cancel()
                attempt.add_done_callback(self._consume_task_result)
                raise TimeoutError
            return attempt.result()
        except asyncio.CancelledError:
            self._cancel_target(binding)
            attempt.cancel()
            if attempt.done():
                self._consume_task_result(attempt)
            else:
                attempt.add_done_callback(self._consume_task_result)
            raise

    def _next_delay(self, policy: WarmSessionPolicy, state: _WarmSessionState) -> float:
        if state.last_status == "throttled":
            base = max(
                0.0,
                3600.0 - (time.monotonic() - state.attempt_window_started_at),
            )
            return base
        elif state.status == "busy":
            base = policy.busy_retry_seconds
        elif state.consecutive_failures:
            base = min(
                policy.max_backoff_seconds,
                policy.initial_backoff_seconds * (2 ** (state.consecutive_failures - 1)),
            )
        else:
            base = policy.refresh_interval_seconds
        return max(0.0, self._with_jitter(base, policy.jitter_ratio))

    @staticmethod
    def _with_jitter(base: float, ratio: float) -> float:
        if base <= 0 or ratio <= 0:
            return base
        span = base * ratio
        return random.uniform(base - span, base + span)

    @staticmethod
    async def _sleep(
        seconds: float,
        stop_event: asyncio.Event,
        refresh_event: asyncio.Event,
    ) -> None:
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        stop_wait = asyncio.create_task(stop_event.wait())
        refresh_wait = asyncio.create_task(refresh_event.wait())
        try:
            await asyncio.wait(
                (stop_wait, refresh_wait),
                timeout=seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for task in (stop_wait, refresh_wait):
                if not task.done():
                    task.cancel()
            await asyncio.gather(stop_wait, refresh_wait, return_exceptions=True)

    @staticmethod
    def _snapshot_state(
        state: _WarmSessionState,
        now: float,
        policy: WarmSessionPolicy,
    ) -> dict[str, Any]:
        active_worker_age_seconds = state.active_worker_age_seconds
        if (
            active_worker_age_seconds is not None
            and state.active_worker_age_recorded_at is not None
        ):
            active_worker_age_seconds = max(
                0.0,
                active_worker_age_seconds + (now - state.active_worker_age_recorded_at),
            )
        return {
            "status": state.status,
            "attempts": state.attempts,
            "successes": state.successes,
            "failures": state.failures,
            "skips": state.skips,
            "consecutive_failures": state.consecutive_failures,
            "last_result": state.last_status,
            "last_status": state.last_status,
            "last_reason": state.last_reason or "",
            "last_latency_ms": state.last_latency_ms,
            "provider_session_key": state.last_provider_session_key,
            "active_worker_age_seconds": active_worker_age_seconds,
            "stuck_busy_threshold_seconds": state.stuck_busy_threshold_seconds,
            "attempt_budget_remaining": max(
                0,
                policy.max_attempts_per_hour - state.attempt_window_count,
            ),
            "last_success_age_seconds": (
                None if state.last_success_at is None else max(0.0, now - state.last_success_at)
            ),
            "next_attempt_in_seconds": (
                None if state.next_attempt_at is None else max(0.0, state.next_attempt_at - now)
            ),
        }

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
