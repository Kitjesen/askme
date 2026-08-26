"""TTS adapter for runtime warm-session maintenance."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from askme.runtime.warm_sessions import WarmSessionResult

logger = logging.getLogger(__name__)

TTSTargetResolver = Callable[[], Any | None]
_BUSY_TTS_REASONS = frozenset(
    {
        "already_running",
        "synthesis_busy",
        "synthesis_started",
    }
)


@dataclass(frozen=True)
class _ActivePrewarmWorker:
    engine: Any
    started_at: float


class TTSWarmSessionTarget:
    """Refresh the live TTS provider session without blocking synthesis.

    Provider prewarm APIs are synchronous today. They run on manager-owned
    daemon workers instead of asyncio's default executor: cancelling the
    adapter therefore cannot make ``asyncio.run()`` wait for an uncooperative
    provider thread during process shutdown.
    """

    name = "tts"

    def __init__(self, resolver: TTSTargetResolver) -> None:
        self._resolver = resolver
        self._active_lock = threading.Lock()
        self._active_workers: dict[threading.Thread, _ActivePrewarmWorker] = {}

    async def ensure_warm(self, *, force_refresh: bool) -> WarmSessionResult:
        engine = self._resolver()
        if engine is None:
            return WarmSessionResult(
                ok=False,
                status="skipped",
                reason="tts_not_available",
                neutral=True,
            )
        prewarm = getattr(engine, "prewarm_provider_session", None)
        if not callable(prewarm):
            return WarmSessionResult(
                ok=False,
                status="skipped",
                reason="prewarm_not_supported",
                neutral=True,
                provider_session_key=self._provider_session_key(engine),
            )

        raw_result = await self._run_provider_prewarm(
            engine,
            prewarm,
            force_refresh=force_refresh,
        )
        provider_session_key = self._provider_session_key(engine)
        if isinstance(raw_result, _ActivePrewarmWorker):
            return WarmSessionResult(
                ok=False,
                status="busy",
                reason="prewarm_already_running",
                neutral=True,
                provider_session_key=provider_session_key,
                details={
                    "prewarm_worker_age_seconds": max(0.0, time.monotonic() - raw_result.started_at)
                },
            )
        if not isinstance(raw_result, dict):
            return WarmSessionResult(
                ok=False,
                status="failed",
                reason="invalid_provider_result",
                provider_session_key=provider_session_key,
            )

        ok = bool(raw_result.get("ok", False))
        raw_status = str(raw_result.get("status") or "").strip().lower()
        reason = str(raw_result.get("reason") or "").strip()
        elapsed_ms = self._optional_float(raw_result.get("elapsed_ms"))
        reused = bool(raw_result.get("reused", False))
        if ok:
            return WarmSessionResult(
                ok=True,
                status=raw_status or "warmed",
                reason=reason,
                elapsed_ms=elapsed_ms,
                reused=reused,
                provider_session_key=provider_session_key,
            )
        if raw_status == "busy" or reason in _BUSY_TTS_REASONS:
            return WarmSessionResult(
                ok=False,
                status="busy",
                reason=reason or raw_status,
                elapsed_ms=elapsed_ms,
                neutral=True,
                provider_session_key=provider_session_key,
            )
        if raw_status in {"cancelled", "skipped", "superseded"}:
            return WarmSessionResult(
                ok=False,
                status="skipped",
                reason=reason or raw_status,
                elapsed_ms=elapsed_ms,
                neutral=True,
                provider_session_key=provider_session_key,
            )
        return WarmSessionResult(
            ok=False,
            status=raw_status or "failed",
            reason=reason or raw_status or "provider_failed",
            elapsed_ms=elapsed_ms,
            provider_session_key=provider_session_key,
        )

    async def _run_provider_prewarm(
        self,
        engine: Any,
        prewarm: Callable[..., Any],
        *,
        force_refresh: bool,
    ) -> Any:
        loop = asyncio.get_running_loop()
        completion: asyncio.Future[Any] = loop.create_future()

        def _worker() -> None:
            current = threading.current_thread()
            try:
                result = prewarm(force_refresh=force_refresh)
            except BaseException as exc:
                self._publish_worker_result(loop, completion, error=exc)
            else:
                self._publish_worker_result(loop, completion, result=result)
            finally:
                with self._active_lock:
                    self._active_workers.pop(current, None)

        worker = threading.Thread(
            target=_worker,
            name="warm-session-tts-provider",
            daemon=True,
        )
        with self._active_lock:
            for active_worker in self._active_workers.values():
                if active_worker.engine is engine:
                    return active_worker
            self._active_workers[worker] = _ActivePrewarmWorker(
                engine=engine,
                started_at=time.monotonic(),
            )
        try:
            worker.start()
        except BaseException:
            with self._active_lock:
                self._active_workers.pop(worker, None)
            raise
        return await completion

    @staticmethod
    def _publish_worker_result(
        loop: asyncio.AbstractEventLoop,
        completion: asyncio.Future[Any],
        *,
        result: Any = None,
        error: BaseException | None = None,
    ) -> None:
        def _settle() -> None:
            if completion.done():
                return
            if error is not None:
                completion.set_exception(error)
            else:
                completion.set_result(result)

        try:
            loop.call_soon_threadsafe(_settle)
        except RuntimeError:
            # The daemon worker may finish after a bounded runtime shutdown.
            # The result has no consumer once its event loop is closed.
            logger.debug("TTS prewarm worker finished after event-loop shutdown")

    def cancel(self) -> None:
        with self._active_lock:
            engines = tuple(worker.engine for worker in self._active_workers.values())
        seen: set[int] = set()
        for engine in engines:
            identity = id(engine)
            if identity in seen:
                continue
            seen.add(identity)
            cancel = getattr(engine, "cancel_provider_prewarm", None)
            if not callable(cancel):
                continue
            try:
                cancel()
            except Exception as exc:
                logger.debug("TTS provider prewarm cancellation failed: %s", exc)

    @staticmethod
    def _provider_session_key(engine: Any) -> str:
        return str(getattr(engine, "backend", type(engine).__name__) or "unknown").strip()

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
