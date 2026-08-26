"""Provider adapters for runtime warm-session maintenance."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from askme.llm.core.contracts import LLMCallContext
from askme.runtime.warm_session_tts import TTSWarmSessionTarget
from askme.runtime.warm_sessions import WarmSessionResult

__all__ = ["LLMWarmSessionTarget", "TTSWarmSessionTarget"]
LLMTargetResolver = Callable[[], tuple[Any, str] | None]


class LLMWarmSessionTarget:
    """Warm the currently active LLM client with a bounded semantic probe."""

    name = "llm"

    def __init__(self, resolver: LLMTargetResolver) -> None:
        self._resolver = resolver
        self._active_cancel: asyncio.Event | None = None

    async def ensure_warm(self, *, force_refresh: bool) -> WarmSessionResult:
        _ = force_refresh
        resolved = self._resolver()
        if resolved is None:
            return WarmSessionResult(
                ok=False,
                status="skipped",
                reason="llm_not_available",
                neutral=True,
            )

        resolved_client, raw_model = resolved
        lease = None
        acquire_warm_target = getattr(resolved_client, "acquire_warm_target", None)
        if callable(acquire_warm_target):
            lease = acquire_warm_target()
            client = lease.client
            model = str(lease.model or getattr(client, "model", "")).strip()
        else:
            client = resolved_client
            model = str(raw_model or getattr(client, "model", "")).strip()
        cancel_event = asyncio.Event()
        self._active_cancel = cancel_event
        started_at = time.perf_counter()
        try:
            if not model:
                return WarmSessionResult(
                    ok=False,
                    status="skipped",
                    reason="model_not_available",
                    neutral=True,
                )
            business_active = self._business_request_is_active(client)
            if business_active is None:
                return WarmSessionResult(
                    ok=False,
                    status="skipped",
                    reason="activity_unavailable",
                    neutral=True,
                    provider_session_key=self._provider_session_key(client, model),
                )
            if business_active:
                return WarmSessionResult(
                    ok=False,
                    status="busy",
                    reason="real_request_priority",
                    neutral=True,
                    provider_session_key=self._provider_session_key(client, model),
                )

            messages = [
                {"role": "system", "content": "Reply with one Chinese character."},
                {"role": "user", "content": "好"},
            ]
            context = LLMCallContext(
                purpose="health_probe",
                channel="system",
                request_class="health_probe",
                privacy_class="operational",
                allow_cache=False,
            )
            async for _chunk in client.chat_stream(
                messages,
                model=model,
                max_tokens=1,
                temperature=0.0,
                cancel_token=cancel_event,
                context=context,
            ):
                pass
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            if cancel_event.is_set():
                business_active = self._business_request_is_active(client)
                if business_active is None:
                    return WarmSessionResult(
                        ok=False,
                        status="skipped",
                        reason="activity_unavailable",
                        elapsed_ms=elapsed_ms,
                        neutral=True,
                        provider_session_key=self._provider_session_key(client, model),
                    )
                return WarmSessionResult(
                    ok=False,
                    status="busy" if business_active else "cancelled",
                    reason="real_request_priority" if business_active else "cancelled",
                    elapsed_ms=elapsed_ms,
                    neutral=True,
                    provider_session_key=self._provider_session_key(client, model),
                )
            return WarmSessionResult(
                ok=True,
                status="probed",
                elapsed_ms=elapsed_ms,
                reused=True,
                provider_session_key=self._provider_session_key(client, model),
            )
        finally:
            if self._active_cancel is cancel_event:
                self._active_cancel = None
            if lease is not None:
                lease.release()

    def cancel(self) -> None:
        cancel_event = self._active_cancel
        if cancel_event is not None:
            cancel_event.set()

    @staticmethod
    def _business_request_is_active(client: Any) -> bool | None:
        """Return business activity, or ``None`` when it cannot be observed safely."""

        try:
            request_activity = getattr(client, "request_activity", None)
        except Exception:
            return None
        if not callable(request_activity):
            return None
        try:
            activity = request_activity()
        except Exception:
            return None
        if not isinstance(activity, dict):
            return None
        try:
            active_business_requests = int(activity.get("active_business_requests", 0) or 0)
        except Exception:
            return None
        if active_business_requests < 0:
            return None
        return active_business_requests > 0

    @staticmethod
    def _provider_session_key(client: Any, model: str) -> str:
        provider = str(getattr(client, "provider_name", "unknown") or "unknown").strip()
        return f"{provider}:{model}"
