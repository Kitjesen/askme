"""
Thin client for NOVA Dog dog-safety-service.

Sends fire-and-forget E-STOP notifications so Thunder's safety layer
is aware of voice-triggered stops — without blocking the local arm stop.

Configuration (env or config.yaml under ``runtime.dog_safety``):
  DOG_SAFETY_SERVICE_URL — base URL of dog-safety-service, e.g. http://localhost:5070
  RUNTIME_BEARER_TOKEN   — Bearer token for service auth
  RUNTIME_OPERATOR_ID    — Operator ID header (default: askme)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any
from uuid import uuid4

import requests

logger = logging.getLogger(__name__)

_ESTOP_PATH = "/api/v1/safety/modes/estop"
_HEALTH_PATH = "/api/v1/health"
_CONNECT_TIMEOUT = 0.3   # seconds — very short, E-STOP must not block
_READ_TIMEOUT = 0.5
_STATE_QUERY_TIMEOUT = (0.5, 1.0)  # read estop state — slightly more generous
_ESTOP_STATE_TTL = 30.0  # seconds before cached state is considered stale


class DogSafetyClient:
    """HTTP client for dog-safety-service E-STOP notification.

    All calls are fire-and-forget daemon threads so the local arm stop
    is never delayed by network latency or service unavailability.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._base_url: str = (
            cfg.get("base_url")
            or os.environ.get("DOG_SAFETY_SERVICE_URL", "")
        ).rstrip("/")
        self._token: str = (
            cfg.get("bearer_token")
            or os.environ.get("RUNTIME_BEARER_TOKEN", "")
        )
        self._operator_id: str = (
            cfg.get("operator_id")
            or os.environ.get("RUNTIME_OPERATOR_ID", "askme")
        )

        # Cached estop state — refreshed lazily; never blocks callers
        self._cached_estop: dict[str, Any] | None = None
        self._cache_ts: float = 0.0
        self._state_lock = threading.Lock()

        if self._base_url:
            logger.info("DogSafetyClient configured: %s", self._base_url)
        else:
            logger.debug("DogSafetyClient: DOG_SAFETY_SERVICE_URL not set, notifications disabled")

    def is_configured(self) -> bool:
        return bool(self._base_url)

    # ── State query (read) ────────────────────────────────────────────────────

    def query_estop_state(self) -> dict[str, Any] | None:
        """Query current estop state from dog-safety-service.

        Returns cached data if still fresh (< 30 s old).
        Returns None if the service is not configured or unreachable.
        This method blocks on network I/O — call via ``asyncio.to_thread()``
        from async contexts unless the cache is guaranteed warm.
        """
        if not self._base_url:
            return None

        now = time.monotonic()
        with self._state_lock:
            if (
                self._cached_estop is not None
                and (now - self._cache_ts) < _ESTOP_STATE_TTL
            ):
                return dict(self._cached_estop)

        headers = {"X-Operator-Id": self._operator_id}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        try:
            resp = requests.get(
                f"{self._base_url}{_ESTOP_PATH}",
                headers=headers,
                timeout=_STATE_QUERY_TIMEOUT,
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            with self._state_lock:
                self._cached_estop = data
                self._cache_ts = time.monotonic()
            return dict(data)
        except Exception as exc:
            logger.debug("DogSafetyClient: estop state query failed: %s", exc)
            return None

    def is_estop_active(self) -> bool:
        """Return True if estop is currently active.

        Uses only the in-memory cache — never triggers a network call.
        Returns False when no cached data is available (service down = assume
        safe to avoid blocking normal operation on a disconnected robot).
        """
        with self._state_lock:
            if self._cached_estop is None:
                return False
            if (time.monotonic() - self._cache_ts) > _ESTOP_STATE_TTL:
                return False  # stale — treat as unknown = safe
            return bool(self._cached_estop.get("enabled", False))

    def notify_estop(self) -> None:
        """Fire E-STOP notification to dog-safety-service in a background thread.

        Non-blocking — returns immediately.  The notification is best-effort;
        local arm stop always proceeds regardless of whether this succeeds.
        """
        if not self._base_url:
            return
        t = threading.Thread(target=self._send_estop, daemon=True)
        t.start()

    def _send_estop(self) -> None:
        request_id = uuid4().hex[:16]
        headers = {
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
            "X-Correlation-Id": request_id,
            "X-Operator-Id": self._operator_id,
            "Idempotency-Key": f"askme-estop-{request_id}",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        try:
            resp = requests.post(
                f"{self._base_url}{_ESTOP_PATH}",
                json={"enabled": True},
                headers=headers,
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
            )
            resp.raise_for_status()
            logger.warning(
                "[DogSafety] E-STOP notified to dog-safety-service (HTTP %d)",
                resp.status_code,
            )
        except requests.Timeout:
            logger.warning("[DogSafety] E-STOP notification timed out (service unreachable)")
        except requests.RequestException as exc:
            logger.warning("[DogSafety] E-STOP notification failed: %s", exc)
