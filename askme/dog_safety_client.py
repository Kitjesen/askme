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
from typing import Any
from uuid import uuid4

import requests

logger = logging.getLogger(__name__)

_ESTOP_PATH = "/api/v1/safety/modes/estop"
_HEALTH_PATH = "/api/v1/health"
_CONNECT_TIMEOUT = 0.3   # seconds — very short, E-STOP must not block
_READ_TIMEOUT = 0.5


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

        if self._base_url:
            logger.info("DogSafetyClient configured: %s", self._base_url)
        else:
            logger.debug("DogSafetyClient: DOG_SAFETY_SERVICE_URL not set, notifications disabled")

    def is_configured(self) -> bool:
        return bool(self._base_url)

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
