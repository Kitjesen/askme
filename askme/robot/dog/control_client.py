"""
Thin HTTP client for NOVA Dog dog-control-service capability dispatch.

Sends capability dispatch requests so Thunder's control layer can execute
姿态 and motion commands triggered by voice — without bypassing the safety
layer (dog-safety-service and dog-control-service are the only two motion
authorities in the NOVA Dog runtime contract).

Configuration (env or config.yaml under ``runtime.dog_control``):
  DOG_CONTROL_SERVICE_URL — base URL of dog-control-service, e.g. http://localhost:5080
  RUNTIME_BEARER_TOKEN    — Bearer token for service auth
  RUNTIME_OPERATOR_ID     — Operator ID header (default: askme)
"""

from __future__ import annotations

import logging
import os
from typing import Any
from uuid import uuid4

import requests

logger = logging.getLogger(__name__)

_DISPATCH_PATH = "/api/v1/control/executions"
_HEALTH_PATH = "/api/v1/health"
_CONNECT_TIMEOUT = 2.0   # seconds
_READ_TIMEOUT = 3.0      # capability dispatch may take a moment to ack


class DogControlClient:
    """HTTP client for dog-control-service capability dispatch.

    Dispatches姿态 and motion capabilities to the runtime control plane.
    All capability requests go through dog-control-service — never bypassing
    the safety layer.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self._base_url: str = (
            cfg.get("base_url")
            or os.environ.get("DOG_CONTROL_SERVICE_URL", "")
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
            logger.info("DogControlClient configured: %s", self._base_url)
        else:
            logger.debug(
                "DogControlClient: DOG_CONTROL_SERVICE_URL not set, "
                "capability dispatch disabled"
            )

    def is_configured(self) -> bool:
        return bool(self._base_url)

    def dispatch_capability(
        self, capability: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Dispatch a capability to dog-control-service.

        This method blocks on network I/O — always call it via
        ``asyncio.to_thread()`` from an async context.

        Args:
            capability: Capability name supported by dog-control-service,
                        e.g. ``stand``, ``sit``, ``start_patrol``.
            params:     Optional parameters for the capability.

        Returns:
            Response dict from the service, or ``{"error": "..."}`` on failure.
        """
        if not self._base_url:
            logger.debug(
                "DogControlClient: dispatch skipped (not configured) — %s", capability
            )
            return {"error": "dog-control-service not configured"}

        request_id = uuid4().hex[:16]
        headers = {
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
            "X-Correlation-Id": request_id,
            "X-Operator-Id": self._operator_id,
            # Required by API_STANDARDS for mutating POSTs — prevents duplicate dispatch
            # on voice-triggered retries (e.g. user repeating "站起来").
            "Idempotency-Key": f"askme-dogctl-{request_id}",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        # Field names match DispatchExecutionRequest schema in control-v1.yaml.
        body: dict[str, Any] = {
            "mission_id": uuid4().hex,
            "mission_type": "posture_command",
            "requested_capability": capability,
            "parameters": params or {},
        }

        try:
            resp = requests.post(
                f"{self._base_url}{_DISPATCH_PATH}",
                json=body,
                headers=headers,
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
            )
            resp.raise_for_status()
            logger.info(
                "[DogControl] Capability '%s' dispatched (HTTP %d)",
                capability, resp.status_code,
            )
            try:
                return resp.json()
            except ValueError:
                return {"status": "ok", "http_status": resp.status_code}
        except requests.Timeout:
            logger.warning(
                "[DogControl] Capability '%s' timed out (service unreachable)", capability
            )
            return {"error": f"timeout dispatching capability '{capability}'"}
        except requests.RequestException as exc:
            logger.warning(
                "[DogControl] Capability '%s' dispatch failed: %s", capability, exc
            )
            return {"error": str(exc)}
