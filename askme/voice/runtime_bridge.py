"""Bridge askme front-door turns into NOVA Dog runtime askme-edge-service."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any
from uuid import uuid4

import requests

from .generated_contracts import AskmeEdgeVoiceContract

logger = logging.getLogger(__name__)


class VoiceRuntimeBridge:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.enabled: bool = bool(cfg.get("enabled", False))
        self.text_enabled: bool = bool(cfg.get("text_enabled", self.enabled))
        self._base_url: str = str(cfg.get("base_url", "")).rstrip("/")
        self._api_key: str | None = str(cfg.get("api_key", "")).strip() or None
        self._operator_id: str = str(cfg.get("operator_id", "askme.voice")).strip()
        self._session_id: str = str(cfg.get("session_id", self._operator_id)).strip()
        self._channel: str = str(cfg.get("channel", "voice")).strip()
        self._text_session_id: str = str(cfg.get("text_session_id", "askme-local-text")).strip()
        self._text_channel: str = str(cfg.get("text_channel", "text")).strip()
        self._robot_id: str | None = _clean_optional(cfg.get("robot_id"))
        self._site_id: str | None = _clean_optional(cfg.get("site_id"))
        self._submit: bool = bool(cfg.get("submit", True))
        self._timeout_s: float = float(cfg.get("timeout", 2.0))
        self._failure_threshold: int = max(1, int(cfg.get("failure_threshold", 2)))
        self._failure_cooldown_s: float = max(0.0, float(cfg.get("failure_cooldown", 15.0)))
        self._state_lock = threading.Lock()
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

        if (self.enabled or self.text_enabled) and not self._base_url:
            logger.warning("VoiceRuntimeBridge enabled but base_url is empty; disabling bridge")
            self.enabled = False
            self.text_enabled = False

        # Fast connectivity pre-check: if the upstream is unreachable at init
        # time, disable immediately instead of waiting for 2 slow failures.
        if (self.enabled or self.text_enabled) and self._base_url:
            try:
                requests.get(f"{self._base_url}{AskmeEdgeVoiceContract.HEALTH.path()}", timeout=0.5)
            except requests.RequestException:
                logger.info(
                    "Voice runtime bridge: upstream %s unreachable at startup; "
                    "disabling bridge (no retries)",
                    self._base_url,
                )
                self.enabled = False
                self.text_enabled = False

    def handle_voice_text(self, text: str) -> dict[str, Any] | None:
        return self.handle_turn(
            text,
            enabled=self.enabled,
            session_id=self._session_id,
            channel=self._channel,
        )

    def handle_text_input(self, text: str) -> dict[str, Any] | None:
        return self.handle_turn(
            text,
            enabled=self.text_enabled,
            session_id=self._text_session_id,
            channel=self._text_channel,
        )

    def handle_turn(
        self,
        text: str,
        *,
        enabled: bool | None = None,
        session_id: str | None = None,
        channel: str | None = None,
        operator_id: str | None = None,
        robot_id: str | None = None,
        site_id: str | None = None,
        submit: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        active = self.enabled if enabled is None else bool(enabled)
        if not active or not self._base_url:
            return None
        recovery_probe = self._begin_request()
        if recovery_probe is None:
            return None

        payload = {
            "text": text,
            "operator_id": operator_id or self._operator_id,
            "session_id": session_id or self._session_id,
            "channel": channel or self._channel,
            "robot_id": robot_id if robot_id is not None else self._robot_id,
            "site_id": site_id if site_id is not None else self._site_id,
            "submit": self._submit if submit is None else bool(submit),
        }
        if metadata:
            payload["metadata"] = metadata

        request_id = uuid4().hex[:16]
        headers = {
            "Content-Type": "application/json",
            "X-Operator-Id": payload["operator_id"],
            "X-Service-Name": "askme",
            "X-Request-Id": request_id,
            "X-Correlation-Id": request_id,
            "Idempotency-Key": f"voice-turn-{request_id}",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            response = requests.post(
                f"{self._base_url}{AskmeEdgeVoiceContract.PROCESS_TURN.path()}",
                json=payload,
                headers=headers,
                timeout=self._timeout_s,
            )
            response.raise_for_status()
            response_payload = response.json()
            if not isinstance(response_payload, dict):
                raise ValueError(
                    f"runtime bridge returned non-object payload: {response_payload!r}"
                )
            self._record_success()
            if recovery_probe:
                logger.info("Voice runtime bridge recovered after cooldown")
            return response_payload
        except (requests.RequestException, ValueError) as exc:
            self._record_failure(exc, recovery_probe=recovery_probe)
            return None
        except Exception as exc:
            self._record_failure(exc, recovery_probe=recovery_probe)
            return None

    def _begin_request(self) -> bool | None:
        """Return whether this request is a recovery probe, or None if bypassed."""
        now = time.monotonic()
        with self._state_lock:
            if self._circuit_open_until > now:
                logger.info(
                    "Voice runtime bridge bypassed for %.1fs after repeated failures",
                    self._circuit_open_until - now,
                )
                return None
            if self._circuit_open_until > 0:
                self._circuit_open_until = 0.0
                self._consecutive_failures = max(0, self._failure_threshold - 1)
                logger.info("Voice runtime bridge cooldown elapsed; probing upstream again")
                return True
        return False

    def _record_success(self) -> None:
        with self._state_lock:
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0

    def _record_failure(self, exc: Exception, *, recovery_probe: bool) -> None:
        with self._state_lock:
            self._consecutive_failures += 1
            failure_count = self._consecutive_failures
            should_open = failure_count >= self._failure_threshold
            if should_open and self._failure_cooldown_s > 0:
                self._circuit_open_until = time.monotonic() + self._failure_cooldown_s

        if should_open:
            logger.warning(
                "Voice runtime bridge disabled for %.1fs after %d consecutive failures: %s",
                self._failure_cooldown_s,
                failure_count,
                exc,
            )
            return

        if recovery_probe:
            logger.warning("Voice runtime bridge recovery probe failed: %s", exc)
            return

        logger.warning(
            "Voice runtime bridge request failed (%d/%d): %s",
            failure_count,
            self._failure_threshold,
            exc,
        )


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
