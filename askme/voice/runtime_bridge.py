"""Bridge askme front-door turns into NOVA Dog runtime askme-edge-service."""

from __future__ import annotations

import logging
from typing import Any

import requests

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

        if (self.enabled or self.text_enabled) and not self._base_url:
            logger.warning("VoiceRuntimeBridge enabled but base_url is empty; disabling bridge")
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

        headers = {
            "Content-Type": "application/json",
            "X-Operator-Id": payload["operator_id"],
            "X-Service-Name": "askme",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            response = requests.post(
                f"{self._base_url}/api/voice/turns",
                json=payload,
                headers=headers,
                timeout=self._timeout_s,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logger.warning("Voice runtime bridge request failed: %s", exc)
            return None


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
