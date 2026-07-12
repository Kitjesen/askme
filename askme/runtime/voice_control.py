"""Persistence helpers for the live voice-system control plane."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from askme.config import project_root

_ALLOWED_STATE_KEYS = {"llm", "asr", "tts", "prompt"}
_SECRET_TOKENS = ("api_key", "access_token", "secret", "password", "token")


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive copy of ``base`` updated by ``patch``."""

    merged = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def contains_secret_key(payload: Any) -> bool:
    """Return whether a persisted payload contains a credential-like key."""

    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = str(key).lower()
            if any(token in normalized for token in _SECRET_TOKENS):
                return True
            if contains_secret_key(value):
                return True
    elif isinstance(payload, list):
        return any(contains_secret_key(item) for item in payload)
    return False


class VoiceControlStateStore:
    """Atomic, non-secret persistence for live runtime selections."""

    def __init__(self, config: dict[str, Any]) -> None:
        voice_cfg = config.get("voice", {}) if isinstance(config.get("voice"), dict) else {}
        configured_path = voice_cfg.get("control_state_path")
        self.enabled = bool(str(configured_path or "").strip())
        raw_path = configured_path or "data/voice/system_control.json"
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = project_root() / path
        self.path = path

    def load(self) -> dict[str, Any]:
        if not self.enabled:
            return {}
        if not self.path.is_file():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {}
        if not isinstance(payload, dict) or contains_secret_key(payload):
            return {}
        return {
            key: deepcopy(value)
            for key, value in payload.items()
            if key in _ALLOWED_STATE_KEYS and isinstance(value, dict)
        }

    def save(self, state: dict[str, Any]) -> dict[str, Any]:
        payload = {
            key: deepcopy(value)
            for key, value in state.items()
            if key in _ALLOWED_STATE_KEYS and isinstance(value, dict)
        }
        if contains_secret_key(payload):
            raise ValueError("runtime control state cannot persist credentials")
        if not self.enabled:
            return payload
        payload["version"] = 1
        payload["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)
        return payload


__all__ = [
    "VoiceControlStateStore",
    "contains_secret_key",
    "deep_merge",
]
