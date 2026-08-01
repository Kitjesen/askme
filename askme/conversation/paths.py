"""Centralized durable storage paths for Conversation Core."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_TURN_LEDGER_PATH = "data/conversation/turn_ledger.jsonl"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_turn_ledger_path(
    config: Mapping[str, Any] | None = None,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve the one durable Conversation Core event-log path."""

    conversation_config = (config or {}).get("conversation", {})
    if not isinstance(conversation_config, Mapping):
        conversation_config = {}
    configured = (
        os.getenv("ASKME_TURN_LEDGER_PATH")
        or conversation_config.get("turn_ledger_path")
        or DEFAULT_TURN_LEDGER_PATH
    )
    path = Path(str(configured)).expanduser()
    if path.is_absolute():
        return path
    root = Path(project_root) if project_root is not None else _project_root()
    return root / path
