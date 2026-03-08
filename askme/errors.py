"""Standardised error response format for askme MCP tools."""

from __future__ import annotations

import json
from typing import Any

# Error code constants
ROBOT_NOT_CONNECTED = "robot_not_connected"
VOICE_NOT_AVAILABLE = "voice_not_available"
SKILL_NOT_FOUND = "skill_not_found"
SKILL_DISABLED = "skill_disabled"
INTERNAL_ERROR = "internal_error"


def error_response(code: str, message: str, details: Any = None) -> str:
    """Return a uniform JSON error string for MCP tool responses.

    All MCP tools should use this for error returns so that clients
    can parse errors consistently.
    """
    resp: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details is not None:
        resp["error"]["details"] = details
    return json.dumps(resp, ensure_ascii=False)
