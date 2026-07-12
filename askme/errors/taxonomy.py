"""Enterprise-grade error taxonomy for the Askme system.

Provides:
- ``ErrorCategory`` — high-level error domains (robot, voice, skill, system, …)
- ``ErrorSeverity`` — operational triage severity
- ``AppError`` — structured exception with code, category, severity, HTTP status
- ``ERRORS`` — a pre-defined error catalog keyed by error code
- ``get_error()`` — catalog lookup helper
- Backward-compatible exports of ``error_response()`` and error-code constants
  (``ROBOT_NOT_CONNECTED``, ``VOICE_NOT_AVAILABLE``, …)
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error category — maps to system domain
# ---------------------------------------------------------------------------

class ErrorCategory(str, Enum):  # noqa: UP042 — keep str+Enum for Python 3.10 compat
    """High-level domain that produced the error."""

    SYSTEM = "system"
    ROBOT = "robot"
    VOICE = "voice"
    SKILL = "skill"
    API = "api"
    MEMORY = "memory"
    AUTH = "auth"
    NETWORK = "network"
    INTERNAL = "internal"


# ---------------------------------------------------------------------------
# Error severity — operational triage
# ---------------------------------------------------------------------------

class ErrorSeverity(str, Enum):  # noqa: UP042 — keep str+Enum for Python 3.10 compat
    """Severity level for operational triage and alerting."""

    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ---------------------------------------------------------------------------
# Structured exception
# ---------------------------------------------------------------------------

class AppError(Exception):
    """Base application exception carrying structured error metadata.

    Usage::

        raise AppError(
            "skill_timeout",
            "Skill execution exceeded maximum duration",
            category=ErrorCategory.SKILL,
            severity=ErrorSeverity.ERROR,
            status_code=504,
            details={"skill": "grasp", "duration_s": 32},
        )
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Any = None,
        status_code: int = 500,
    ) -> None:
        self.code = code
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details
        self.status_code = status_code
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation suitable for JSON encoding."""
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "status_code": self.status_code,
        }
        if self.details is not None:
            result["details"] = self.details
        return result

    def to_error_body(self) -> dict[str, Any]:
        """Return the MCP-style ``{"error": {...}}`` body."""
        return {"error": self.to_dict()}


# ---------------------------------------------------------------------------
# Backward-compatible error-code constants
# ---------------------------------------------------------------------------

ROBOT_NOT_CONNECTED = "robot_not_connected"
VOICE_NOT_AVAILABLE = "voice_not_available"
SKILL_NOT_FOUND = "skill_not_found"
SKILL_DISABLED = "skill_disabled"
INTERNAL_ERROR = "internal_error"


# ---------------------------------------------------------------------------
# Error catalog
# ---------------------------------------------------------------------------

ERRORS: dict[str, dict[str, Any]] = {
    ROBOT_NOT_CONNECTED: {
        "category": ErrorCategory.ROBOT,
        "severity": ErrorSeverity.ERROR,
        "status_code": 503,
        "message": "Robot arm not connected or not enabled",
    },
    VOICE_NOT_AVAILABLE: {
        "category": ErrorCategory.VOICE,
        "severity": ErrorSeverity.ERROR,
        "status_code": 503,
        "message": "Voice I/O not initialised",
    },
    SKILL_NOT_FOUND: {
        "category": ErrorCategory.SKILL,
        "severity": ErrorSeverity.WARNING,
        "status_code": 404,
        "message": "Skill not found",
    },
    SKILL_DISABLED: {
        "category": ErrorCategory.SKILL,
        "severity": ErrorSeverity.WARNING,
        "status_code": 403,
        "message": "Skill is disabled",
    },
    INTERNAL_ERROR: {
        "category": ErrorCategory.INTERNAL,
        "severity": ErrorSeverity.ERROR,
        "status_code": 500,
        "message": "Internal server error",
    },
}


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

def get_error(code: str) -> dict[str, Any] | None:
    """Look up pre-defined error metadata by *code*, or *None* if unknown."""
    return ERRORS.get(code)


# ---------------------------------------------------------------------------
# Backward-compatible MCP error helper
# ---------------------------------------------------------------------------

def error_response(code: str, message: str, details: Any = None) -> str:
    """Return a uniform JSON error string for MCP tool responses.

    All MCP tools should use this for error returns so that clients
    can parse errors consistently.
    """
    resp: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details is not None:
        resp["error"]["details"] = details
    return json.dumps(resp, ensure_ascii=False)


__all__ = [
    "AppError",
    "ERRORS",
    "ErrorCategory",
    "ErrorSeverity",
    "INTERNAL_ERROR",
    "ROBOT_NOT_CONNECTED",
    "SKILL_DISABLED",
    "SKILL_NOT_FOUND",
    "VOICE_NOT_AVAILABLE",
    "error_response",
    "get_error",
]
