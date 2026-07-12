"""Askme error types and taxonomy.

Re-exports all symbols from :mod:`askme.errors.taxonomy` for backward
compatibility with existing imports::

    from askme.errors import error_response, ROBOT_NOT_CONNECTED
"""

from askme.errors.taxonomy import (
    # Catalog
    ERRORS,
    # Backward-compatible error-code constants
    INTERNAL_ERROR,
    ROBOT_NOT_CONNECTED,
    SKILL_DISABLED,
    SKILL_NOT_FOUND,
    VOICE_NOT_AVAILABLE,
    # Core classes
    AppError,
    ErrorCategory,
    ErrorSeverity,
    # Helpers
    error_response,
    get_error,
)

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
