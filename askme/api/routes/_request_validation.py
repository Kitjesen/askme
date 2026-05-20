"""Small request-validation helpers for API route modules."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from typing import Any

from fastapi.responses import JSONResponse


class RequestFieldError(ValueError):
    """Client supplied an invalid API field value."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(message)
        self.field = field
        self.message = message


def _body_value(
    body: Mapping[str, Any],
    field: str,
    aliases: Iterable[str],
) -> tuple[str, Any]:
    for name in (field, *aliases):
        if name in body:
            return name, body.get(name)
    return field, None


def optional_int_field(
    body: Mapping[str, Any],
    field: str,
    *,
    aliases: Iterable[str] = (),
    default: int | None = None,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int | None:
    """Parse an optional integer request field or raise a structured field error."""

    supplied_name, raw_value = _body_value(body, field, aliases)
    if raw_value in (None, ""):
        return default
    if isinstance(raw_value, bool):
        raise RequestFieldError(field, f"{supplied_name} must be an integer.")
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise RequestFieldError(field, f"{supplied_name} must be an integer.") from exc
    if min_value is not None and value < min_value:
        raise RequestFieldError(field, f"{supplied_name} must be >= {min_value}.")
    if max_value is not None and value > max_value:
        raise RequestFieldError(field, f"{supplied_name} must be <= {max_value}.")
    return value


def optional_float_field(
    body: Mapping[str, Any],
    field: str,
    *,
    aliases: Iterable[str] = (),
    default: float | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    """Parse an optional finite float request field or raise a structured field error."""

    supplied_name, raw_value = _body_value(body, field, aliases)
    if raw_value in (None, ""):
        return default
    if isinstance(raw_value, bool):
        raise RequestFieldError(field, f"{supplied_name} must be a number.")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise RequestFieldError(field, f"{supplied_name} must be a number.") from exc
    if not math.isfinite(value):
        raise RequestFieldError(field, f"{supplied_name} must be a finite number.")
    if min_value is not None and value < min_value:
        raise RequestFieldError(field, f"{supplied_name} must be >= {min_value}.")
    if max_value is not None and value > max_value:
        raise RequestFieldError(field, f"{supplied_name} must be <= {max_value}.")
    return value


def field_error_response(
    exc: RequestFieldError,
    *,
    headers: Mapping[str, str],
) -> JSONResponse:
    """Return a stable 400 response for bad client fields."""

    return JSONResponse(
        {
            "ok": False,
            "error": "invalid_request_field",
            "field": exc.field,
            "message": exc.message,
        },
        status_code=400,
        headers=dict(headers),
    )


def route_error_response(
    logger: logging.Logger,
    *,
    public_error: str,
    exc: Exception,
    headers: Mapping[str, str],
    status_code: int = 500,
) -> JSONResponse:
    """Log route failures with traceback while returning a stable public error."""

    logger.exception("%s: %s", public_error, exc)
    return JSONResponse(
        {"ok": False, "error": public_error},
        status_code=status_code,
        headers=dict(headers),
    )


__all__ = [
    "RequestFieldError",
    "field_error_response",
    "optional_float_field",
    "optional_int_field",
    "route_error_response",
]
