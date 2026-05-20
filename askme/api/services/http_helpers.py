"""Reusable HTTP helper utilities for the askme FastAPI surfaces."""

from __future__ import annotations

import ipaddress
import logging
from collections.abc import Callable
from inspect import Parameter, isawaitable, signature
from typing import Any

from fastapi.responses import JSONResponse

_REMOTE_BIND_HOSTS = frozenset(("", "0.0.0.0", "::", "[::]"))
_LOOPBACK_HOSTS = frozenset(("localhost", "127.0.0.1", "::1", "[::1]"))

logger = logging.getLogger(__name__)


def public_error_payload(
    error: str,
    *,
    message: str | None = None,
    reason: str | None = None,
    next_action: str | None = None,
) -> dict[str, Any]:
    """Build a stable public error envelope while preserving legacy `error`."""

    payload: dict[str, Any] = {"ok": False, "error": error}
    if message:
        payload["message"] = message
    if reason:
        payload["reason"] = reason
    if next_action:
        payload["next_action"] = next_action
    return payload


def require_json_object(value: Any, *, message: str = "JSON object body required") -> dict[str, Any]:
    """Return a JSON object body or raise a stable client-facing ValueError."""

    if not isinstance(value, dict):
        raise ValueError(message)
    return value


def json_snapshot_response(
    provider: Callable[[], dict[str, Any]],
    endpoint_name: str,
) -> JSONResponse:
    """Return a no-store JSON response for a snapshot provider."""

    payload = snapshot_payload(provider, endpoint_name)
    if isinstance(payload, JSONResponse):
        return payload
    return JSONResponse(payload, headers={"Cache-Control": "no-store"})


def snapshot_payload(
    provider: Callable[[], dict[str, Any]],
    endpoint_name: str,
) -> dict[str, Any] | JSONResponse:
    """Run a snapshot provider and convert failures into a public JSON response."""

    try:
        return provider()
    except Exception as exc:
        logger.error("Askme %s endpoint failed: %s", endpoint_name, exc, exc_info=True)
        return JSONResponse(
            {"status": "error", "error": str(exc)},
            status_code=500,
            headers={"Cache-Control": "no-store"},
        )


async def maybe_await(value: Any) -> Any:
    """Await awaitable values while accepting synchronous handler returns."""

    if isawaitable(value):
        return await value
    return value


def accepted_keyword_args(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter kwargs to the callable signature unless it accepts **kwargs."""

    if not kwargs:
        return {}
    try:
        parameters = signature(func).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == Parameter.VAR_KEYWORD for param in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def clean_secret(value: Any) -> str | None:
    """Normalize optional operator-provided secrets."""

    text = "" if value is None else str(value).strip()
    return text or None


def is_remote_bind_host(host: str) -> bool:
    """Return whether a host binding exposes the API beyond loopback."""

    cleaned = host.strip().lower()
    if cleaned in _REMOTE_BIND_HOSTS:
        return True
    if cleaned in _LOOPBACK_HOSTS:
        return False
    try:
        address = ipaddress.ip_address(cleaned.strip("[]"))
    except ValueError:
        return True
    return not address.is_loopback


__all__ = [
    "accepted_keyword_args",
    "clean_secret",
    "is_remote_bind_host",
    "json_snapshot_response",
    "maybe_await",
    "public_error_payload",
    "require_json_object",
    "snapshot_payload",
]
