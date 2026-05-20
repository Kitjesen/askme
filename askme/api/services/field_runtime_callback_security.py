"""Trust and delivery helpers for Field runtime callbacks."""

from __future__ import annotations

import hmac
import time
from datetime import datetime, timezone
from typing import Any

from askme.runtime.task.field_callbacks import (
    FIELD_RUNTIME_CALLBACK_SIGNATURE_ALG,
    derive_field_runtime_callback_id,
    sign_field_runtime_callback_payload,
    unsigned_field_runtime_callback_payload,
)

_UTC = timezone.utc
_FIELD_RUNTIME_CALLBACK_TIMESTAMP_FIELDS = (
    "runtime_signature_timestamp",
    "signature_timestamp",
)
_FIELD_RUNTIME_CALLBACK_ID_FIELDS = (
    "runtime_callback_id",
    "callback_id",
    "delivery_id",
    "message_id",
)


def field_runtime_callback_signature_value(body: dict[str, Any]) -> str:
    """Return the supplied callback signature, accepting legacy field names."""

    for key in ("runtime_signature", "signature", "x_signature"):
        value = body.get(key)
        if value:
            return str(value).strip()
    return ""


def parse_field_runtime_timestamp(value: Any) -> float | None:
    """Parse numeric or ISO runtime callback timestamps."""

    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_UTC)
    return parsed.timestamp()


def field_runtime_callback_timestamp(body: dict[str, Any]) -> float | None:
    """Return the first valid runtime callback timestamp."""

    for key in _FIELD_RUNTIME_CALLBACK_TIMESTAMP_FIELDS:
        parsed = parse_field_runtime_timestamp(body.get(key))
        if parsed is not None:
            return parsed
    return None


def field_runtime_callback_id(body: dict[str, Any]) -> str:
    """Return a supplied callback id or derive one from the unsigned body."""

    for key in _FIELD_RUNTIME_CALLBACK_ID_FIELDS:
        value = body.get(key)
        if value:
            return str(value).strip()
    return derive_field_runtime_callback_id(body)


def field_runtime_callback_trust(
    body: dict[str, Any],
    *,
    secret: str,
    max_age_s: float,
    now: float | None = None,
) -> dict[str, Any]:
    """Verify a Field runtime callback signature and timestamp."""

    base = {
        "signature_alg": FIELD_RUNTIME_CALLBACK_SIGNATURE_ALG,
        "secret_configured": bool(secret),
        "signature_verified": False,
        "timestamp_verified": False,
    }
    if not secret:
        return {
            **base,
            "trusted": True,
            "status": "unsigned",
            "reason": "runtime_callback_secret_not_configured",
        }
    signature_alg = str(
        body.get("runtime_signature_alg")
        or body.get("signature_alg")
        or FIELD_RUNTIME_CALLBACK_SIGNATURE_ALG
    )
    if signature_alg != FIELD_RUNTIME_CALLBACK_SIGNATURE_ALG:
        return {
            **base,
            "trusted": False,
            "status": "blocked",
            "reason": "unsupported_runtime_signature_alg",
        }
    actual_signature = field_runtime_callback_signature_value(body)
    if not actual_signature:
        return {
            **base,
            "trusted": False,
            "status": "blocked",
            "reason": "missing_runtime_signature",
        }
    expected_signature = sign_field_runtime_callback_payload(body, secret=secret)
    if not hmac.compare_digest(actual_signature, expected_signature):
        return {
            **base,
            "trusted": False,
            "status": "blocked",
            "reason": "runtime_signature_mismatch",
        }
    timestamp = field_runtime_callback_timestamp(body)
    if timestamp is None:
        return {
            **base,
            "signature_verified": True,
            "trusted": False,
            "status": "blocked",
            "reason": "missing_runtime_signature_timestamp",
        }
    current = time.time() if now is None else now
    age_s = current - timestamp
    if age_s < -5.0:
        return {
            **base,
            "signature_verified": True,
            "trusted": False,
            "status": "blocked",
            "reason": "runtime_signature_from_future",
            "signature_age_s": round(age_s, 3),
        }
    if age_s > max_age_s:
        return {
            **base,
            "signature_verified": True,
            "trusted": False,
            "status": "blocked",
            "reason": "runtime_signature_expired",
            "signature_age_s": round(age_s, 3),
        }
    return {
        **base,
        "trusted": True,
        "status": "trusted",
        "reason": "signature_verified",
        "signature_verified": True,
        "timestamp_verified": True,
        "signature_age_s": round(age_s, 3),
    }


def field_runtime_callback_delivery_body(
    body: dict[str, Any],
    *,
    trust: dict[str, Any],
) -> dict[str, Any]:
    """Build the callback delivery payload archived against a field event."""

    delivery = unsigned_field_runtime_callback_payload(body)
    delivery.setdefault("runtime_callback_id", field_runtime_callback_id(body))
    delivery["runtime_callback_trust"] = trust
    return delivery


__all__ = [
    "field_runtime_callback_delivery_body",
    "field_runtime_callback_id",
    "field_runtime_callback_signature_value",
    "field_runtime_callback_timestamp",
    "field_runtime_callback_trust",
    "parse_field_runtime_timestamp",
    "sign_field_runtime_callback_payload",
]
