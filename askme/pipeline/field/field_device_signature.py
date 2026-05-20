"""Field-device ingest signature helpers.

These helpers are intentionally leaf-level so CLI tools, ingest bridge code,
and the field runtime can share one signing contract without importing the
field-operation service.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any

FIELD_DEVICE_SIGNATURE_ALG = "hmac-sha256"
FIELD_DEVICE_SIGNATURE_FIELDS = {
    "device_signature",
    "signature",
    "x_signature",
    "device_signature_alg",
    "signature_alg",
}


def sign_field_device_payload(body: dict[str, Any], *, secret: str) -> str:
    """Return the HMAC signature expected on a field-device ingest payload."""

    encoded = json.dumps(
        unsigned_field_device_payload(body),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(str(secret).encode("utf-8"), encoded, hashlib.sha256).hexdigest()


def unsigned_field_device_payload(body: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in body.items()
        if key not in FIELD_DEVICE_SIGNATURE_FIELDS
    }


def field_device_signature_value(body: dict[str, Any]) -> str:
    for key in ("device_signature", "signature", "x_signature"):
        value = body.get(key)
        if value:
            return str(value).strip()
    return ""


def field_device_signature_timestamp(body: dict[str, Any]) -> float | None:
    return parse_field_device_timestamp(
        body.get("device_signature_timestamp") or body.get("signature_timestamp")
    )


def field_device_id(body: dict[str, Any], normalized: dict[str, Any]) -> str:
    sensor = normalized.get("sensor") if isinstance(normalized.get("sensor"), dict) else {}
    robot = normalized.get("robot") if isinstance(normalized.get("robot"), dict) else {}
    for value in (
        normalized.get("device_id"),
        normalized.get("source_id"),
        normalized.get("camera_id"),
        body.get("device_id"),
        body.get("source_id"),
        body.get("cameraIndexCode"),
        body.get("camera_id"),
        sensor.get("sensor_id"),
        robot.get("robot_id"),
        robot.get("device_id"),
    ):
        if value:
            return str(value).strip()
    return ""


def parse_field_device_timestamp(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = None
    if parsed is not None:
        return parsed
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # noqa: UP017 - keep Python 3.10 compatible.
    return dt.timestamp()


__all__ = [
    "FIELD_DEVICE_SIGNATURE_ALG",
    "FIELD_DEVICE_SIGNATURE_FIELDS",
    "field_device_id",
    "field_device_signature_timestamp",
    "field_device_signature_value",
    "parse_field_device_timestamp",
    "sign_field_device_payload",
    "unsigned_field_device_payload",
]
