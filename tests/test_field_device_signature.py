from __future__ import annotations

import ast
from pathlib import Path

from askme.pipeline.field_operations import (
    sign_field_device_payload as legacy_sign_field_device_payload,
)

from askme.pipeline.field.field_device_signature import (
    FIELD_DEVICE_SIGNATURE_ALG,
    FIELD_DEVICE_SIGNATURE_FIELDS,
    field_device_id,
    field_device_signature_timestamp,
    field_device_signature_value,
    parse_field_device_timestamp,
    sign_field_device_payload,
    unsigned_field_device_payload,
)


def test_field_device_signature_is_stable_and_legacy_compatible() -> None:
    first = {
        "source": "sensor",
        "device_id": "smoke-01",
        "sensor": {"smoke_level": 0.9, "temperature_c": 72},
        "device_signature_timestamp": 1770000000.0,
    }
    reordered = {
        "device_signature_timestamp": 1770000000.0,
        "sensor": {"temperature_c": 72, "smoke_level": 0.9},
        "device_id": "smoke-01",
        "source": "sensor",
    }

    assert FIELD_DEVICE_SIGNATURE_ALG == "hmac-sha256"
    assert sign_field_device_payload(first, secret="secret") == sign_field_device_payload(
        reordered,
        secret="secret",
    )
    assert legacy_sign_field_device_payload(first, secret="secret") == sign_field_device_payload(
        first,
        secret="secret",
    )


def test_field_device_signature_ignores_signature_fields() -> None:
    base = {
        "source": "sensor",
        "device_id": "smoke-01",
        "value": 1,
        "device_signature_timestamp": 1770000000.0,
    }
    signed = {
        **base,
        "device_signature": "old",
        "signature": "legacy",
        "x_signature": "header",
        "device_signature_alg": "hmac-sha256",
        "signature_alg": "hmac-sha256",
    }

    assert FIELD_DEVICE_SIGNATURE_FIELDS == {
        "device_signature",
        "signature",
        "x_signature",
        "device_signature_alg",
        "signature_alg",
    }
    assert unsigned_field_device_payload(signed) == base
    assert sign_field_device_payload(signed, secret="secret") == sign_field_device_payload(
        base,
        secret="secret",
    )


def test_field_device_signature_value_timestamp_and_id_resolution() -> None:
    assert field_device_signature_value({"device_signature": " primary ", "signature": "fallback"}) == "primary"
    assert field_device_signature_value({"signature": " legacy "}) == "legacy"
    assert field_device_signature_value({"x_signature": " header "}) == "header"
    assert field_device_signature_value({}) == ""

    assert field_device_signature_timestamp({"device_signature_timestamp": 1770000000.0}) == 1770000000.0
    assert field_device_signature_timestamp({"signature_timestamp": "1970-01-01T00:00:01Z"}) == 1.0
    assert parse_field_device_timestamp("bad") is None

    assert field_device_id({"cameraIndexCode": "hik-01"}, {}) == "hik-01"
    assert field_device_id({}, {"sensor": {"sensor_id": "smoke-01"}}) == "smoke-01"
    assert field_device_id({}, {"robot": {"robot_id": "dog-01"}}) == "dog-01"
    assert field_device_id({}, {"robot": {"device_id": "robot-device-01"}}) == "robot-device-01"


def test_field_device_signature_helper_is_leaf_and_field_operations_uses_it() -> None:
    helper_path = Path("askme/pipeline/field/field_device_signature.py")
    service_path = Path("askme/pipeline/field/field_operations.py")
    helper_tree = ast.parse(helper_path.read_text(encoding="utf-8"))
    service_tree = ast.parse(service_path.read_text(encoding="utf-8"))

    helper_imports = {
        node.module
        for node in ast.walk(helper_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(helper_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    service_imports = {
        node.module
        for node in ast.walk(service_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    service_defs = {
        node.name
        for node in ast.walk(service_tree)
        if isinstance(node, ast.FunctionDef)
    }

    assert "askme.pipeline.field.field_operations" not in helper_imports
    assert "askme.health_server" not in helper_imports
    assert "askme.pipeline.field.field_device_signature" in service_imports
    assert "sign_field_device_payload" not in service_defs
    assert "_unsigned_field_device_payload" not in service_defs
    assert "_field_device_signature_value" not in service_defs
    assert "_field_device_signature_timestamp" not in service_defs
    assert "_field_device_id" not in service_defs
