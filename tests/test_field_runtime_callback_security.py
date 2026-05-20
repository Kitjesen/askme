from askme.api.services.field_runtime_callback_security import (
    field_runtime_callback_delivery_body,
    field_runtime_callback_trust,
    parse_field_runtime_timestamp,
    sign_field_runtime_callback_payload,
)


def test_field_runtime_callback_trust_accepts_signed_payload() -> None:
    body = {
        "status": "completed",
        "run_id": "run-1",
        "runtime_signature_timestamp": 1770000000.0,
    }
    body["runtime_signature_alg"] = "hmac-sha256"
    body["runtime_signature"] = sign_field_runtime_callback_payload(body, secret="secret")

    trust = field_runtime_callback_trust(
        body,
        secret="secret",
        max_age_s=30.0,
        now=1770000001.5,
    )

    assert trust["trusted"] is True
    assert trust["status"] == "trusted"
    assert trust["signature_verified"] is True
    assert trust["timestamp_verified"] is True
    assert trust["signature_age_s"] == 1.5


def test_field_runtime_callback_trust_blocks_mismatch_and_expiry() -> None:
    mismatch = {
        "status": "completed",
        "runtime_signature_timestamp": 1770000000.0,
        "runtime_signature_alg": "hmac-sha256",
        "runtime_signature": "bad",
    }
    assert field_runtime_callback_trust(
        mismatch,
        secret="secret",
        max_age_s=30.0,
        now=1770000001.0,
    )["reason"] == "runtime_signature_mismatch"

    expired = {
        "status": "completed",
        "runtime_signature_timestamp": 1770000000.0,
        "runtime_signature_alg": "hmac-sha256",
    }
    expired["runtime_signature"] = sign_field_runtime_callback_payload(expired, secret="secret")
    trust = field_runtime_callback_trust(
        expired,
        secret="secret",
        max_age_s=30.0,
        now=1770000100.0,
    )

    assert trust["trusted"] is False
    assert trust["reason"] == "runtime_signature_expired"


def test_field_runtime_callback_trust_allows_unsigned_when_secret_missing() -> None:
    trust = field_runtime_callback_trust(
        {"status": "completed"},
        secret="",
        max_age_s=30.0,
        now=1770000000.0,
    )

    assert trust["trusted"] is True
    assert trust["status"] == "unsigned"
    assert trust["reason"] == "runtime_callback_secret_not_configured"


def test_field_runtime_callback_delivery_body_removes_signature_fields() -> None:
    body = {
        "status": "completed",
        "run_id": "run-1",
        "runtime_signature_timestamp": 1770000000.0,
        "runtime_signature_alg": "hmac-sha256",
        "runtime_signature": "secret-signature",
    }
    trust = {"trusted": True, "status": "trusted"}

    delivery = field_runtime_callback_delivery_body(body, trust=trust)

    assert delivery["status"] == "completed"
    assert delivery["run_id"] == "run-1"
    assert delivery["runtime_callback_id"].startswith("sha256:")
    assert delivery["runtime_callback_trust"] == trust
    assert "runtime_signature" not in delivery
    assert "runtime_signature_alg" not in delivery


def test_parse_field_runtime_timestamp_accepts_iso_zulu() -> None:
    assert parse_field_runtime_timestamp("1970-01-01T00:00:01Z") == 1.0
