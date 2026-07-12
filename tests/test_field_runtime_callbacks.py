import pytest
from askme.runtime.field_callbacks import (
    build_field_runtime_callback_payload,
    build_field_runtime_callback_sequence,
    derive_field_runtime_callback_id,
    field_event_id_from_runtime_result,
    sign_field_runtime_callback_payload,
    unsigned_field_runtime_callback_payload,
)


def test_build_field_runtime_callback_payload_signs_canonical_body():
    payload = build_field_runtime_callback_payload(
        status="executing",
        secret="runtime-secret",
        runtime_callback_id="callback-1",
        run_id="run-1",
        handoff_id="handoff-1",
        robot_motion_policy="retreat_to_safe_distance",
        timestamp=1778510000.0,
    )

    assert payload["runtime_signature_alg"] == "hmac-sha256"
    assert payload["runtime_signature"] == sign_field_runtime_callback_payload(
        payload,
        secret="runtime-secret",
    )
    unsigned = unsigned_field_runtime_callback_payload(payload)
    assert "runtime_signature" not in unsigned
    assert unsigned["runtime_callback_id"] == "callback-1"


def test_build_field_runtime_callback_payload_rejects_unknown_status():
    with pytest.raises(ValueError, match="unsupported field runtime callback status"):
        build_field_runtime_callback_payload(status="motor_override_now")


def test_derive_field_runtime_callback_id_ignores_signature_fields():
    body = {
        "status": "completed",
        "runtime_signature_timestamp": 1778510000.0,
        "run_id": "run-1",
        "runtime_signature": "old-signature",
    }
    without_signature = dict(body)
    without_signature.pop("runtime_signature")

    assert derive_field_runtime_callback_id(body) == derive_field_runtime_callback_id(
        without_signature
    )


def test_build_field_runtime_callback_sequence_from_runtime_result():
    result = {
        "accepted": True,
        "run": {
            "run_id": "run-1",
            "profile": "shadow",
            "current_state": "shadowed",
            "runtime_events": [
                {"state": "submitted"},
                {"state": "validating"},
                {"state": "preflight"},
                {"state": "shadowed"},
            ],
        },
        "handoff": {
            "handoff_id": "handoff-1",
            "source_plan": {
                "reference": {"resolved": {"field_event_id": "evt-1"}},
                "mission": {
                    "mission": {
                        "field_event": {
                            "event_id": "evt-1",
                            "robot_motion_policy": "keep_distance_observe",
                        }
                    }
                },
            },
        },
    }

    payloads = build_field_runtime_callback_sequence(result, secret="runtime-secret")

    assert field_event_id_from_runtime_result(result) == "evt-1"
    assert [item["status"] for item in payloads] == [
        "submitted",
        "validating",
        "preflight",
        "shadowed",
    ]
    assert payloads[-1]["robot_motion_policy"] == "keep_distance_observe"
    assert payloads[-1]["runtime_signature"] == sign_field_runtime_callback_payload(
        payloads[-1],
        secret="runtime-secret",
    )


def test_build_field_runtime_callback_sequence_maps_rejection():
    result = {
        "accepted": False,
        "reason": "external_runtime_disabled",
        "run": {"run_id": "run-1", "current_state": "blocked"},
        "handoff": {"handoff_id": "handoff-1", "source_plan": {}},
    }

    payloads = build_field_runtime_callback_sequence(
        result,
        event_id="evt-1",
        reason="runtime_not_enabled",
    )

    assert len(payloads) == 1
    assert payloads[0]["status"] == "rejected"
    assert payloads[0]["reason"] == "runtime_not_enabled"
