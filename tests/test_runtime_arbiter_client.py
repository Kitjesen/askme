from __future__ import annotations

from askme.runtime.arbiter_client import RuntimeArbiterClient


def test_external_runtime_client_is_disabled_by_default() -> None:
    client = RuntimeArbiterClient.from_config("external", {"endpoint": "http://runtime.local"})

    error = client.validate_submit_ready()
    envelope = client.submission_envelope({"handoff_id": "handoff-1", "plan_id": "plan-1"})

    assert error is not None
    assert error.code == "external_runtime_disabled"
    assert error.to_dict()["endpoint_configured"] is True
    assert envelope["accepted"] is False
    assert envelope["error"]["enable_external_runtime"] is False
    assert envelope["hardware_dispatch"] is False


def test_lab_runtime_client_requires_endpoint_when_enabled() -> None:
    client = RuntimeArbiterClient.from_config("lab", {"enable_external_runtime": True})

    error = client.validate_submit_ready()

    assert error is not None
    assert error.code == "external_runtime_endpoint_required"
    assert error.to_dict()["endpoint_configured"] is False


def test_external_runtime_client_builds_transport_diagnostic_when_explicitly_enabled() -> None:
    client = RuntimeArbiterClient.from_config(
        "external",
        {"enable_external_runtime": True, "endpoint": "http://runtime.local/submit"},
    )

    envelope = client.submission_envelope({"handoff_id": "handoff-1", "plan_id": "plan-1"})

    assert envelope["accepted"] is True
    assert envelope["dispatch_mode"] == "transport_managed"
    assert envelope["handoff_id"] == "handoff-1"
    assert envelope["hardware_dispatch"] is False
