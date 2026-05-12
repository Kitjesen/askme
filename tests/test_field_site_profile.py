from __future__ import annotations

import time
from pathlib import Path

import pytest

from askme.pipeline.field_operations import FieldOperationsService
from askme.pipeline.field_site_profile import (
    build_site_profile_report,
    field_operations_config_from_site_profile,
    load_field_site_profile,
    validate_field_site_profile,
)


def test_demo_field_site_profile_passes_and_exports_runtime_config() -> None:
    profile_path = Path("deploy/site-profiles/park-demo.yaml")

    report = build_site_profile_report(profile_path)

    assert report["status"] == "passed"
    assert report["summary"]["site_id"] == "inovx-demo-park"
    assert report["summary"]["parking_restricted_count"] >= 1
    assert report["summary"]["help_point_count"] >= 1
    assert report["summary"]["device_sources"]["camera"] >= 1
    assert report["summary"]["device_sources"]["sensor"] >= 1
    assert report["summary"]["device_sources"]["robot"] >= 1
    assert report["readiness"]["wayfinding_configured"] is True
    assert report["field_operations_config"]["site_map"]["zones"]["main-road-1"][
        "parking_allowed"
    ] is False
    assert report["field_operations_config"]["dingtalk_webhooks"]["security"] == (
        "${ASKME_DINGTALK_SECURITY_WEBHOOK}"
    )


def test_field_site_profile_env_check_warns_for_unset_references(monkeypatch) -> None:
    monkeypatch.delenv("ASKME_DINGTALK_SECURITY_WEBHOOK", raising=False)

    report = build_site_profile_report(
        Path("deploy/site-profiles/park-demo.yaml"),
        check_env=True,
    )

    assert report["status"] == "passed"
    assert (
        "responder_groups.security.webhook_env references unset environment variable "
        "ASKME_DINGTALK_SECURITY_WEBHOOK"
    ) in report["warnings"]


def test_field_site_profile_rejects_missing_product_critical_sections() -> None:
    report = validate_field_site_profile({
        "site": {"site_id": "bad"},
        "zones": {
            "parking": {"type": "parking_area", "parking_allowed": True},
        },
        "responder_groups": {},
        "devices": {},
        "thresholds": {},
    })

    assert report["status"] == "failed"
    assert "site.name is required" in report["errors"]
    assert "zones must include at least one main_channel" in report["errors"]
    assert "zones must include at least one help_point" in report["errors"]
    assert "devices must contain at least one registered device" in report["errors"]
    assert "responder_groups.security is required" in report["errors"]


def test_field_site_profile_exports_device_registry_with_env_placeholders() -> None:
    profile = load_field_site_profile(Path("deploy/site-profiles/park-demo.yaml"))

    config = field_operations_config_from_site_profile(profile)

    assert config["device_registry"]["camera-main-road-1"]["secret"] == (
        "${ASKME_FIELD_CAMERA_MAIN_ROAD_SECRET}"
    )
    assert config["device_registry"]["robot-thunder-1"]["robot_id"] == "thunder-1"


@pytest.mark.asyncio
async def test_field_operations_service_loads_site_profile_for_real_ingest(tmp_path: Path) -> None:
    service = FieldOperationsService(config={
        "archive_path": str(tmp_path / "events.jsonl"),
        "site_profile_path": "deploy/site-profiles/park-demo.yaml",
    })

    result = await service.ingest_payload({
        "source": "camera",
        "observed_at": time.time(),
        "zone_id": "main-road-1",
        "detections": [{"label": "vehicle", "confidence": 0.92}],
        "duration_s": 180,
        "image_path": "artifacts/evidence/car.jpg",
    })

    assert result["accepted"] is True
    assert result["normalized"]["location"] == "B区主通道"
    assert result["event"]["incident_topic"] == "traffic.illegal_parking"
    assert result["event"]["location"] == "B区主通道"


def test_field_operations_service_rejects_invalid_site_profile(tmp_path: Path) -> None:
    profile = tmp_path / "bad-site.yaml"
    profile.write_text(
        "site:\n  site_id: bad\nzones: {}\nresponder_groups: {}\ndevices: {}\nthresholds: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="field site profile validation failed"):
        FieldOperationsService(config={"site_profile_path": str(profile)})


def test_field_operations_service_can_surface_site_profile_env_warnings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("ASKME_DINGTALK_SECURITY_WEBHOOK", raising=False)
    service = FieldOperationsService(config={
        "archive_path": str(tmp_path / "events.jsonl"),
        "site_profile_path": "deploy/site-profiles/park-demo.yaml",
        "site_profile_check_env": True,
    })

    payload = service.readiness_payload()

    assert any(
        item.startswith("field site profile: responder_groups.security.webhook_env")
        for item in payload["warnings"]
    )
    assert (
        "Set site profile environment variables for DingTalk responders and field devices"
        in payload["next_actions"]
    )
