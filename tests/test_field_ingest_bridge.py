from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from askme.pipeline.field_ingest_bridge import run_field_ingest_bridge_once
from askme.pipeline.field_operations import FieldOperationsService

FIXTURES = Path(__file__).parent / "fixtures" / "field_devices"


def test_field_ingest_bridge_reads_only_new_jsonl_events(tmp_path: Path) -> None:
    source = tmp_path / "camera-events.jsonl"
    state = tmp_path / "bridge-state.json"
    source.write_text(
        json.dumps({"detections": [{"class_id": "2"}], "duration_s": 180}) + "\n",
        encoding="utf-8",
    )

    first = run_field_ingest_bridge_once(
        source=source,
        server="http://runtime.local:8765",
        state_path=state,
        dry_run=True,
    )

    assert first["count"] == 1
    assert first["results"][0]["normalized"]["detections"][0]["label"] == "vehicle"

    source.write_text(
        source.read_text(encoding="utf-8")
        + json.dumps({"sensor": {"temperature_c": 70, "smoke_level": 0.8}}) + "\n",
        encoding="utf-8",
    )

    second = run_field_ingest_bridge_once(
        source=source,
        server="http://runtime.local:8765",
        state_path=state,
        dry_run=True,
    )
    third = run_field_ingest_bridge_once(
        source=source,
        server="http://runtime.local:8765",
        state_path=state,
        dry_run=True,
    )

    assert second["count"] == 1
    assert second["results"][0]["normalized"]["source"] == "sensor"
    assert third["count"] == 0


def test_field_ingest_bridge_json_snapshot_uses_fingerprint(tmp_path: Path) -> None:
    source = tmp_path / "robot-diagnostic.json"
    state = tmp_path / "bridge-state.json"
    source.write_text(
        json.dumps({"robot": {"fault_type": "joint_motor_fault", "joint_id": "hip-left"}}),
        encoding="utf-8",
    )

    first = run_field_ingest_bridge_once(
        source=source,
        server="http://runtime.local:8765",
        state_path=state,
        dry_run=True,
    )
    second = run_field_ingest_bridge_once(
        source=source,
        server="http://runtime.local:8765",
        state_path=state,
        dry_run=True,
    )

    assert first["count"] == 1
    assert first["results"][0]["normalized"]["robot"]["fault_type"] == "joint_motor_fault"
    assert second["count"] == 0


def test_field_ingest_bridge_posts_normalized_events(tmp_path: Path) -> None:
    source = tmp_path / "events.jsonl"
    state = tmp_path / "bridge-state.json"
    source.write_text(
        json.dumps({
            "timestamp": 1770000000,
            "detections": [{"class_id": "2", "confidence": 0.95}],
            "zone": {"id": "road-1", "name": "主通道", "type": "main_channel", "parking_allowed": False},
            "duration_s": 180,
        })
        + "\n",
        encoding="utf-8",
    )
    posted: list[dict[str, object]] = []

    def fake_post(url: str, body: dict[str, object], timeout_s: float) -> dict[str, object]:
        posted.append({"url": url, "body": body, "timeout_s": timeout_s})
        return {
            "status": "triggered",
            "accepted": True,
            "normalized": {"scenario_id": "illegal_parking"},
            "event": {"event_id": "field-1"},
        }

    payload = run_field_ingest_bridge_once(
        source=source,
        server="http://runtime.local:8765/",
        state_path=state,
        dry_run=False,
        timeout_s=3,
        post_func=fake_post,
    )

    assert payload["status"] == "ok"
    assert payload["results"][0]["posted"] is True
    assert payload["results"][0]["scenario_id"] == "illegal_parking"
    assert payload["summary"]["posted"] == 1
    assert payload["summary"]["accepted"] == 1
    assert payload["summary"]["events_created"] == 1
    assert payload["summary"]["scenario_counts"] == {"illegal_parking": 1}
    assert payload["summary"]["source_format"] == "jsonl"
    assert posted[0]["url"] == "http://runtime.local:8765/api/field/ingest"
    assert posted[0]["body"]["detections"][0]["label"] == "vehicle"
    assert posted[0]["body"]["zone_id"] == "road-1"


def test_field_ingest_bridge_signs_registered_device_payload(tmp_path: Path) -> None:
    source = tmp_path / "signed-sensor.jsonl"
    state = tmp_path / "bridge-state.json"
    source.write_text(
        json.dumps({
            "device_id": "smoke-01",
            "source": "sensor",
            "observed_at": 1770000000,
            "sensor": {"temperature_c": 72, "smoke_level": 0.9},
            "location": "power-room",
            "image_path": "artifacts/evidence/smoke.jpg",
        })
        + "\n",
        encoding="utf-8",
    )
    service = FieldOperationsService(
        config={
            "archive_path": str(tmp_path / "events.jsonl"),
            "require_trusted_devices": True,
            "max_input_age_s": 999999999,
            "device_registry": {
                "smoke-01": {
                    "allowed_sources": ["sensor"],
                    "hmac_secret": "device-secret",
                    "require_signature": True,
                }
            },
        }
    )

    def post_to_service(_url: str, body: dict[str, object], _timeout_s: float) -> dict[str, object]:
        return asyncio.run(service.ingest_payload(dict(body)))

    payload = run_field_ingest_bridge_once(
        source=source,
        server="http://runtime.local:8765/",
        state_path=state,
        dry_run=False,
        device_secrets={"smoke-01": "device-secret"},
        post_func=post_to_service,
    )

    result = payload["results"][0]
    assert payload["status"] == "ok"
    assert result["posted"] is True
    assert result["device_signing"]["signed"] is True
    assert result["normalized"]["device_signature"]
    assert result["accepted"] is True
    assert result["scenario_id"] == "fire_or_smoke"
    assert payload["summary"]["signed"] == 1
    assert payload["summary"]["accepted"] == 1


def test_field_ingest_bridge_normalizes_site_a_device_fixture(tmp_path: Path) -> None:
    payload = run_field_ingest_bridge_once(
        source=FIXTURES / "site-a-device-events.jsonl",
        server="http://runtime.local:8765",
        state_path=tmp_path / "fixture-state.json",
        dry_run=True,
    )

    assert payload["status"] == "ok"
    assert payload["count"] == 7
    assert payload["summary"]["processed"] == 7
    assert payload["summary"]["source_format"] == "jsonl"
    assert payload["summary"]["source_counts"]["camera"] == 5
    assert payload["summary"]["source_counts"]["sensor"] == 1
    assert payload["summary"]["source_counts"]["robot"] == 1
    assert payload["summary"]["device_counts"]["cam-main-road-01"] == 1
    assert payload["summary"]["device_counts"]["bin-17"] == 1
    normalized = [item["normalized"] for item in payload["results"]]
    assert normalized[0]["plate_number"] == "TEST-A12345"
    assert normalized[1]["sensor"]["smoke_level"] == 1.0
    assert normalized[2]["robot"]["fault_type"] == "joint_motor_fault"
    assert normalized[3]["sensor"]["fill_ratio"] == 91
    assert len(normalized[4]["detections"]) == 6
    assert normalized[5]["zone_type"] == "window"
    assert normalized[6]["help_point_id"] == "guide-01"


@pytest.mark.asyncio
async def test_site_a_device_fixture_drives_field_scenarios(tmp_path: Path) -> None:
    service = FieldOperationsService(
        config={
            "archive_path": str(tmp_path / "field-events.jsonl"),
            "max_input_age_s": 999999999,
            "site_map": {
                "zones": {
                    "main-road-1": {
                        "name": "Main Road",
                        "type": "main_channel",
                        "parking_allowed": False,
                    },
                    "window-corner-1": {
                        "name": "North Window Corner",
                        "type": "window",
                        "parking_allowed": False,
                    },
                    "guide-01": {
                        "name": "Visitor Center Help Point",
                        "type": "help_point",
                        "help_point_id": "guide-01",
                    },
                }
            },
        }
    )
    events = [
        json.loads(line)
        for line in (FIXTURES / "site-a-device-events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    results = [await service.ingest_payload(event) for event in events]
    scenario_ids = {
        result.get("normalized", {}).get("scenario_id")
        for result in results
        if result.get("accepted") is True
    }

    assert scenario_ids == {
        "illegal_parking",
        "fire_or_smoke",
        "robot_abnormal_incident",
        "trash_bin_full",
        "crowd_gathering",
        "night_stranger_photo",
        "wayfinding_help_point",
    }
    assert all(result.get("accepted") is True for result in results)
