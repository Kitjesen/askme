"""Field device adapter tests."""

from __future__ import annotations

import time

import pytest

from askme.pipeline.field_ingest_adapters import normalize_field_ingest_payload
from askme.pipeline.field_operations import FieldOperationsService


@pytest.mark.asyncio
async def test_camera_detection_frame_class_id_triggers_illegal_parking(tmp_path):
    service = FieldOperationsService(
        config={
            "archive_path": str(tmp_path / "field-events.jsonl"),
            "site_map": {
                "zones": {
                    "main-road-1": {
                        "name": "Main road",
                        "type": "main_channel",
                        "parking_allowed": False,
                    },
                }
            },
        },
    )

    result = await service.ingest_payload({
        "source": "camera",
        "timestamp": time.time(),
        "frame_id": 42,
        "zone_id": "main-road-1",
        "detections": [{"class_id": "2", "confidence": 0.93}],
        "duration_s": 180,
        "image_path": "artifacts/evidence/car.jpg",
    })

    assert result["accepted"] is True
    assert result["normalized"]["detections"][0]["label"] == "vehicle"
    assert result["normalized"]["scenario_id"] == "illegal_parking"
    assert result["event"]["location"] == "Main road"


@pytest.mark.asyncio
async def test_sensor_payload_with_nested_temperature_triggers_fire(tmp_path):
    service = FieldOperationsService(config={"archive_path": str(tmp_path / "events.jsonl")})

    result = await service.ingest_payload({
        "device_id": "smoke-01",
        "timestamp": time.time(),
        "sensor": {"temperature_c": 72, "smoke_level": 0.82},
        "location": "Power room",
        "image_path": "artifacts/evidence/smoke.jpg",
    })

    assert result["accepted"] is True
    assert result["normalized"]["source"] == "sensor"
    assert result["normalized"]["scenario_id"] == "fire_or_smoke"
    assert result["event"]["incident_topic"] == "safety.fire_or_smoke"


@pytest.mark.asyncio
async def test_robot_diagnostic_fault_triggers_joint_motor_incident(tmp_path):
    service = FieldOperationsService(config={"archive_path": str(tmp_path / "events.jsonl")})

    result = await service.ingest_payload({
        "topic": "/thunder/diagnostics",
        "timestamp": time.time(),
        "diagnostic": {
            "fault_type": "joint_motor_fault",
            "fault_code": "MOTOR_OVERCURRENT",
            "joint_id": "hip-left",
        },
        "location": "A east road",
    })

    assert result["accepted"] is True
    assert result["normalized"]["source"] == "robot"
    assert result["normalized"]["robot"]["fault_code"] == "MOTOR_OVERCURRENT"
    assert result["event"]["incident_topic"] == "actuator.joint_motor_fault"


def test_map_zone_payload_is_flattened_for_ingest_contract():
    payload = normalize_field_ingest_payload({
        "source": "camera",
        "map_zone": {
            "id": "guide-01",
            "name": "Visitor center help point",
            "type": "help_point",
            "help_point_id": "guide-01",
        },
        "detections": [{"class_id": "person", "confidence": 0.9}],
        "duration_s": 8,
    })

    assert payload["zone_id"] == "guide-01"
    assert payload["zone_type"] == "help_point"
    assert payload["help_point_id"] == "guide-01"
    assert payload["detections"][0]["label"] == "person"


@pytest.mark.asyncio
async def test_ultralytics_boxes_trigger_illegal_parking(tmp_path):
    service = FieldOperationsService(
        config={
            "archive_path": str(tmp_path / "events.jsonl"),
            "site_map": {
                "zones": {
                    "road-1": {"name": "B区主通道", "type": "main_channel", "parking_allowed": False},
                }
            },
        }
    )

    result = await service.ingest_payload({
        "frame": {
            "timestamp": time.time(),
            "boxes": [{"cls": 2, "conf": 0.94, "xyxy": [12, 20, 120, 160]}],
        },
        "zone_id": "road-1",
        "duration_s": 180,
        "image_path": "artifacts/evidence/car.jpg",
    })

    assert result["accepted"] is True
    assert result["normalized"]["detections"][0]["label"] == "vehicle"
    assert result["normalized"]["detections"][0]["confidence"] == 0.94
    assert result["normalized"]["scenario_id"] == "illegal_parking"


@pytest.mark.asyncio
async def test_hikvision_anpr_webhook_triggers_illegal_parking(tmp_path):
    service = FieldOperationsService(
        config={
            "archive_path": str(tmp_path / "events.jsonl"),
            "site_map": {
                "zones": {
                    "main-road-1": {
                        "name": "B区主通道",
                        "type": "main_channel",
                        "parking_allowed": False,
                    },
                }
            },
        }
    )

    result = await service.ingest_payload({
        "eventType": "ANPR",
        "dateTime": time.time(),
        "cameraIndexCode": "cam-main-road-01",
        "ANPR": {"plateNo": "沪A12345"},
        "zone_id": "main-road-1",
        "duration_s": 180,
        "pictureUrl": "artifacts/evidence/anpr-car.jpg",
    })

    assert result["accepted"] is True
    assert result["normalized"]["source"] == "camera"
    assert result["normalized"]["scenario_id"] == "illegal_parking"
    assert result["normalized"]["plate_number"] == "沪A12345"
    assert result["event"]["incident_topic"] == "traffic.illegal_parking"
    assert "沪A12345" in result["event"]["dingtalk"]


@pytest.mark.asyncio
async def test_mqtt_smoke_alarm_payload_triggers_fire_event(tmp_path):
    service = FieldOperationsService(config={"archive_path": str(tmp_path / "events.jsonl")})

    result = await service.ingest_payload({
        "topic": "site/A/power-room/smoke-01",
        "payload": {
            "timestamp": time.time(),
            "temperatureC": 72,
            "smokeAlarm": True,
            "location": "配电间门口",
            "imageUrl": "artifacts/evidence/smoke.jpg",
        },
    })

    assert result["accepted"] is True
    assert result["normalized"]["source"] == "sensor"
    assert result["normalized"]["scenario_id"] == "fire_or_smoke"
    assert result["normalized"]["temperature_c"] == 72
    assert result["event"]["incident_topic"] == "safety.fire_or_smoke"
    assert result["event"]["voice_directive"]["playback_mode"] == "immediate"


@pytest.mark.asyncio
async def test_telemetry_trash_fill_triggers_cleaning_event(tmp_path):
    service = FieldOperationsService(config={"archive_path": str(tmp_path / "events.jsonl")})

    result = await service.ingest_payload({
        "device_id": "bin-17",
        "timestamp": time.time(),
        "telemetry": {"fill_percent": 91},
        "detections": [{"label": "trash_bin", "confidence": 0.88}],
        "bin_id": "bin-17",
        "location": "游客中心门口",
        "image_path": "artifacts/evidence/bin.jpg",
    })

    assert result["accepted"] is True
    assert result["normalized"]["source"] == "camera"
    assert result["normalized"]["scenario_id"] == "trash_bin_full"
    assert result["event"]["notification_group"] == "cleaning"


@pytest.mark.asyncio
async def test_ros_diagnostic_status_triggers_joint_motor_incident(tmp_path):
    service = FieldOperationsService(config={"archive_path": str(tmp_path / "events.jsonl")})

    result = await service.ingest_payload({
        "topic": "/diagnostics",
        "timestamp": time.time(),
        "status": [
            {
                "name": "left_hip_motor",
                "level": 2,
                "message": "motor overcurrent fault",
                "values": [
                    {"key": "joint_id", "value": "hip-left"},
                    {"key": "fault_code", "value": "MOTOR_OVERCURRENT"},
                ],
            }
        ],
        "location": "A区东侧",
    })

    assert result["accepted"] is True
    assert result["normalized"]["source"] == "robot"
    assert result["normalized"]["robot"]["fault_type"] == "joint_motor_fault"
    assert result["event"]["incident_topic"] == "actuator.joint_motor_fault"


@pytest.mark.asyncio
async def test_robot_status_stuck_maps_to_immobilized_incident(tmp_path):
    service = FieldOperationsService(config={"archive_path": str(tmp_path / "events.jsonl")})

    result = await service.ingest_payload({
        "topic": "/thunder/status",
        "timestamp": time.time(),
        "robot": {"nav_state": "stuck", "recoverable": False},
        "location": "A区东侧",
    })

    assert result["accepted"] is True
    assert result["normalized"]["source"] == "robot"
    assert result["normalized"]["fault_type"] == "immobilized"
    assert result["event"]["incident_topic"] == "navigation.immobilized"
    assert result["event"]["voice_directive"]["interrupt_current_speech"] is True


@pytest.mark.asyncio
async def test_multiple_person_detections_trigger_crowd_gathering(tmp_path):
    service = FieldOperationsService(config={"archive_path": str(tmp_path / "events.jsonl")})

    result = await service.ingest_payload({
        "timestamp": time.time(),
        "detections": [{"label": "person", "confidence": 0.81} for _ in range(6)],
        "duration_min": 35,
        "location": "北广场",
        "image_path": "artifacts/evidence/crowd.jpg",
    })

    assert result["accepted"] is True
    assert result["normalized"]["person_count"] == 6
    assert result["normalized"]["scenario_id"] == "crowd_gathering"
