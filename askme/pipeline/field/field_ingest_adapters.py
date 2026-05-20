"""Adapters from real field-device payloads into askme field-ingest contracts.

The field operations service owns business rules, alerting, dedupe, and
archive. This module only translates device/event shapes into the stable
``/api/field/ingest`` payload.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

_CLASS_LABELS = {
    "0": "person",
    "1": "bicycle",
    "2": "vehicle",
    "3": "motorcycle",
    "5": "bus",
    "7": "truck",
    "person": "person",
    "human": "person",
    "pedestrian": "person",
    "行人": "person",
    "人员": "person",
    "人": "person",
    "陌生人": "person",
    "car": "vehicle",
    "vehicle": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "van": "vehicle",
    "motorcycle": "vehicle",
    "汽车": "vehicle",
    "车辆": "vehicle",
    "货车": "vehicle",
    "客车": "vehicle",
    "fire": "fire",
    "flame": "fire",
    "火灾": "fire",
    "火焰": "fire",
    "明火": "fire",
    "smoke": "smoke",
    "烟雾": "smoke",
    "烟": "smoke",
    "trash": "trash",
    "trash_bin": "trash_bin",
    "bin": "trash_bin",
    "garbage_bin": "trash_bin",
    "垃圾": "trash",
    "垃圾桶": "trash_bin",
    "满溢垃圾桶": "trash_bin",
    "phone": "phone",
    "cell_phone": "phone",
    "cell phone": "phone",
    "mobile_phone": "phone",
    "mobile phone": "phone",
    "camera": "camera",
    "手机": "phone",
    "相机": "camera",
    "拍照": "taking_photo",
    "taking_photo": "taking_photo",
}


def normalize_field_ingest_payload(body: dict[str, Any]) -> dict[str, Any]:
    """Return a field-ingest payload using askme's stable raw event contract."""
    payload = dict(body)
    payload = _flatten_device_envelope(payload)
    payload = _normalize_zone(payload)
    payload = _normalize_camera(payload)
    payload = _normalize_sensor(payload)
    payload = _normalize_robot(payload)
    return payload


def _normalize_camera(payload: dict[str, Any]) -> dict[str, Any]:
    payload = _normalize_camera_vendor_fields(payload)
    detections = _extract_detections(payload)
    if not detections and isinstance(payload.get("frame"), dict):
        frame = payload["frame"]
        detections = _extract_detections(frame)
        payload.setdefault("timestamp", frame.get("timestamp"))
        payload.setdefault("frame_id", frame.get("frame_id"))
    if not detections and isinstance(payload.get("result"), dict):
        detections = _extract_detections(payload["result"])
    if not detections and isinstance(payload.get("results"), dict):
        detections = _extract_detections(payload["results"])
    if not detections:
        return payload

    normalized = dict(payload)
    normalized.setdefault("source", "camera")
    normalized.setdefault("observed_at", _first_present(payload, "observed_at", "timestamp", "_ts"))
    normalized["detections"] = [_normalize_detection(item) for item in detections]
    if "camera_id" not in normalized:
        normalized["camera_id"] = (
            payload.get("camera_id")
            or payload.get("device_id")
            or payload.get("sensor_id")
            or payload.get("frame_id")
            or ""
        )
    return normalized


def _flatten_device_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten common webhook/MQTT envelopes without losing the raw body.

    Real camera platforms and IoT gateways often wrap the useful event in
    ``payload``, ``event``, ``message``, or ``params``. The business layer should
    receive the normalized facts, while the full raw envelope remains in
    ``raw_device_payload`` for audit.
    """

    for key in ("payload", "event", "message", "params"):
        nested = payload.get(key)
        if not isinstance(nested, dict):
            continue
        flattened = dict(nested)
        for parent_key, value in payload.items():
            flattened.setdefault(parent_key, value)
        flattened.setdefault("raw_device_payload", payload)
        return flattened
    return payload


def _normalize_camera_vendor_fields(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    event_type = str(
        _first_present(
            result,
            "event_type",
            "eventType",
            "event_name",
            "eventName",
            "alarm_type",
            "alarmType",
            "eventDescription",
        )
        or ""
    ).strip()
    lowered_event = event_type.lower()
    _copy_alias_value(result, field="camera_id", aliases=("camera_id", "cameraId", "channelID", "channel_id", "cameraIndexCode", "device_id", "deviceId"))
    _copy_alias_value(result, field="observed_at", aliases=("observed_at", "dateTime", "datetime", "eventTime", "event_time", "triggerTime", "timestamp"))
    _copy_alias_value(result, field="image_path", aliases=("image_path", "image_url", "imageUrl", "pic_url", "picUrl", "picture", "pictureUrl", "snapshot_url", "snapshotUrl"))
    _copy_alias_value(result, field="location", aliases=("location", "place", "placeName", "site", "address"))
    _copy_alias_value(result, field="bin_id", aliases=("bin_id", "binId", "trashBinId", "garbageBinId"))
    _copy_alias_value(result, field="fill_ratio", aliases=("fill_ratio", "fillRatio", "fill_percent", "fillPercent", "fullness"))
    _copy_alias_value(result, field="person_count", aliases=("person_count", "personCount", "people_count", "peopleCount", "count"))

    for nested_key in ("ANPR", "anpr", "vehicle", "Vehicle", "licensePlate", "plate"):
        nested = result.get(nested_key)
        if isinstance(nested, dict):
            _copy_alias_value(nested, field="plate_number", aliases=("plate_number", "plateNo", "plate_no", "license", "licensePlate", "number"))
            if nested.get("plate_number") and not result.get("plate_number"):
                result["plate_number"] = nested["plate_number"]
            if not result.get("image_path"):
                _copy_alias_value(nested, field="image_path", aliases=("image_path", "imageUrl", "picUrl", "pictureUrl"))
                if nested.get("image_path"):
                    result["image_path"] = nested["image_path"]

    if not result.get("plate_number"):
        _copy_alias_value(result, field="plate_number", aliases=("plate_number", "plateNo", "plate_no", "license", "licensePlate"))

    vehicle_event = any(
        token in lowered_event
        for token in ("vehicle", "parking", "illegal", "anpr", "license", "plate", "车辆", "违停", "停车", "车牌")
    )
    smoke_event = any(token in lowered_event for token in ("smoke", "fire", "flame", "烟", "火", "火灾", "烟雾"))
    person_event = any(
        token in lowered_event
        for token in ("person", "human", "intrusion", "loiter", "stranger", "人员", "行人", "陌生人", "徘徊")
    )
    trash_event = any(token in lowered_event for token in ("trash", "garbage", "bin", "垃圾", "垃圾桶", "满溢", "满载"))
    crowd_event = any(token in lowered_event for token in ("crowd", "gather", "聚集", "人群", "多人"))
    photo_event = any(
        token in lowered_event
        for token in ("photo", "photograph", "phone", "camera", "snapshot", "taking_photo", "拍照", "手机", "相机")
    )
    if result.get("plate_number") or vehicle_event:
        _append_detection(result, {"label": "vehicle", "confidence": _first_present(result, "confidence", "score") or 0.9})
    if smoke_event:
        _append_detection(result, {"label": "smoke", "confidence": _first_present(result, "confidence", "score") or 0.9})
    if person_event:
        _append_detection(result, {"label": "person", "confidence": _first_present(result, "confidence", "score") or 0.85})
    if crowd_event:
        _append_detection(result, {"label": "person", "confidence": _first_present(result, "confidence", "score") or 0.85})
    if trash_event:
        result.setdefault("fill_ratio", 1.0)
        _append_detection(result, {"label": "trash_bin", "confidence": _first_present(result, "confidence", "score") or 0.85})
    if photo_event:
        result["taking_photo"] = True
        _append_detection(result, {"label": "phone", "confidence": _first_present(result, "confidence", "score") or 0.75})
    return result


def _extract_detections(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract common detector outputs into detection dictionaries.

    Supports native Askme payloads plus common camera/VLM exports:
    Ultralytics/YOLO style ``boxes`` with ``cls``/``conf``, Roboflow style
    ``predictions``, and generic ``objects`` or ``items`` lists.
    """
    for key in ("detections", "predictions", "objects", "boxes", "items"):
        detections = _list_of_dicts(payload.get(key))
        if detections:
            return detections
    return []


def _normalize_detection(item: dict[str, Any]) -> dict[str, Any]:
    raw_label = _first_present(item, "label", "class", "class_name", "name", "class_id", "cls", "category")
    label = _CLASS_LABELS.get(str(raw_label).strip().lower(), str(raw_label or "").strip())
    result = dict(item)
    result["label"] = label
    if "confidence" not in result:
        result["confidence"] = _first_present(item, "score", "probability", "conf")
    if "class_id" not in result and "cls" in item:
        result["class_id"] = item.get("cls")
    if "bbox" not in result:
        result["bbox"] = _first_present(item, "bbox", "box", "xyxy")
    return result


def _normalize_sensor(payload: dict[str, Any]) -> dict[str, Any]:
    sensor = dict(payload.get("sensor") or {}) if isinstance(payload.get("sensor"), dict) else {}
    for nested_key in ("telemetry", "values", "data", "properties", "reported"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            sensor.update({key: value for key, value in nested.items() if key not in sensor})
    _copy_alias(sensor, payload, target="temperature_c", aliases=("temperature", "temp_c", "temp", "temperatureC"))
    _copy_alias(sensor, payload, target="smoke_level", aliases=("smoke", "smoke_density", "smoke_ppm", "smokeLevel"))
    _copy_alias(sensor, payload, target="fill_ratio", aliases=("fill", "fill_percent", "fillRatio", "fullness"))
    if _truthy(_first_present(payload, "smoke_detected", "smokeAlarm", "fire_alarm", "fireAlarm")):
        sensor.setdefault("smoke_level", 1.0)
    if _truthy(_first_present(sensor, "smoke_detected", "smokeAlarm", "fire_alarm", "fireAlarm")):
        sensor.setdefault("smoke_level", 1.0)
    has_sensor_fact = any(
        key in payload or key in sensor
        for key in (
            "temperature_c",
            "smoke_level",
            "fill_ratio",
            "sensor_type",
            "sensor_id",
            "humidity",
            "gas_level",
        )
    )
    if not has_sensor_fact:
        return payload
    normalized = dict(payload)
    normalized.setdefault("source", "sensor")
    sensor.setdefault("sensor_id", payload.get("sensor_id") or payload.get("device_id") or "")
    for key in ("temperature_c", "smoke_level", "fill_ratio", "humidity", "gas_level"):
        if key in payload and key not in sensor:
            sensor[key] = payload[key]
    observed = _first_present(payload, "observed_at", "timestamp", "_ts", "source_timestamp")
    if observed is None:
        observed = _first_present(sensor, "observed_at", "timestamp", "source_timestamp")
    normalized.setdefault("observed_at", observed)
    normalized["sensor"] = sensor
    return normalized


def _normalize_robot(payload: dict[str, Any]) -> dict[str, Any]:
    robot = dict(payload.get("robot") or {}) if isinstance(payload.get("robot"), dict) else {}
    fault_type = _first_present(payload, "fault_type", "fault")
    if fault_type is None:
        fault_type = _first_present(robot, "fault_type", "fault")
    if fault_type is None:
        fault_type = _fault_type_from_robot_state(payload, robot)
    diagnostic = payload.get("diagnostic") if isinstance(payload.get("diagnostic"), dict) else {}
    if fault_type is None:
        fault_type = _first_present(diagnostic, "fault_type", "fault")
    diagnostic_status = None
    if fault_type is None:
        diagnostic_status = _first_diagnostic_status(payload)
        if diagnostic_status is not None:
            fault_type = _fault_type_from_diagnostic_status(diagnostic_status)
            robot.setdefault("diagnostic_name", diagnostic_status.get("name") or diagnostic_status.get("hardware_id") or "")
            robot.setdefault("diagnostic_message", diagnostic_status.get("message") or "")
    if fault_type is None and _looks_like_robot_topic(payload):
        fault_type = payload.get("event_type") or payload.get("type")
    if fault_type is None:
        return payload

    normalized = dict(payload)
    normalized.setdefault("source", "robot")
    robot.setdefault("fault_type", fault_type)
    for key in ("fault_code", "joint_id", "joint_name", "diagnostic_code", "severity"):
        value = _first_present(payload, key)
        if value is None:
            value = _first_present(diagnostic, key)
        if value is None and isinstance(diagnostic_status, dict) and isinstance(diagnostic_status.get("values_map"), dict):
            value = _first_present(diagnostic_status["values_map"], key)
        if value is not None and key not in robot:
            robot[key] = value
    normalized.setdefault("observed_at", _first_present(payload, "observed_at", "timestamp", "_ts"))
    normalized["robot"] = robot
    return normalized


def _fault_type_from_robot_state(payload: dict[str, Any], robot: dict[str, Any]) -> str | None:
    text = " ".join(
        str(part or "")
        for part in (
            payload.get("robot_state"),
            payload.get("nav_state"),
            payload.get("motion_state"),
            payload.get("event_type"),
            payload.get("type"),
            robot.get("robot_state"),
            robot.get("nav_state"),
            robot.get("motion_state"),
            robot.get("status"),
            robot.get("state"),
        )
    ).lower()
    fallen = _truthy(_first_present(payload, "is_fallen", "fallen")) or _truthy(_first_present(robot, "is_fallen", "fallen"))
    recoverable = _first_present(payload, "recoverable")
    if recoverable is None:
        recoverable = _first_present(robot, "recoverable")
    blocked_by_human = _truthy(_first_present(payload, "blocked_by_human", "human_blocking")) or _truthy(_first_present(robot, "blocked_by_human", "human_blocking"))
    if fallen or "fall" in text:
        if recoverable is False or str(recoverable).lower() in {"false", "0", "no"} or "unrecoverable" in text:
            return "fall_unrecoverable"
        return "fall_unrecoverable"
    if blocked_by_human or "malicious" in text or "human_block" in text or "恶意挡路" in text or "人为挡路" in text:
        return "malicious_blocking"
    if any(token in text for token in ("immobilized", "stuck", "cannot_move", "blocked", "卡住", "无法运动", "无法移动")):
        return "immobilized"
    if any(token in text for token in ("joint", "motor", "actuator", "overcurrent", "stall", "关节", "电机", "过流")):
        return "joint_motor_fault"
    return None


def _first_diagnostic_status(payload: dict[str, Any]) -> dict[str, Any] | None:
    statuses = _list_of_dicts(payload.get("status"))
    for status in statuses:
        level = _float_or_none(status.get("level"))
        message = str(status.get("message") or "")
        name = str(status.get("name") or "")
        values = _diagnostic_values(status)
        text = " ".join([message, name, " ".join(f"{k}:{v}" for k, v in values.items())]).lower()
        if level is not None and level >= 2:
            return {**status, "values_map": values}
        if any(token in text for token in ("fault", "error", "overcurrent", "stall", "fall", "immobilized")):
            return {**status, "values_map": values}
    return None


def _diagnostic_values(status: dict[str, Any]) -> dict[str, Any]:
    values = status.get("values")
    if isinstance(values, dict):
        return values
    result: dict[str, Any] = {}
    for item in _list_of_dicts(values):
        key = item.get("key") or item.get("name")
        if key not in (None, ""):
            result[str(key)] = item.get("value")
    return result


def _fault_type_from_diagnostic_status(status: dict[str, Any]) -> str:
    text = " ".join(
        str(part or "")
        for part in (
            status.get("name"),
            status.get("message"),
            status.get("hardware_id"),
            (status.get("values_map") or {}).get("fault_type") if isinstance(status.get("values_map"), dict) else "",
            (status.get("values_map") or {}).get("fault") if isinstance(status.get("values_map"), dict) else "",
        )
    ).lower()
    if "fall" in text:
        return "fall_unrecoverable"
    if "immobilized" in text or "stuck" in text:
        return "immobilized"
    if "block" in text:
        return "malicious_blocking"
    if any(token in text for token in ("joint", "motor", "actuator", "overcurrent", "stall")):
        return "joint_motor_fault"
    return "immobilized"


def _normalize_zone(payload: dict[str, Any]) -> dict[str, Any]:
    zone = payload.get("zone")
    if not isinstance(zone, dict):
        zone = payload.get("map_zone")
    if not isinstance(zone, dict):
        return payload
    normalized = dict(payload)
    normalized.setdefault("zone_id", zone.get("zone_id") or zone.get("id"))
    normalized.setdefault("zone_name", zone.get("name") or zone.get("zone_name"))
    normalized.setdefault("location", zone.get("location") or zone.get("name"))
    normalized.setdefault("zone_type", zone.get("type") or zone.get("zone_type"))
    if "parking_allowed" in zone and "parking_allowed" not in normalized:
        normalized["parking_allowed"] = bool(zone.get("parking_allowed"))
    if "help_point_id" in zone and "help_point_id" not in normalized:
        normalized["help_point_id"] = zone.get("help_point_id")
    return normalized


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, dict)):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _first_present(source: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in source and source.get(key) not in (None, ""):
            return source.get(key)
    return None


def _copy_alias(
    sensor: dict[str, Any],
    payload: dict[str, Any],
    *,
    target: str,
    aliases: tuple[str, ...],
) -> None:
    if target in sensor:
        return
    for source in (payload, sensor):
        value = _first_present(source, *aliases)
        if value not in (None, ""):
            sensor[target] = value
            return


def _copy_alias_value(
    values: dict[str, Any],
    *,
    target_key: str | None = None,
    field: str | None = None,
    aliases: tuple[str, ...],
) -> None:
    key = target_key or field
    if not key or values.get(key) not in (None, ""):
        return
    value = _first_present(values, *aliases)
    if value not in (None, ""):
        values[key] = value


def _append_detection(payload: dict[str, Any], detection: dict[str, Any]) -> None:
    detections = payload.get("detections")
    if not isinstance(detections, list):
        detections = []
    label = str(detection.get("label") or "").lower()
    if label and not any(isinstance(item, dict) and str(item.get("label") or "").lower() == label for item in detections):
        detections.append(detection)
    payload["detections"] = detections


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "alarm", "detected"}
    return False


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _looks_like_robot_topic(payload: dict[str, Any]) -> bool:
    text = " ".join(str(payload.get(key) or "") for key in ("topic", "type", "event_type"))
    lowered = text.lower()
    return any(token in lowered for token in ("joint", "motor", "fault", "diagnostic"))


def stamp_observed_at(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with ``observed_at`` set when a device did not provide one."""
    result = dict(payload)
    result.setdefault("observed_at", time.time())
    return result
