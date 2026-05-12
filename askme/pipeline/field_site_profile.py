"""Field site profile validation and conversion.

A site profile is the customer-facing deployment contract for field operations:
zones, help points, parking policy, responder groups, devices, and thresholds.
It lets a deployment team change the park map without editing Python code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

REQUIRED_RESPONDER_GROUPS = ("security", "cleaning", "operations")
REQUIRED_DEVICE_SOURCES = ("camera", "sensor", "robot")


def load_field_site_profile(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("site profile root must be a mapping")
    return payload


def build_site_profile_report(path: Path, *, check_env: bool = False) -> dict[str, Any]:
    profile = load_field_site_profile(path)
    report = validate_field_site_profile(profile, check_env=check_env)
    report["profile_path"] = str(path)
    if report["status"] == "passed":
        report["field_operations_config"] = field_operations_config_from_site_profile(profile)
    return report


def validate_field_site_profile(profile: dict[str, Any], *, check_env: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    site = _mapping(profile.get("site"))
    zones = _mapping(profile.get("zones"))
    responders = _mapping(profile.get("responder_groups"))
    devices = _mapping(profile.get("devices"))
    thresholds = _mapping(profile.get("thresholds"))

    if not site.get("site_id"):
        errors.append("site.site_id is required")
    if not site.get("name"):
        errors.append("site.name is required")
    if not zones:
        errors.append("zones must contain at least one zone")
    if not devices:
        errors.append("devices must contain at least one registered device")

    main_channels = _zones_by_type(zones, "main_channel")
    help_points = _zones_by_type(zones, "help_point")
    parking_restricted = [
        zone_id for zone_id, zone in main_channels.items() if zone.get("parking_allowed") is False
    ]
    if not main_channels:
        errors.append("zones must include at least one main_channel")
    if not parking_restricted:
        errors.append("at least one main_channel must set parking_allowed=false")
    if not help_points:
        errors.append("zones must include at least one help_point")
    for zone_id, zone in help_points.items():
        if not zone.get("help_point_id"):
            errors.append(f"zones.{zone_id}.help_point_id is required for help_point zones")
        if not zone.get("location"):
            errors.append(f"zones.{zone_id}.location is required for help_point zones")

    for group in REQUIRED_RESPONDER_GROUPS:
        responder = _mapping(responders.get(group))
        if not responder:
            errors.append(f"responder_groups.{group} is required")
            continue
        _require_env_reference(
            responder,
            key="webhook_env",
            errors=errors,
            warnings=warnings,
            label=f"responder_groups.{group}.webhook_env",
            check_env=check_env,
        )
        _require_env_reference(
            responder,
            key="secret_env",
            errors=errors,
            warnings=warnings,
            label=f"responder_groups.{group}.secret_env",
            check_env=check_env,
        )

    source_counts: dict[str, int] = {}
    zone_ids = set(zones)
    for device_id, device in devices.items():
        if not isinstance(device, dict):
            errors.append(f"devices.{device_id} must be a mapping")
            continue
        source = str(device.get("source") or "")
        if not source:
            errors.append(f"devices.{device_id}.source is required")
        source_counts[source] = source_counts.get(source, 0) + 1
        zone_id = str(device.get("zone_id") or "")
        if zone_id and zone_id not in zone_ids:
            errors.append(f"devices.{device_id}.zone_id references unknown zone {zone_id}")
        _require_env_reference(
            device,
            key="secret_env",
            errors=errors,
            warnings=warnings,
            label=f"devices.{device_id}.secret_env",
            check_env=check_env,
        )
    for source in REQUIRED_DEVICE_SOURCES:
        if source_counts.get(source, 0) <= 0:
            errors.append(f"devices must include at least one {source} device")

    _validate_thresholds(thresholds, errors)
    status = "passed" if not errors else "failed"
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "site_id": str(site.get("site_id") or ""),
            "site_name": str(site.get("name") or ""),
            "map_version": str(site.get("map_version") or ""),
            "zone_count": len(zones),
            "main_channel_count": len(main_channels),
            "parking_restricted_count": len(parking_restricted),
            "help_point_count": len(help_points),
            "device_count": len(devices),
            "device_sources": source_counts,
            "responder_groups": sorted(responders),
        },
        "readiness": {
            "map_configured": bool(zones),
            "parking_policy_configured": bool(parking_restricted),
            "wayfinding_configured": bool(help_points),
            "responder_groups_configured": all(group in responders for group in REQUIRED_RESPONDER_GROUPS),
            "device_registry_configured": bool(devices),
        },
    }


def field_operations_config_from_site_profile(profile: dict[str, Any]) -> dict[str, Any]:
    zones = _mapping(profile.get("zones"))
    responders = _mapping(profile.get("responder_groups"))
    devices = _mapping(profile.get("devices"))
    thresholds = _mapping(profile.get("thresholds"))
    site = _mapping(profile.get("site"))
    config = {
        "site_id": site.get("site_id"),
        "site_name": site.get("name"),
        "site_map": {"zones": zones},
        "device_registry": {
            device_id: _device_registry_entry(device)
            for device_id, device in devices.items()
            if isinstance(device, dict)
        },
        "dingtalk_webhooks": {
            group: _env_placeholder(responder.get("webhook_env"))
            for group, responder in responders.items()
            if isinstance(responder, dict)
        },
        "dingtalk_secrets": {
            group: _env_placeholder(responder.get("secret_env"))
            for group, responder in responders.items()
            if isinstance(responder, dict)
        },
        "thresholds": thresholds,
    }
    config.update(_field_threshold_config(thresholds))
    return config


def _device_registry_entry(device: dict[str, Any]) -> dict[str, Any]:
    entry = {
        "source": device.get("source"),
        "zone_id": device.get("zone_id"),
        "secret": _env_placeholder(device.get("secret_env")),
    }
    for key in ("name", "camera_id", "sensor_type", "robot_id"):
        if device.get(key):
            entry[key] = device.get(key)
    return {key: value for key, value in entry.items() if value not in ("", None)}


def _field_threshold_config(thresholds: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "parking_duration_s": "parking_duration_s",
        "night_stranger_dwell_s": "night_stranger_dwell_s",
        "fire_temperature_c": "fire_temperature_c",
        "smoke_level": "smoke_level",
        "trash_fill_ratio": "trash_fill_ratio",
        "crowd_person_count": "crowd_person_count",
        "crowd_duration_min": "crowd_duration_min",
    }
    return {
        target: thresholds[source]
        for source, target in mapping.items()
        if source in thresholds
    }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _zones_by_type(zones: dict[str, Any], zone_type: str) -> dict[str, dict[str, Any]]:
    return {
        zone_id: zone
        for zone_id, zone in zones.items()
        if isinstance(zone, dict) and str(zone.get("type") or "") == zone_type
    }


def _require_env_reference(
    item: dict[str, Any],
    *,
    key: str,
    errors: list[str],
    warnings: list[str],
    label: str,
    check_env: bool,
) -> None:
    env_name = str(item.get(key) or "")
    if not env_name:
        errors.append(f"{label} is required")
        return
    if check_env and not os.getenv(env_name):
        warnings.append(f"{label} references unset environment variable {env_name}")


def _validate_thresholds(thresholds: dict[str, Any], errors: list[str]) -> None:
    required = {
        "parking_duration_s": 1,
        "night_stranger_dwell_s": 1,
        "fire_temperature_c": 1,
        "smoke_level": 0,
        "trash_fill_ratio": 0,
        "crowd_person_count": 1,
        "crowd_duration_min": 1,
    }
    for key, minimum in required.items():
        raw = thresholds.get(key)
        if raw is None:
            errors.append(f"thresholds.{key} is required")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"thresholds.{key} must be numeric")
            continue
        if value < minimum:
            errors.append(f"thresholds.{key} must be >= {minimum}")


def _env_placeholder(env_name: Any) -> str:
    return f"${{{env_name}}}" if env_name else ""
