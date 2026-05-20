"""Validation rules for customer field site profiles."""

from __future__ import annotations

import os
from typing import Any

from askme.pipeline.field.customer_project_template_support import (
    _delivery_namespace,
    _delivery_tenant_id,
    _mapping,
    _string_list,
)

REQUIRED_RESPONDER_GROUPS = ("security", "cleaning", "operations")
REQUIRED_DEVICE_SOURCES = ("camera", "sensor", "robot")


def validate_field_site_profile(profile: dict[str, Any], *, check_env: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    site = _mapping(profile.get("site"))
    customer = _mapping(profile.get("customer"))
    zones = _mapping(profile.get("zones"))
    responders = _mapping(profile.get("responder_groups"))
    devices = _mapping(profile.get("devices"))
    thresholds = _mapping(profile.get("thresholds"))
    managed_objects = _mapping(profile.get("managed_objects"))

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
    _validate_customer_project(customer, warnings)
    object_summary = _validate_managed_objects(
        managed_objects,
        zones=zones,
        responders=responders,
        source_counts=source_counts,
        errors=errors,
        warnings=warnings,
    )
    status = "passed" if not errors else "failed"
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "tenant_id": _delivery_tenant_id(customer),
            "delivery_namespace": _delivery_namespace(customer),
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
            "customer_id": str(customer.get("customer_id") or ""),
            "customer_name": str(customer.get("customer_name") or ""),
            "industry": str(customer.get("industry") or ""),
            "project_id": str(customer.get("project_id") or ""),
            "project_name": str(customer.get("project_name") or ""),
            "managed_object_type_count": object_summary["object_type_count"],
            "managed_object_categories": object_summary["categories"],
        },
        "readiness": {
            "map_configured": bool(zones),
            "parking_policy_configured": bool(parking_restricted),
            "wayfinding_configured": bool(help_points),
            "responder_groups_configured": all(group in responders for group in REQUIRED_RESPONDER_GROUPS),
            "device_registry_configured": bool(devices),
            "customer_project_configured": bool(customer.get("customer_id") and customer.get("project_id")),
            "managed_objects_configured": object_summary["object_type_count"] > 0,
        },
        "managed_objects_summary": object_summary,
    }


def _validate_customer_project(customer: dict[str, Any], warnings: list[str]) -> None:
    missing = [
        field
        for field in ("customer_id", "customer_name", "industry", "project_id", "project_name")
        if not customer.get(field)
    ]
    if missing:
        warnings.append(
            "customer project metadata is incomplete: "
            + ", ".join(f"customer.{field}" for field in missing)
        )


def _validate_managed_objects(
    managed_objects: dict[str, Any],
    *,
    zones: dict[str, Any],
    responders: dict[str, Any],
    source_counts: dict[str, int],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    if not managed_objects:
        warnings.append("managed_objects is recommended for solution-provider customer projects")
        return {
            "object_type_count": 0,
            "categories": [],
            "scenario_ids": [],
            "customer_visible_count": 0,
        }
    zone_types = {
        str(zone.get("type") or "")
        for zone in zones.values()
        if isinstance(zone, dict) and zone.get("type")
    }
    categories: set[str] = set()
    scenario_ids: set[str] = set()
    skill_packages: set[str] = set()
    acceptance_tests: set[str] = set()
    binding_missing_count = 0
    visible_count = 0
    for object_id, item in managed_objects.items():
        if not isinstance(item, dict):
            errors.append(f"managed_objects.{object_id} must be a mapping")
            continue
        display_name = str(item.get("display_name") or "")
        category = str(item.get("category") or "")
        if not display_name:
            errors.append(f"managed_objects.{object_id}.display_name is required")
        if not category:
            errors.append(f"managed_objects.{object_id}.category is required")
        else:
            categories.add(category)
        object_scenarios = _string_list(item.get("scenario_ids"))
        if not object_scenarios:
            errors.append(f"managed_objects.{object_id}.scenario_ids must contain at least one scenario")
        scenario_ids.update(object_scenarios)
        object_zone_types = _string_list(item.get("zone_types"))
        if not object_zone_types:
            errors.append(f"managed_objects.{object_id}.zone_types must contain at least one zone type")
        unknown_zone_types = sorted(set(object_zone_types) - zone_types)
        if unknown_zone_types:
            errors.append(
                f"managed_objects.{object_id}.zone_types references unknown zone types: "
                + ", ".join(unknown_zone_types)
            )
        object_sources = _string_list(item.get("device_sources"))
        missing_sources = sorted(source for source in object_sources if source_counts.get(source, 0) <= 0)
        if missing_sources:
            warnings.append(
                f"managed_objects.{object_id}.device_sources have no registered device: "
                + ", ".join(missing_sources)
            )
        responder_group = str(item.get("responder_group") or "")
        if responder_group and responder_group not in responders:
            errors.append(
                f"managed_objects.{object_id}.responder_group references unknown responder group {responder_group}"
            )
        if item.get("customer_visible", True):
            visible_count += 1
        bindings = _mapping(item.get("bindings"))
        binding_report = _validate_managed_object_bindings(object_id, bindings, warnings)
        binding_missing_count += binding_report["missing_count"]
        skill_packages.update(binding_report["skill_packages"])
        acceptance_tests.update(binding_report["acceptance_tests"])
    return {
        "object_type_count": len([item for item in managed_objects.values() if isinstance(item, dict)]),
        "categories": sorted(categories),
        "scenario_ids": sorted(scenario_ids),
        "customer_visible_count": visible_count,
        "bound_object_type_count": len(managed_objects) - binding_missing_count,
        "binding_missing_count": binding_missing_count,
        "skill_packages": sorted(skill_packages),
        "acceptance_tests": sorted(acceptance_tests),
    }


def _validate_managed_object_bindings(
    object_id: str,
    bindings: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    required = ("vision_models", "sensor_protocols", "skill_packages", "acceptance_tests")
    missing = [key for key in required if not _string_list(bindings.get(key))]
    if missing:
        warnings.append(
            f"managed_objects.{object_id}.bindings missing product bindings: "
            + ", ".join(missing)
        )
    return {
        "missing_count": 1 if missing else 0,
        "skill_packages": _string_list(bindings.get("skill_packages")),
        "acceptance_tests": _string_list(bindings.get("acceptance_tests")),
    }


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


__all__ = [
    "REQUIRED_DEVICE_SOURCES",
    "REQUIRED_RESPONDER_GROUPS",
    "validate_field_site_profile",
    "_require_env_reference",
    "_validate_customer_project",
    "_validate_managed_object_bindings",
    "_validate_managed_objects",
    "_validate_thresholds",
    "_zones_by_type",
]
