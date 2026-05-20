"""Customer-project executable binding plans for field deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from askme.pipeline.field.customer_project_managed_objects import (
    managed_object_catalog_from_site_profile,
)
from askme.pipeline.field.customer_project_profiles import (
    _customer_payload,
    _normalize_customer_project_profile,
    find_site_profile_path,
)
from askme.pipeline.field.customer_project_scope import _delivery_scope_payload
from askme.pipeline.field.customer_project_template_support import (
    _mapping,
    _string_list,
    load_field_site_profile,
)
from askme.pipeline.field.paths import DEFAULT_DELIVERY_RESOURCE_ROOT
from askme.skills.contracts.field_capability_contracts import field_capability_routes

_SCENARIO_REQUIRED_INPUTS: dict[str, tuple[str, ...]] = {
    "robot_abnormal_incident": ("location", "fault_type"),
    "night_stranger_photo": ("location", "zone_name", "image_path"),
    "illegal_parking": ("location", "zone_name", "image_path"),
    "fire_or_smoke": ("location", "image_path"),
    "trash_bin_full": ("location", "bin_id", "image_path"),
    "urgent_patrol_dispatch": ("target_location", "operator_id"),
    "crowd_gathering": ("location", "person_count", "duration_min", "image_path"),
    "wayfinding_help_point": ("help_point_id", "location"),
    "visitor_escort": ("destination", "location"),
}


def build_customer_project_execution_bindings(
    profile_root: Path,
    identifier: str,
    *,
    delivery_resource_root: Path | None = DEFAULT_DELIVERY_RESOURCE_ROOT,
) -> dict[str, Any]:
    """Return executable ingest/runtime binding plans for one customer project."""
    path = find_site_profile_path(profile_root, identifier)
    if path is None:
        return {"found": False, "reason": "profile_not_found"}
    profile = _normalize_customer_project_profile(load_field_site_profile(path))
    catalog = managed_object_catalog_from_site_profile(
        profile,
        delivery_resource_root=delivery_resource_root,
    )
    devices_by_source = _field_devices_by_source(profile)
    plans = [
        _managed_object_execution_binding_plan(item, profile, devices_by_source)
        for item in catalog.get("objects", [])
        if isinstance(item, dict)
    ]
    summary = _execution_binding_summary(plans)
    return {
        "found": True,
        "profile_path": str(path),
        "project_scope": _delivery_scope_payload(profile),
        "customer": _customer_payload(profile),
        "site": _mapping(profile.get("site")),
        "summary": summary,
        "plans": plans,
        "plans_by_object_id": {str(item.get("object_id") or ""): item for item in plans},
        "customer_claim": _execution_binding_customer_claim(summary),
        "next_step": _execution_binding_next_step(summary),
    }


def _managed_object_execution_binding_plan(
    obj: dict[str, Any],
    profile: dict[str, Any],
    devices_by_source: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    bindings = _mapping(obj.get("bindings"))
    required_sources = _string_list(obj.get("device_sources"))
    checks = _mapping(obj.get("resource_binding_status")).get("checks")
    check_by_key = {
        (
            str(_mapping(item).get("resource_type") or ""),
            str(_mapping(item).get("resource_id") or ""),
        ): _mapping(item)
        for item in (checks if isinstance(checks, list) else [])
    }
    blockers: list[str] = []
    manual_checks: list[str] = []
    source_plans = []
    for source in required_sources:
        matched_devices = devices_by_source.get(source, [])
        if not matched_devices:
            blockers.append(f"No registered {source} device can feed this object.")
        source_plans.append({
            "source": source,
            "status": "ready" if matched_devices else "blocked",
            "device_count": len(matched_devices),
            "devices": matched_devices,
        })

    adapters = []
    covered_sources: set[str] = set()
    for protocol in _string_list(bindings.get("sensor_protocols")):
        sources = _sensor_protocol_execution_sources(protocol)
        matched_adapter_devices = [
            device
            for source in sorted(set(sources).intersection(required_sources))
            for device in devices_by_source.get(source, [])
        ]
        covered_sources.update(source for source in sources if source in required_sources)
        check = check_by_key.get(("sensor_protocols", protocol), {})
        status = _execution_check_status(check, default="ready")
        if status == "blocked":
            blockers.append(f"Sensor protocol {protocol} is blocked.")
        elif status == "manual_check":
            manual_checks.append(f"Sensor protocol {protocol} requires review.")
        adapters.append({
            "protocol_id": protocol,
            "adapter": _sensor_protocol_adapter_name(protocol),
            "sources": sources,
            "matched_required_sources": sorted(set(sources).intersection(required_sources)),
            "status": status,
            "message": str(check.get("message") or ""),
            "adapter_contract": _field_ingest_adapter_contract(
                protocol,
                sources=sources,
                matched_devices=matched_adapter_devices,
            ),
        })
    for source in required_sources:
        if source == "camera" and _string_list(bindings.get("vision_models")):
            covered_sources.add(source)
        if source not in covered_sources:
            manual_checks.append(f"No explicit protocol covers source {source}.")

    vision_models = [
        _execution_resource_ref("vision_models", model_id, check_by_key)
        for model_id in _string_list(bindings.get("vision_models"))
    ]
    scenario_ids = _string_list(obj.get("scenario_ids"))
    scenario_id = next(iter(scenario_ids), "")
    required_inputs = _scenario_required_inputs_for_object(obj)
    skill_routes = []
    for route in field_capability_routes(
        _string_list(bindings.get("skill_packages")),
        scenario_id=scenario_id,
        required_inputs=required_inputs,
    ):
        skill_routes.append({
            **_execution_resource_ref(
                "skill_packages",
                str(route.get("package_id") or ""),
                check_by_key,
            ),
            **route,
            "runtime_route": route.get("route"),
            "safety_boundary": route.get("hardware_boundary"),
        })
    acceptance_tests = [
        _execution_resource_ref("acceptance_tests", ref, check_by_key)
        for ref in _string_list(bindings.get("acceptance_tests"))
    ]
    for bucket_name, bucket in (
        ("vision model", vision_models),
        ("skill package", skill_routes),
        ("acceptance test", acceptance_tests),
    ):
        if not bucket:
            blockers.append(f"No {bucket_name} binding is configured.")
        for item in bucket:
            status = str(item.get("status") or "")
            if status == "blocked":
                blockers.append(f"{bucket_name} {item.get('resource_id')} is blocked.")
            elif status == "manual_check":
                manual_checks.append(f"{bucket_name} {item.get('resource_id')} requires review.")
            if bucket_name == "skill package" and not item.get("installed_contract"):
                manual_checks.append(
                    f"skill package {item.get('resource_id')} has no installed executable contract."
                )

    resource_status = str(_mapping(obj.get("resource_binding_status")).get("overall_status") or "blocked")
    if resource_status == "blocked":
        blockers.append("Resource binding status is blocked.")
    elif resource_status == "manual_check":
        manual_checks.append("Resource binding status requires review.")

    overall_status = "blocked" if blockers else "manual_check" if manual_checks else "ready"
    return {
        "object_id": str(obj.get("object_id") or ""),
        "display_name": str(obj.get("display_name") or obj.get("object_id") or ""),
        "category": str(obj.get("category") or ""),
        "scenario_ids": scenario_ids,
        "overall_status": overall_status,
        "required_sources": required_sources,
        "scope_constraints": {
            "tenant_ids": _string_list(obj.get("tenant_ids")),
            "delivery_namespaces": _string_list(obj.get("delivery_namespaces")),
            "customer_ids": _string_list(obj.get("customer_ids")),
            "project_ids": _string_list(obj.get("project_ids")),
            "site_ids": _string_list(obj.get("site_ids")),
        },
        "source_plans": source_plans,
        "vision_models": vision_models,
        "input_adapters": adapters,
        "skill_routes": skill_routes,
        "acceptance_tests": acceptance_tests,
        "ingest_contract": _managed_object_ingest_contract(obj, profile),
        "runtime_contract": {
            "callback_endpoint": "/api/field/events/{event_id}/runtime-delivery",
            "handoff_boundary": (
                "Field event actions are not direct hardware commands; runtime callbacks must "
                "be signed/trusted before they are recorded."
            ),
        },
        "bridge_contract": _managed_object_bridge_contract(required_sources),
        "blockers": sorted(set(blockers)),
        "manual_checks": sorted(set(manual_checks)),
        "customer_status": {
            "ready": "对象绑定已具备现场接入计划，可进入设备联调和客户试点验证。",
            "manual_check": "对象绑定已有接入计划，但仍需交付复核后才能对客户承诺。",
            "blocked": "对象绑定仍有阻断项，不能作为真实现场接入承诺。",
        }[overall_status],
    }


def _scenario_required_inputs_for_object(obj: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for scenario_id in _string_list(obj.get("scenario_ids")):
        values.extend(_SCENARIO_REQUIRED_INPUTS.get(scenario_id, ()))
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _managed_object_ingest_contract(obj: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    source = next(iter(_string_list(obj.get("device_sources"))), "camera")
    zone_id = _first_zone_for_object(obj, profile)
    sample: dict[str, Any] = {
        "source": source,
        "observed_at": "ISO-8601 or unix timestamp",
        "zone_id": zone_id,
        "scenario_id": next(iter(_string_list(obj.get("scenario_ids"))), ""),
        "managed_object_id": str(obj.get("object_id") or ""),
    }
    labels = _string_list(obj.get("object_labels"))
    if source == "camera":
        sample["detections"] = [{"label": labels[0] if labels else "object", "confidence": 0.9}]
        sample["image_path"] = "artifacts/evidence/example.jpg"
    elif source == "sensor":
        sample["sensor"] = {"temperature_c": 25, "smoke_level": 0.0}
    elif source == "robot":
        sample["robot_id"] = "robot-id"
        sample["runtime_status"] = "reported"
    return {
        "endpoint": "/api/field/ingest",
        "method": "POST",
        "sample_payload": sample,
        "required_fields": ["source", "observed_at", "zone_id"],
        "normalizer": "askme.pipeline.field.field_ingest_adapters.normalize_field_ingest_payload",
        "bridge": "field-ingest-bridge",
    }


def _first_zone_for_object(obj: dict[str, Any], profile: dict[str, Any]) -> str:
    zones = _mapping(profile.get("zones"))
    wanted = set(_string_list(obj.get("zone_types")))
    for zone_id, zone in zones.items():
        if str(_mapping(zone).get("type") or "") in wanted:
            return str(zone_id)
    return ""


def _field_devices_by_source(profile: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    devices: dict[str, list[dict[str, Any]]] = {}
    for device_id, device in _mapping(profile.get("devices")).items():
        if not isinstance(device, dict):
            continue
        source = str(device.get("source") or "").strip()
        if not source:
            continue
        devices.setdefault(source, []).append({
            "device_id": str(device_id),
            "name": str(device.get("name") or device_id),
            "source": source,
            "zone_id": str(device.get("zone_id") or ""),
            "camera_id": str(device.get("camera_id") or ""),
            "sensor_type": str(device.get("sensor_type") or ""),
            "robot_id": str(device.get("robot_id") or ""),
            "secret_configured": bool(device.get("secret_env")),
            "secret_env": str(device.get("secret_env") or ""),
        })
    return devices


def _sensor_protocol_execution_sources(protocol_id: str) -> list[str]:
    text = str(protocol_id or "").lower()
    sources: list[str] = []
    if any(token in text for token in ("camera", "vision", "detection", "video")):
        sources.append("camera")
    if any(token in text for token in ("sensor", "smoke", "temperature", "mqtt", "iot")):
        sources.append("sensor")
    if any(token in text for token in ("robot", "route", "runtime", "status")):
        sources.append("robot")
    if "voice" in text:
        sources.append("robot")
    return sorted(set(sources)) or ["custom"]


def _sensor_protocol_adapter_name(protocol_id: str) -> str:
    sources = _sensor_protocol_execution_sources(protocol_id)
    if sources == ["camera"]:
        return "camera_detection_json"
    if "sensor" in sources:
        return "sensor_telemetry_json"
    if "robot" in sources:
        return "robot_runtime_json"
    return "custom_field_ingest_json"


def _field_ingest_adapter_contract(
    protocol_id: str,
    *,
    sources: list[str],
    matched_devices: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the delivery contract for running a real field ingest adapter."""
    secret_envs = sorted({
        str(device.get("secret_env") or "")
        for device in matched_devices
        if str(device.get("secret_env") or "").strip()
    })
    device_ids = sorted({
        str(device.get("device_id") or "")
        for device in matched_devices
        if str(device.get("device_id") or "").strip()
    })
    return {
        "protocol_id": str(protocol_id or ""),
        "adapter": _sensor_protocol_adapter_name(protocol_id),
        "normalizer": "askme.pipeline.field.field_ingest_adapters.normalize_field_ingest_payload",
        "bridge": "field-ingest-bridge",
        "bridge_runner": "scripts/runtime/bridges/field_ingest_bridge.py",
        "ingest_endpoint": "/api/field/ingest",
        "supported_formats": ["json", "jsonl", "ndjson"],
        "accepted_sources": sorted(set(_string_list(sources))),
        "matched_device_ids": device_ids,
        "device_signature_required": bool(secret_envs),
        "device_secret_envs": secret_envs,
        "dry_run_command": (
            "python -m askme runtime field-ingest-bridge <device-events.jsonl> "
            "--site-profile <site-profile.yaml> --dry-run --json"
        ),
        "live_command": (
            "python -m askme runtime field-ingest-bridge <device-events.jsonl> "
            "--server http://<askme-host>:8765 --site-profile <site-profile.yaml> --watch"
        ),
        "sign_command": (
            "python -m askme runtime field-sign-device-payload <device-event.json> "
            "--secret-env <DEVICE_SECRET_ENV> --output signed-device-event.json"
        ),
        "sample_fixture": "tests/fixtures/field_devices/site-a-device-events.jsonl",
        "verification_outputs": [
            "summary.posted",
            "summary.accepted",
            "summary.failed",
            "summary.signed",
            "results[].normalized.source",
            "results[].normalized.scenario_id",
            "results[].device_signing.reason",
            "results[].event.event_id",
        ],
        "customer_boundary": (
            "Dry-run proves parsing only. Customer signoff needs live post evidence, "
            "trusted device signatures, event archive entries, and runtime callback evidence."
        ),
    }


def _managed_object_bridge_contract(required_sources: list[str]) -> dict[str, Any]:
    return {
        "bridge": "field-ingest-bridge",
        "ingest_endpoint": "/api/field/ingest",
        "sources": sorted(set(_string_list(required_sources))),
        "dry_run_first": True,
        "live_post_required_for_customer_signoff": True,
        "trusted_device_signature_required_for_production": True,
        "state_file": "<bridge-state.json>",
        "sample_fixture": "tests/fixtures/field_devices/site-a-device-events.jsonl",
        "summary_fields": [
            "processed",
            "posted",
            "accepted",
            "failed",
            "signed",
            "events_created",
            "scenario_counts",
            "source_counts",
            "device_counts",
        ],
    }


def _execution_resource_ref(
    resource_type: str,
    resource_id: str,
    checks: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    check = checks.get((resource_type, resource_id), {})
    return {
        "resource_type": resource_type,
        "resource_id": resource_id,
        "status": _execution_check_status(check, default="ready"),
        "publish_status": str(check.get("publish_status") or ""),
        "message": str(check.get("message") or ""),
        "reference_status": str(check.get("reference_status") or ""),
    }


def _execution_check_status(check: dict[str, Any], *, default: str) -> str:
    status = str(check.get("status") or default)
    if status in {"linked", "ready", "configured"}:
        return "ready"
    if status in {"manual_check", "unregistered", "node_unresolved", "draft", "pilot", "deprecated"}:
        return "manual_check"
    return "blocked"


def _execution_binding_summary(plans: list[dict[str, Any]]) -> dict[str, Any]:
    ready = len([item for item in plans if item.get("overall_status") == "ready"])
    manual = len([item for item in plans if item.get("overall_status") == "manual_check"])
    blocked = len([item for item in plans if item.get("overall_status") == "blocked"])
    overall = "blocked" if blocked else "manual_check" if manual else "ready" if plans else "blocked"
    return {
        "overall_status": overall,
        "object_count": len(plans),
        "ready_object_count": ready,
        "manual_check_object_count": manual,
        "blocked_object_count": blocked,
    }


def _execution_binding_customer_claim(summary: dict[str, Any]) -> str:
    status = str(summary.get("overall_status") or "blocked")
    if status == "ready":
        return "客户项目对象绑定已形成可执行接入计划，可进入现场接入验证。"
    if status == "manual_check":
        return "客户项目对象绑定已有接入计划，但仍需交付复核后才能对客户承诺。"
    return "客户项目对象绑定仍有阻断项，不能承诺真实现场接入。"


def _execution_binding_next_step(summary: dict[str, Any]) -> str:
    status = str(summary.get("overall_status") or "blocked")
    if status == "ready":
        return "使用每个对象的 ingest 示例接入真实设备 payload，并记录 runtime 回传。"
    if status == "manual_check":
        return "复核未覆盖的输入源、试点资源和验收用例后再接入现场。"
    return "先补齐缺失设备、资源注册、协议绑定或技能包绑定。"


__all__ = [
    "build_customer_project_execution_bindings",
    "_SCENARIO_REQUIRED_INPUTS",
    "_execution_binding_customer_claim",
    "_execution_binding_next_step",
    "_execution_binding_summary",
    "_execution_check_status",
    "_execution_resource_ref",
    "_field_devices_by_source",
    "_field_ingest_adapter_contract",
    "_first_zone_for_object",
    "_managed_object_bridge_contract",
    "_managed_object_execution_binding_plan",
    "_managed_object_ingest_contract",
    "_scenario_required_inputs_for_object",
    "_sensor_protocol_adapter_name",
    "_sensor_protocol_execution_sources",
]
