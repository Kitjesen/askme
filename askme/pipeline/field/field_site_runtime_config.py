"""Runtime config and environment handoff renderers for field site profiles."""

from __future__ import annotations

from typing import Any

from askme.pipeline.field.customer_project_managed_objects import (
    managed_object_catalog_from_site_profile,
)
from askme.pipeline.field.customer_project_profiles import _customer_payload
from askme.pipeline.field.customer_project_template_support import (
    _mapping,
    site_profile_env_references,
)


def field_operations_config_from_site_profile(profile: dict[str, Any]) -> dict[str, Any]:
    zones = _mapping(profile.get("zones"))
    responders = _mapping(profile.get("responder_groups"))
    devices = _mapping(profile.get("devices"))
    thresholds = _mapping(profile.get("thresholds"))
    site = _mapping(profile.get("site"))
    customer = _customer_payload(profile)
    managed_objects = managed_object_catalog_from_site_profile(profile)
    config = {
        "customer_project": customer | {"site_id": site.get("site_id"), "site_name": site.get("name")},
        "customer_id": customer.get("customer_id"),
        "project_id": customer.get("project_id"),
        "industry": customer.get("industry"),
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
        "managed_objects": managed_objects["objects_by_id"],
    }
    config.update(_field_threshold_config(thresholds))
    return config


def render_site_profile_env_template(
    profile: dict[str, Any],
    *,
    include_comments: bool = True,
) -> str:
    """Render a deterministic .env template for site deployment handoff."""
    refs = site_profile_env_references(profile)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ref in refs:
        grouped.setdefault(str(ref.get("category") or "other"), []).append(ref)

    lines: list[str] = []
    site = _mapping(profile.get("site"))
    if include_comments:
        lines.extend(
            [
                "# AskMe field site environment template",
                f"# Site: {site.get('site_id') or '-'} / {site.get('name') or '-'}",
                "# Fill values before running field-readiness --check-site-env.",
                "",
            ]
        )
    for category, title in (
        ("dingtalk_webhook", "DingTalk responder webhooks"),
        ("dingtalk_secret", "DingTalk responder signing secrets"),
        ("field_device_secret", "Field device HMAC ingest secrets"),
        ("other", "Other site secrets"),
    ):
        items = grouped.get(category) or []
        if not items:
            continue
        if include_comments:
            lines.append(f"# {title}")
        for item in items:
            env_name = str(item.get("env_name") or "").strip()
            if not env_name:
                continue
            if include_comments:
                lines.append(f"# {item.get('reference')}: {item.get('purpose')}")
            lines.append(f"{env_name}=")
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


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


def _env_placeholder(env_name: Any) -> str:
    return f"${{{env_name}}}" if env_name else ""


__all__ = [
    "field_operations_config_from_site_profile",
    "render_site_profile_env_template",
    "_device_registry_entry",
    "_env_placeholder",
    "_field_threshold_config",
]
