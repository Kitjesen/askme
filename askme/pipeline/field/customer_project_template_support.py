"""Leaf support helpers for customer-project templates.

This module intentionally has no dependency on ``field_site_profile`` or the
template release/catalog facades. It keeps pure parsing, path, hashing, and
release metadata utilities out of the large compatibility module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DELIVERY_NAMESPACE = "default"
TEMPLATE_PUBLISH_STATUSES = {"draft", "pilot", "published", "deprecated", "blocked"}
TEMPLATE_RELEASE_REQUEST_STATUSES = {"pending", "approved", "rejected", "cancelled"}
TEMPLATE_RELEASE_FIELDS = (
    "version",
    "publish_status",
    "release_channel",
    "owner",
    "upgrade_policy",
    "min_runtime_version",
    "release_note",
)


def load_field_site_profile(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("site profile root must be a mapping")
    return payload


def site_profile_env_references(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Return deployment environment variables referenced by a field site profile."""
    responders = _mapping(profile.get("responder_groups"))
    devices = _mapping(profile.get("devices"))
    refs: list[dict[str, Any]] = []
    for group, responder in sorted(responders.items()):
        if not isinstance(responder, dict):
            continue
        group_name = str(responder.get("name") or group)
        refs.extend(
            [
                _env_reference(
                    responder.get("webhook_env"),
                    category="dingtalk_webhook",
                    reference=f"responder_groups.{group}.webhook_env",
                    owner=str(group),
                    owner_label=group_name,
                    purpose=f"{group_name} DingTalk robot webhook URL",
                ),
                _env_reference(
                    responder.get("secret_env"),
                    category="dingtalk_secret",
                    reference=f"responder_groups.{group}.secret_env",
                    owner=str(group),
                    owner_label=group_name,
                    purpose=f"{group_name} DingTalk robot signing secret",
                ),
            ]
        )
    for device_id, device in sorted(devices.items()):
        if not isinstance(device, dict):
            continue
        device_name = str(device.get("name") or device_id)
        refs.append(
            _env_reference(
                device.get("secret_env"),
                category="field_device_secret",
                reference=f"devices.{device_id}.secret_env",
                owner=str(device_id),
                owner_label=device_name,
                purpose=f"{device_name} field-device HMAC ingest secret",
                metadata={
                    "source": str(device.get("source") or ""),
                    "zone_id": str(device.get("zone_id") or ""),
                },
            )
        )
    return _dedupe_env_references(refs)


def _site_profile_paths(root: Path, *, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root]
    paths = list(root.rglob(pattern))
    if pattern != "*.yaml":
        return [path for path in paths if path.is_file()]
    paths.extend(root.rglob("*.yml"))
    return sorted({path.resolve() for path in paths if path.is_file()})


def _find_template_path(root: Path, template_id: str) -> Path | None:
    target = str(template_id or "").strip()
    if not target:
        return None
    for path in _site_profile_paths(Path(root), pattern="*.yaml"):
        try:
            profile = load_field_site_profile(path)
        except Exception:
            continue
        template = _mapping(profile.get("template"))
        if target in {str(template.get("template_id") or ""), path.stem}:
            return path
    return None


def _delivery_tenant_id(customer: dict[str, Any]) -> str:
    return _non_empty_text(
        customer.get("tenant_id")
        or customer.get("organization_id")
        or customer.get("org_id")
        or DEFAULT_DELIVERY_NAMESPACE
    )


def _delivery_namespace(customer: dict[str, Any]) -> str:
    return _non_empty_text(
        customer.get("delivery_namespace")
        or customer.get("tenant_namespace")
        or customer.get("namespace")
        or _delivery_tenant_id(customer)
    )


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "").strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item or "").strip()]
    if value in (None, ""):
        return []
    return [str(value)]


def _is_semver(value: str) -> bool:
    return bool(re.match(r"^\d+\.\d+\.\d+(?:[-+][A-Za-z0-9_.-]+)?$", str(value or "")))


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _clean_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): value
        for key, value in payload.items()
        if value not in (None, "")
    }


def _clean_nested_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if value in (None, ""):
            continue
        if isinstance(value, dict):
            result[str(key)] = _clean_nested_mapping(value)
        elif isinstance(value, list):
            result[str(key)] = [item for item in value if item not in (None, "")]
        else:
            result[str(key)] = value
    return result


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _non_empty_text(value: Any) -> str:
    text = str(value or "").strip()
    return text or DEFAULT_DELIVERY_NAMESPACE


def _slug(value: Any) -> str:
    text = str(value or "item").strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = text.strip(".-_")
    return text or "item"


def _env_reference(
    env_name: Any,
    *,
    category: str,
    reference: str,
    owner: str,
    owner_label: str,
    purpose: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    name = str(env_name or "").strip()
    return {
        "env_name": name,
        "category": category,
        "reference": reference,
        "owner": owner,
        "owner_label": owner_label,
        "purpose": purpose,
        "required": True,
        "configured": bool(name and os.getenv(name)),
        "metadata": metadata or {},
    }


def _dedupe_env_references(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    by_env: dict[str, dict[str, Any]] = {}
    for ref in refs:
        env_name = str(ref.get("env_name") or "").strip()
        if not env_name:
            continue
        existing = by_env.get(env_name)
        if existing is None:
            clone = dict(ref)
            clone["references"] = [str(ref.get("reference") or "")]
            by_env[env_name] = clone
            deduped.append(clone)
            continue
        references = existing.setdefault("references", [])
        reference = str(ref.get("reference") or "")
        if reference and reference not in references:
            references.append(reference)
        if not existing.get("configured") and ref.get("configured"):
            existing["configured"] = True
    return deduped


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


__all__ = [
    "DEFAULT_DELIVERY_NAMESPACE",
    "TEMPLATE_PUBLISH_STATUSES",
    "TEMPLATE_RELEASE_FIELDS",
    "TEMPLATE_RELEASE_REQUEST_STATUSES",
    "load_field_site_profile",
    "site_profile_env_references",
]
