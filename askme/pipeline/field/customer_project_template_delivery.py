"""Customer-project template delivery view helpers.

These helpers build customer-readable template/package summaries, delivery
scope, prerequisites, acceptance criteria, dependency matrices, and rollout
checklists. They intentionally depend only on leaf support utilities and the
delivery-resource registry constants, not on ``field_site_profile``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from askme.pipeline.field.customer_project_template_support import (
    TEMPLATE_PUBLISH_STATUSES,
    _is_semver,
    _mapping,
    _sha256_json,
    _slug,
    _string_list,
    site_profile_env_references,
)
from askme.pipeline.field.delivery_resource_registry import DELIVERY_RESOURCE_TYPES


def _template_delivery_summary(
    *,
    template: dict[str, Any],
    customer: dict[str, Any],
    managed_objects: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    """Summarize one industry template as a reusable delivery product."""
    _ = customer
    objects = [
        item
        for item in managed_objects.get("objects", [])
        if isinstance(item, dict)
    ]
    acceptance = _mapping(managed_objects.get("acceptance_summary"))
    return {
        "template_status": str(report.get("status") or "failed"),
        "template_version": str(template.get("version") or "0.0.0"),
        "publish_status": str(template.get("publish_status") or "draft"),
        "release_channel": str(template.get("release_channel") or template.get("publish_status") or "draft"),
        "customer_fit": str(
            template.get("customer_fit")
            or template.get("customer_claim")
            or "Use when the customer needs this industry scenario as a repeatable starter."
        ),
        "default_object_count": int(managed_objects.get("object_type_count") or len(objects)),
        "default_objects": [
            {
                "object_id": str(item.get("object_id") or ""),
                "display_name": str(item.get("display_name") or item.get("object_id") or ""),
                "category": str(item.get("category") or "uncategorized"),
            }
            for item in objects[:6]
        ],
        "object_categories": _string_list(managed_objects.get("categories")),
        "scenario_ids": _string_list(managed_objects.get("scenario_ids")),
        "device_sources": _unique_template_object_values(objects, "device_sources"),
        "responder_groups": sorted({
            str(item.get("responder_group") or "")
            for item in objects
            if str(item.get("responder_group") or "").strip()
        }),
        "vision_models": _unique_template_binding_values(objects, "vision_models"),
        "sensor_protocols": _unique_template_binding_values(objects, "sensor_protocols"),
        "skill_packages": _unique_template_binding_values(objects, "skill_packages"),
        "acceptance_tests": _unique_template_binding_values(objects, "acceptance_tests"),
        "acceptance_status": str(acceptance.get("overall_status") or "blocked"),
        "ready_object_count": int(acceptance.get("ready_object_count") or 0),
        "manual_check_object_count": int(acceptance.get("manual_check_object_count") or 0),
        "blocked_object_count": int(acceptance.get("blocked_object_count") or 0),
        "delivery_boundary": (
            "Template is a starter package. Delivery must replace customer scope, bind the real map/devices/"
            "credentials, and run onsite acceptance before production claims."
        ),
    }


def _customer_delivery_surface(
    *,
    profile: dict[str, Any],
    template: dict[str, Any],
    customer: dict[str, Any],
    managed_objects: dict[str, Any],
    report: dict[str, Any],
    env_references: list[dict[str, Any]] | None = None,
    surface: str = "project",
) -> dict[str, Any]:
    """Return customer-readable delivery scope, prerequisites, and acceptance criteria."""
    objects = [
        item
        for item in managed_objects.get("objects", [])
        if isinstance(item, dict)
    ]
    return {
        "applicability_scope": _customer_delivery_applicability_scope(
            template=template,
            customer=customer,
            managed_objects=managed_objects,
            surface=surface,
        ),
        "out_of_scope": _customer_delivery_out_of_scope(surface),
        "customer_prerequisites": _customer_delivery_prerequisites(
            profile,
            env_references=env_references,
            report=report,
        ),
        "scenario_acceptance_criteria": _customer_delivery_scenario_acceptance_criteria(objects),
        "dependency_matrix": _customer_delivery_dependency_matrix(objects),
    }


def _customer_delivery_applicability_scope(
    *,
    template: dict[str, Any],
    customer: dict[str, Any],
    managed_objects: dict[str, Any],
    surface: str,
) -> dict[str, Any]:
    industry = str(customer.get("industry") or template.get("industry") or "unspecified")
    categories = _string_list(managed_objects.get("categories"))
    scenarios = _string_list(managed_objects.get("scenario_ids"))
    objects = [
        item
        for item in managed_objects.get("objects", [])
        if isinstance(item, dict)
    ]
    managed_object_types = sorted({
        str(item.get("category") or "uncategorized")
        for item in objects
        if str(item.get("category") or "").strip()
    } or set(categories))
    site_types = {
        "manufacturing": ["factory", "production line", "utility corridor"],
        "creative_park": ["creative park", "visitor service area", "mixed-use campus"],
        "warehouse": ["warehouse", "loading zone", "logistics aisle"],
        "scenic_area": ["scenic area", "visitor route", "service point"],
    }.get(industry, [industry.replace("_", " ") or "customer site"])
    return {
        "scope_type": "askme.customer_delivery_applicability_scope.v1",
        "surface": surface,
        "industries": [industry] if industry else [],
        "site_types": site_types,
        "scenarios": scenarios,
        "managed_object_types": managed_object_types,
        "default_object_count": int(managed_objects.get("object_type_count") or len(objects)),
        "default_objects": [
            {
                "object_id": str(item.get("object_id") or ""),
                "display_name": str(item.get("display_name") or item.get("object_id") or ""),
                "category": str(item.get("category") or "uncategorized"),
            }
            for item in objects[:8]
        ],
        "customer_fit": str(
            template.get("customer_fit")
            or template.get("customer_claim")
            or f"Use this {industry} starter when the customer needs repeatable robot service delivery."
        ),
    }


def _customer_delivery_out_of_scope(surface: str) -> list[str]:
    noun = "template" if surface == "template" else "customer project package"
    return [
        f"This {noun} is not a production go-live certificate.",
        "It does not replace onsite map validation, live device tests, notification tests, or robot runtime acceptance.",
        "It does not prove enterprise IAM/SSO, customer network, or responder credentials are production-ready.",
        "Open-domain chat and unsupervised hardware control are outside this delivery boundary.",
    ]


def _customer_delivery_prerequisites(
    profile: dict[str, Any],
    *,
    env_references: list[dict[str, Any]] | None,
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    site = _mapping(profile.get("site"))
    devices = _mapping(profile.get("devices"))
    responders = _mapping(profile.get("responder_groups"))
    env_refs = env_references if isinstance(env_references, list) else site_profile_env_references(profile)
    required_env_count = len([item for item in env_refs if isinstance(item, dict) and item.get("required")])
    missing_env_count = len([
        item
        for item in env_refs
        if isinstance(item, dict) and item.get("required") and not item.get("configured")
    ])
    validation_status = str(report.get("status") or "failed")
    return [
        {
            "prerequisite_id": "site_map_and_routes",
            "label": "Site map and service routes",
            "owner": "customer operations + delivery engineer",
            "required": True,
            "status": "manual_check" if site.get("map_version") else "blocked",
            "evidence_required": ["map_version", "route list", "no-go zones"],
            "next_step": "Confirm the customer map, route scope, service points, and restricted areas onsite.",
        },
        {
            "prerequisite_id": "field_devices",
            "label": "Cameras, sensors, and robot sources",
            "owner": "delivery engineer",
            "required": True,
            "status": "manual_check" if devices else "blocked",
            "evidence_required": ["device inventory", "source id", "zone binding"],
            "next_step": "Bind every managed object to real camera, sensor, voice, or robot event sources.",
        },
        {
            "prerequisite_id": "credentials_and_notifications",
            "label": "Credentials and responder notification groups",
            "owner": "customer IT + customer operations",
            "required": True,
            "status": (
                "ready"
                if required_env_count and missing_env_count == 0
                else "manual_check"
                if required_env_count
                else "blocked"
            ),
            "evidence_required": ["secret env vars", "DingTalk/WeCom/Feishu test", "responder roster"],
            "next_step": "Configure live credentials and run notification smoke tests before handoff.",
            "required_env_count": required_env_count,
            "missing_env_count": missing_env_count,
            "responder_group_count": len(responders),
        },
        {
            "prerequisite_id": "onsite_acceptance_window",
            "label": "Onsite acceptance owner and test window",
            "owner": "delivery manager + customer signatory",
            "required": True,
            "status": "manual_check" if validation_status == "passed" else "blocked",
            "evidence_required": ["test schedule", "customer reviewer", "signed acceptance dossier"],
            "next_step": "Schedule onsite scenario tests and collect signed customer acceptance evidence.",
        },
    ]


def _customer_delivery_scenario_acceptance_criteria(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for obj in objects:
        object_id = str(obj.get("object_id") or "")
        display_name = str(obj.get("display_name") or object_id)
        evidence = _string_list(obj.get("evidence_required"))
        bindings = _mapping(obj.get("bindings"))
        acceptance_tests = _string_list(bindings.get("acceptance_tests"))
        for scenario_id in _string_list(obj.get("scenario_ids")):
            row = rows.setdefault(
                scenario_id,
                {
                    "scenario_id": scenario_id,
                    "managed_object_ids": [],
                    "managed_object_labels": [],
                    "required_evidence": [],
                    "acceptance_tests": [],
                    "pass_condition": (
                        "Accepted when live event evidence, notification/archive evidence, "
                        "and linked scenario tests are all reviewed for this customer site."
                    ),
                    "blocking_if_missing": False,
                },
            )
            row["managed_object_ids"].append(object_id)
            row["managed_object_labels"].append(display_name)
            row["required_evidence"].extend(evidence)
            row["acceptance_tests"].extend(acceptance_tests)
    result = []
    for row in rows.values():
        row["managed_object_ids"] = sorted(set(row["managed_object_ids"]))
        row["managed_object_labels"] = sorted(set(row["managed_object_labels"]))
        row["required_evidence"] = sorted(set(row["required_evidence"]))
        row["acceptance_tests"] = sorted(set(row["acceptance_tests"]))
        row["blocking_if_missing"] = not row["required_evidence"] or not row["acceptance_tests"]
        result.append(row)
    return sorted(result, key=lambda item: str(item.get("scenario_id") or ""))


def _customer_delivery_dependency_matrix(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for obj in objects:
        object_id = str(obj.get("object_id") or "")
        bindings = _mapping(obj.get("bindings"))
        checks = _mapping(obj.get("resource_binding_status")).get("checks")
        check_by_key = (
            {
                (
                    str(_mapping(item).get("resource_type") or ""),
                    str(_mapping(item).get("resource_id") or ""),
                ): _mapping(item)
                for item in checks
                if isinstance(item, dict)
            }
            if isinstance(checks, list)
            else {}
        )
        for resource_type in DELIVERY_RESOURCE_TYPES:
            for resource_id in _string_list(bindings.get(resource_type)):
                key = (resource_type, resource_id)
                check = check_by_key.get(key, {})
                row = rows.setdefault(
                    key,
                    {
                        "resource_type": resource_type,
                        "resource_id": resource_id,
                        "customer_label": str(check.get("display_name") or resource_id),
                        "status": str(check.get("status") or "manual_check"),
                        "source": str(check.get("source") or ""),
                        "managed_object_ids": [],
                        "blocking_if_missing": str(check.get("status") or "")
                        in {"blocked", "unregistered"},
                    },
                )
                row["managed_object_ids"].append(object_id)
                if str(check.get("status") or "") in {"blocked", "unregistered"}:
                    row["blocking_if_missing"] = True
    result = []
    for row in rows.values():
        row["managed_object_ids"] = sorted(set(row["managed_object_ids"]))
        result.append(row)
    return sorted(
        result,
        key=lambda item: (
            str(item.get("resource_type") or ""),
            str(item.get("resource_id") or ""),
        ),
    )


def _template_package_summary(
    *,
    profile: dict[str, Any],
    template: dict[str, Any],
    path: Path,
    report: dict[str, Any],
    delivery_summary: dict[str, Any],
) -> dict[str, Any]:
    """Return product-release metadata for one reusable industry template."""
    template_id = str(template.get("template_id") or path.stem)
    version = str(template.get("version") or "0.0.0")
    publish_status = str(template.get("publish_status") or "draft")
    release_channel = str(template.get("release_channel") or publish_status)
    blockers: list[str] = []
    manual_checks: list[str] = []
    if not _is_semver(version):
        blockers.append("Template version must use semantic version format.")
    if report.get("status") != "passed":
        blockers.append("Template profile validation is failing.")
    if publish_status not in TEMPLATE_PUBLISH_STATUSES:
        blockers.append("Template publish_status is not recognized.")
    if publish_status in {"draft", "pilot"}:
        manual_checks.append("Template is not marked as published; customer use requires delivery-owner approval.")
    if publish_status == "deprecated":
        manual_checks.append("Template is deprecated; use only for existing customer maintenance.")
    if int(delivery_summary.get("default_object_count") or 0) <= 0:
        blockers.append("Template has no default managed objects.")
    for field, label in (
        ("scenario_ids", "scenario coverage"),
        ("device_sources", "device sources"),
        ("skill_packages", "skill packages"),
        ("acceptance_tests", "acceptance evidence"),
    ):
        if not _string_list(delivery_summary.get(field)):
            blockers.append(f"Template is missing {label}.")
    acceptance_status = str(delivery_summary.get("acceptance_status") or "blocked")
    if acceptance_status == "blocked":
        blockers.append("Template managed-object acceptance is blocked.")
    elif acceptance_status == "manual_check":
        manual_checks.append("Template acceptance references require manual review before signoff.")
    if blockers:
        product_status = "blocked"
    elif manual_checks:
        product_status = "manual_check"
    else:
        product_status = "ready"
    dependencies = {
        "managed_object_count": int(delivery_summary.get("default_object_count") or 0),
        "scenario_count": len(_string_list(delivery_summary.get("scenario_ids"))),
        "device_source_count": len(_string_list(delivery_summary.get("device_sources"))),
        "vision_model_count": len(_string_list(delivery_summary.get("vision_models"))),
        "sensor_protocol_count": len(_string_list(delivery_summary.get("sensor_protocols"))),
        "skill_package_count": len(_string_list(delivery_summary.get("skill_packages"))),
        "acceptance_test_count": len(_string_list(delivery_summary.get("acceptance_tests"))),
    }
    return {
        "package_type": "askme.customer_project_template",
        "package_schema": "askme.customer_project_template.v1",
        "package_id": f"{_slug(template_id)}@{version}",
        "template_id": template_id,
        "version": version,
        "publish_status": publish_status,
        "release_channel": release_channel,
        "owner": str(template.get("owner") or "unassigned"),
        "upgrade_policy": str(template.get("upgrade_policy") or "manual_review"),
        "min_runtime_version": str(template.get("min_runtime_version") or ""),
        "product_status": product_status,
        "customer_status": {
            "ready": "Template is published and ready to seed a customer project after onsite binding.",
            "manual_check": "Template can be used for pilot delivery with product-owner review.",
            "blocked": "Template must be fixed before customer delivery.",
        }[product_status],
        "blocker_count": len(blockers),
        "manual_check_count": len(manual_checks),
        "blockers": blockers,
        "manual_checks": manual_checks,
        "dependencies": dependencies,
        "template_sha256": _sha256_json(profile),
        "source_path": str(path),
        "next_step": {
            "ready": "Create a scoped customer project and run onsite acceptance.",
            "manual_check": "Assign a delivery owner to approve pilot use before creating customer projects.",
            "blocked": "Fix blockers in the template YAML or managed-object bindings.",
        }[product_status],
    }


def _template_runtime_blueprint_binding(
    *,
    template: dict[str, Any],
    customer: dict[str, Any],
    delivery_summary: dict[str, Any],
    blueprints_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a reusable industry template to a customer-visible runtime blueprint."""

    payload = blueprints_payload if isinstance(blueprints_payload, dict) else _load_runtime_blueprints_payload()
    items = _customer_visible_runtime_blueprints(payload)
    selected, match_reason = _select_template_runtime_blueprint(
        template=template,
        customer=customer,
        delivery_summary=delivery_summary,
        items=items,
    )
    missing_fields = _template_runtime_blueprint_missing_fields(delivery_summary)
    status = _template_runtime_blueprint_status(
        selected=selected,
        template=template,
        missing_fields=missing_fields,
    )
    return {
        "binding_type": "askme.customer_project_template.runtime_blueprint_binding.v1",
        "status": status,
        "selected_blueprint": _runtime_blueprint_public_binding(selected),
        "available_customer_blueprint_count": len(items),
        "match_reason": match_reason,
        "missing_template_fields": missing_fields,
        "template_runtime_requirements": {
            "scenario_count": len(_string_list(delivery_summary.get("scenario_ids"))),
            "skill_package_count": len(_string_list(delivery_summary.get("skill_packages"))),
            "device_source_count": len(_string_list(delivery_summary.get("device_sources"))),
            "vision_model_count": len(_string_list(delivery_summary.get("vision_models"))),
            "sensor_protocol_count": len(_string_list(delivery_summary.get("sensor_protocols"))),
            "acceptance_test_count": len(_string_list(delivery_summary.get("acceptance_tests"))),
        },
        "policy": {
            "template_must_bind_runtime_blueprint_before_delivery": True,
            "runtime_blueprint_does_not_replace_onsite_acceptance": True,
            "created_projects_must_recheck_blueprint_binding": True,
        },
        "customer_claim": _template_runtime_blueprint_customer_claim(status, selected),
        "next_step": _template_runtime_blueprint_next_step(
            status=status,
            selected=selected,
            missing_fields=missing_fields,
        ),
    }


def _template_delivery_checklist(delivery_summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return customer-project rollout steps that every template must pass."""
    object_count = int(delivery_summary.get("default_object_count") or 0)
    acceptance_status = str(delivery_summary.get("acceptance_status") or "blocked")
    binding_count = sum(
        len(_string_list(delivery_summary.get(key)))
        for key in ("vision_models", "sensor_protocols", "skill_packages")
    )
    return [
        {
            "step_id": "validate_template",
            "label": "Validate template package",
            "status": "ready" if delivery_summary.get("template_status") == "passed" else "blocked",
            "evidence": str(delivery_summary.get("template_status") or "failed"),
            "next_step": "Fix template YAML before using it for a customer project.",
        },
        {
            "step_id": "review_template_release",
            "label": "Review template release",
            "status": "ready" if delivery_summary.get("publish_status") == "published" else "manual_check",
            "evidence": (
                f"version {delivery_summary.get('template_version') or '0.0.0'} / "
                f"{delivery_summary.get('publish_status') or 'draft'}"
            ),
            "next_step": "Promote the template to published only after pilot acceptance evidence is attached.",
        },
        {
            "step_id": "replace_customer_scope",
            "label": "Replace customer scope",
            "status": "manual_check",
            "evidence": "Set tenant, delivery namespace, customer, project, and site identifiers.",
            "next_step": "Create a scoped customer project from this template.",
        },
        {
            "step_id": "review_managed_objects",
            "label": "Review managed objects",
            "status": "ready" if object_count else "blocked",
            "evidence": f"{object_count} default managed object(s).",
            "next_step": "Remove irrelevant objects and add customer-specific objects.",
        },
        {
            "step_id": "bind_runtime_capabilities",
            "label": "Bind runtime capabilities",
            "status": "manual_check" if binding_count else "blocked",
            "evidence": (
                f"{len(_string_list(delivery_summary.get('vision_models')))} vision model(s), "
                f"{len(_string_list(delivery_summary.get('sensor_protocols')))} sensor protocol(s), "
                f"{len(_string_list(delivery_summary.get('skill_packages')))} skill package(s)."
            ),
            "next_step": "Bind the project to real devices, model versions, protocols, and enabled skill packages.",
        },
        {
            "step_id": "run_acceptance",
            "label": "Run acceptance evidence",
            "status": acceptance_status if acceptance_status in {"ready", "manual_check", "blocked"} else "manual_check",
            "evidence": (
                f"{delivery_summary.get('ready_object_count', 0)} ready / "
                f"{delivery_summary.get('manual_check_object_count', 0)} manual / "
                f"{delivery_summary.get('blocked_object_count', 0)} blocked object(s)."
            ),
            "next_step": "Run repository and onsite acceptance tests before customer signoff.",
        },
        {
            "step_id": "export_handoff_package",
            "label": "Export handoff package",
            "status": "manual_check",
            "evidence": "Export package after scope, map, devices, responders, and acceptance evidence are reviewed.",
            "next_step": "Use the export package for deployment, review, and customer handoff.",
        },
    ]


def _load_runtime_blueprints_payload() -> dict[str, Any]:
    try:
        from askme.blueprints import catalog_payload

        payload = catalog_payload()
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _customer_visible_runtime_blueprints(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("items") if isinstance(payload, dict) else []
    if not isinstance(items, list):
        return []
    visible = [
        item
        for item in items
        if isinstance(item, dict) and bool(item.get("customer_visible"))
    ]
    return sorted(visible, key=lambda item: str(item.get("name") or ""))


def _select_template_runtime_blueprint(
    *,
    template: dict[str, Any],
    customer: dict[str, Any],
    delivery_summary: dict[str, Any],
    items: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    if not items:
        return None, "runtime_blueprint_catalog_empty"

    by_name = {str(item.get("name") or ""): item for item in items if item.get("name")}
    explicit = _template_explicit_runtime_blueprint(template)
    if explicit:
        matched = by_name.get(explicit)
        if matched is not None:
            return matched, f"explicit_template_binding:{explicit}"
        return None, f"explicit_template_binding_not_found:{explicit}"

    industry = str(customer.get("industry") or template.get("industry") or "").strip().lower()
    scenario_ids = set(_string_list(delivery_summary.get("scenario_ids")))
    skill_packages = set(_string_list(delivery_summary.get("skill_packages")))

    if industry in {"creative_park", "park", "campus", "scenic_area", "manufacturing", "factory", "warehouse"}:
        matched = by_name.get("edge_robot")
        if matched is not None:
            return matched, f"industry_default:{industry or 'customer_site'}"

    if scenario_ids and scenario_ids <= {"wayfinding_help_point", "visitor_escort"}:
        matched = by_name.get("lingtu_voice") or by_name.get("voice_perception")
        if matched is not None:
            return matched, "visitor_service_scenarios"

    if any("voice" in item for item in skill_packages):
        matched = by_name.get("voice_perception") or by_name.get("voice")
        if matched is not None:
            return matched, "voice_skill_package"

    return items[0], "first_customer_visible_runtime_blueprint"


def _template_explicit_runtime_blueprint(template: dict[str, Any]) -> str:
    runtime = _mapping(template.get("runtime"))
    for source in (template, runtime):
        for key in ("runtime_blueprint", "runtime_blueprint_id", "blueprint", "blueprint_id"):
            value = str(source.get(key) or "").strip()
            if value:
                return value
    return ""


def _template_runtime_blueprint_missing_fields(delivery_summary: dict[str, Any]) -> list[str]:
    missing = []
    for field in (
        "scenario_ids",
        "skill_packages",
        "device_sources",
        "acceptance_tests",
    ):
        if not _string_list(delivery_summary.get(field)):
            missing.append(field)
    return missing


def _template_runtime_blueprint_status(
    *,
    selected: dict[str, Any] | None,
    template: dict[str, Any],
    missing_fields: list[str],
) -> str:
    if selected is None or missing_fields:
        return "blocked"
    package = _mapping(selected.get("delivery_package"))
    readiness = _mapping(selected.get("readiness"))
    blueprint_status = str(package.get("status") or readiness.get("status") or "manual_check")
    publish_status = str(template.get("publish_status") or "draft")
    if blueprint_status in {"ready", "ready_for_site_validation"} and publish_status == "published":
        return "ready"
    return "manual_check"


def _runtime_blueprint_public_binding(selected: dict[str, Any] | None) -> dict[str, Any]:
    if not selected:
        return {}
    package = _mapping(selected.get("delivery_package"))
    readiness = _mapping(selected.get("readiness"))
    return {
        "name": str(selected.get("name") or ""),
        "title": str(selected.get("title") or selected.get("name") or ""),
        "product_stage": str(selected.get("product_stage") or ""),
        "status": str(package.get("status") or readiness.get("status") or "unknown"),
        "package_id": str(package.get("package_id") or ""),
        "primary_loop": str(selected.get("primary_loop") or ""),
        "deployment_targets": _string_list(selected.get("deployment_targets")),
        "capabilities": _string_list(selected.get("capabilities"))[:8],
        "scenarios": _string_list(selected.get("scenarios"))[:8],
        "missing_config": _string_list(readiness.get("missing_config")),
        "startup_command": str(selected.get("startup_command") or ""),
    }


def _template_runtime_blueprint_customer_claim(
    status: str,
    selected: dict[str, Any] | None,
) -> str:
    name = str(_mapping(selected).get("title") or _mapping(selected).get("name") or "runtime blueprint")
    if status == "ready":
        return f"Template is bound to {name} and can seed a customer pilot after onsite acceptance."
    if status == "manual_check":
        return f"Template is bound to {name}, but delivery must review runtime config and onsite evidence before customer claims."
    return "Template is not ready for customer delivery until runtime blueprint binding blockers are fixed."


def _template_runtime_blueprint_next_step(
    *,
    status: str,
    selected: dict[str, Any] | None,
    missing_fields: list[str],
) -> str:
    if missing_fields:
        return "Add template runtime evidence: " + ", ".join(missing_fields)
    if selected is None:
        return "Add or restore at least one customer-visible runtime blueprint."
    selected_public = _runtime_blueprint_public_binding(selected)
    missing_config = _string_list(selected_public.get("missing_config"))
    if missing_config:
        return "Complete runtime configuration: " + ", ".join(missing_config[:6])
    if status == "ready":
        return "Create a customer project and run onsite acceptance against this runtime blueprint."
    return "Delivery owner must review the runtime blueprint before using this template in a customer pilot."


def _unique_template_object_values(objects: list[dict[str, Any]], key: str) -> list[str]:
    return sorted({
        value
        for item in objects
        for value in _string_list(item.get(key))
    })


def _unique_template_binding_values(objects: list[dict[str, Any]], key: str) -> list[str]:
    return sorted({
        value
        for item in objects
        for value in _string_list(_mapping(item.get("bindings")).get(key))
    })


__all__ = [
    "_customer_delivery_applicability_scope",
    "_customer_delivery_dependency_matrix",
    "_customer_delivery_out_of_scope",
    "_customer_delivery_prerequisites",
    "_customer_delivery_scenario_acceptance_criteria",
    "_customer_delivery_surface",
    "_template_delivery_checklist",
    "_template_delivery_summary",
    "_template_package_summary",
    "_template_runtime_blueprint_binding",
    "_unique_template_binding_values",
    "_unique_template_object_values",
]
