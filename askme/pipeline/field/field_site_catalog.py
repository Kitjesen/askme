"""Customer site-profile reports and project catalog views."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from askme.pipeline.field.customer_project_managed_objects import (
    _managed_object_binding_missing_count,
    managed_object_catalog_from_site_profile,
)
from askme.pipeline.field.customer_project_profiles import (
    _customer_payload,
    _customer_project_catalog_filters,
    _customer_project_delivery_status,
    _customer_project_matches_filters,
    _customer_project_product_acceptance_gate,
    _customer_rows,
    _object_change_log_payload,
    customer_project_catalog_acceptance_gate,
    customer_project_catalog_summary_from_projects,
)
from askme.pipeline.field.customer_project_template_delivery import (
    _unique_template_binding_values,
)
from askme.pipeline.field.customer_project_template_support import (
    DEFAULT_DELIVERY_NAMESPACE,
    _mapping,
    _site_profile_paths,
    _string_list,
    load_field_site_profile,
    site_profile_env_references,
)
from askme.pipeline.field.field_site_runtime_config import (
    field_operations_config_from_site_profile,
)
from askme.pipeline.field.field_site_validation import validate_field_site_profile


def build_site_profile_report(path: Path, *, check_env: bool = False) -> dict[str, Any]:
    profile = load_field_site_profile(path)
    report = validate_field_site_profile(profile, check_env=check_env)
    report["profile_path"] = str(path)
    if report["status"] == "passed":
        report["field_operations_config"] = field_operations_config_from_site_profile(profile)
    return report


def build_site_profile_catalog(
    root: Path,
    *,
    check_env: bool = False,
    pattern: str = "*.yaml",
) -> dict[str, Any]:
    """Return a customer-site catalog for multi-site field deployments."""
    root = Path(root)
    profile_paths = _site_profile_paths(root, pattern=pattern)
    sites = [_site_profile_catalog_item(path, check_env=check_env) for path in profile_paths]
    sites.sort(key=lambda item: (str(item.get("site_id") or ""), str(item.get("profile_path") or "")))
    passed = [item for item in sites if item.get("status") == "passed"]
    blocked = [item for item in sites if item.get("status") != "passed"]
    production_ready = [item for item in sites if item.get("deployment_stage") == "production_ready"]
    env_missing = sum(int(item.get("env_missing_count") or 0) for item in sites)
    customer_summary = _customer_project_summary(sites)
    return {
        "root": str(root),
        "check_env": bool(check_env),
        "sites": sites,
        "summary": {
            "site_count": len(sites),
            "configured_count": len(passed),
            "blocked_count": len(blocked),
            "production_ready_count": len(production_ready),
            "env_missing_count": env_missing,
            **customer_summary,
            "multi_site_ready": bool(sites) and not blocked,
        },
        "customer_claim": (
            "Customer projects can define their own sites, devices, responder groups, and managed objects."
            if sites and not blocked
            else "Some customer project profiles are missing required deployment configuration."
        ),
        "next_step": _site_catalog_next_step(sites, check_env=check_env),
    }


def build_customer_project_catalog(
    root: Path,
    *,
    check_env: bool = False,
    pattern: str = "*.yaml",
    tenant_id: str = "",
    delivery_namespace: str = "",
    customer_id: str = "",
    project_id: str = "",
    site_id: str = "",
    industry: str = "",
    gate_status: str = "",
    deployment_stage: str = "",
) -> dict[str, Any]:
    """Return product-facing customer/project/object coverage for solution rollout."""
    catalog = build_site_profile_catalog(root, check_env=check_env, pattern=pattern)
    sites = catalog.get("sites") if isinstance(catalog.get("sites"), list) else []
    projects: list[dict[str, Any]] = []
    for site in sites:
        customer = _mapping(site.get("customer"))
        project = {
            "tenant_id": customer.get("tenant_id") or DEFAULT_DELIVERY_NAMESPACE,
            "delivery_namespace": customer.get("delivery_namespace") or DEFAULT_DELIVERY_NAMESPACE,
            "customer_id": customer.get("customer_id") or "",
            "customer_name": customer.get("customer_name") or "Unassigned customer",
            "industry": customer.get("industry") or "unspecified",
            "project_id": customer.get("project_id") or site.get("site_id") or "",
            "project_name": customer.get("project_name") or site.get("site_name") or "",
            "site_id": site.get("site_id") or "",
            "site_name": site.get("site_name") or "",
            "delivery_model": customer.get("delivery_model") or "solution_project",
            "deployment_stage": site.get("deployment_stage") or "blocked",
            "status": site.get("status") or "failed",
            "managed_objects_summary": site.get("managed_objects_summary") or {},
            "managed_objects": site.get("managed_objects") or [],
            "object_change_log": site.get("object_change_log") or [],
            "delivery_workflow": site.get("delivery_workflow") or {},
            "env_missing_count": int(site.get("env_missing_count") or 0),
            "next_step": site.get("next_step") or "",
        }
        project["product_acceptance_gate"] = _customer_project_product_acceptance_gate(project)
        projects.append(project)
    filters = _customer_project_catalog_filters(
        tenant_id=tenant_id,
        delivery_namespace=delivery_namespace,
        customer_id=customer_id,
        project_id=project_id,
        site_id=site_id,
        industry=industry,
        gate_status=gate_status,
        deployment_stage=deployment_stage,
    )
    if filters:
        projects = [
            project for project in projects if _customer_project_matches_filters(project, filters)
        ]
    delivery_acceptance_gate = customer_project_catalog_acceptance_gate(projects)
    summary = customer_project_catalog_summary_from_projects(
        projects,
        base_summary=catalog.get("summary") if isinstance(catalog.get("summary"), dict) else {},
    )
    if filters:
        summary["filtered"] = True
        summary["filters"] = filters
    return {
        "root": catalog.get("root"),
        "check_env": catalog.get("check_env"),
        "filters": filters,
        "summary": summary,
        "delivery_acceptance_gate": delivery_acceptance_gate,
        "projects": projects,
        "customers": _customer_rows(projects),
        "customer_claim": (
            "AskMe is configured as a repeatable solution product: each customer project declares its own managed objects."
        ),
        "next_step": catalog.get("next_step") or _site_catalog_next_step(sites, check_env=check_env),
    }


def _site_profile_catalog_item(path: Path, *, check_env: bool) -> dict[str, Any]:
    try:
        profile = load_field_site_profile(path)
        report = validate_field_site_profile(profile, check_env=check_env)
    except Exception as exc:
        return {
            "site_id": "",
            "site_name": path.stem,
            "profile_path": str(path),
            "status": "failed",
            "deployment_stage": "blocked",
            "customer_status": "Blocked",
            "errors": [str(exc)],
            "warnings": [],
            "readiness": {},
            "env_missing_count": 0,
            "delivery_workflow": _customer_project_delivery_workflow(
                profile={},
                report={"status": "failed", "errors": [str(exc)], "warnings": []},
                managed_objects={},
                env_missing=[],
            ),
            "next_step": "Fix the site profile file so it can be parsed and validated.",
        }
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    errors = report.get("errors") if isinstance(report.get("errors"), list) else []
    managed_objects = managed_object_catalog_from_site_profile(profile)
    env_refs = site_profile_env_references(profile)
    env_missing = [item for item in env_refs if item.get("required") and not item.get("configured")]
    status = str(report.get("status") or "failed")
    deployment_stage = _site_deployment_stage(status, warnings, env_missing, check_env=check_env)
    return {
        "site_id": str(summary.get("site_id") or ""),
        "site_name": str(summary.get("site_name") or path.stem),
        "map_version": str(summary.get("map_version") or ""),
        "profile_path": str(path),
        "status": status,
        "deployment_stage": deployment_stage,
        "customer_status": _site_customer_status(deployment_stage),
        "errors": errors,
        "warnings": warnings,
        "readiness": report.get("readiness") if isinstance(report.get("readiness"), dict) else {},
        "summary": summary,
        "customer": _customer_payload(profile),
        "managed_objects_summary": managed_objects | {
            "objects": managed_objects["objects"][:8],
            "objects_by_id": {},
        },
        "managed_objects": managed_objects["objects"],
        "object_change_log": _object_change_log_payload(profile),
        "delivery_workflow": _customer_project_delivery_workflow(
            profile=profile,
            report=report,
            managed_objects=managed_objects,
            env_missing=env_missing,
        ),
        "env_reference_count": len(env_refs),
        "env_missing_count": len(env_missing),
        "env_missing": [
            {
                "env_name": str(item.get("env_name") or ""),
                "category": str(item.get("category") or ""),
                "owner": str(item.get("owner") or ""),
                "purpose": str(item.get("purpose") or ""),
            }
            for item in env_missing[:12]
        ],
        "next_step": _site_profile_next_step(status, deployment_stage, errors, warnings, env_missing),
    }


def _customer_project_delivery_workflow(
    *,
    profile: dict[str, Any],
    report: dict[str, Any],
    managed_objects: dict[str, Any],
    env_missing: list[dict[str, Any]],
) -> dict[str, Any]:
    customer = _customer_payload(profile) if isinstance(profile, dict) else {}
    site = _mapping(profile.get("site")) if isinstance(profile, dict) else {}
    summary = _mapping(report.get("summary"))
    readiness = _mapping(report.get("readiness"))
    objects = [
        item
        for item in managed_objects.get("objects", [])
        if isinstance(item, dict)
    ]
    acceptance = _mapping(managed_objects.get("acceptance_summary"))
    binding_missing_count = _managed_object_binding_missing_count(objects)
    vision_models = _unique_template_binding_values(objects, "vision_models")
    sensor_protocols = _unique_template_binding_values(objects, "sensor_protocols")
    skill_packages = _unique_template_binding_values(objects, "skill_packages")
    scope_fields = {
        "tenant_id": customer.get("tenant_id"),
        "delivery_namespace": customer.get("delivery_namespace"),
        "customer_id": customer.get("customer_id"),
        "project_id": customer.get("project_id"),
        "site_id": site.get("site_id") or summary.get("site_id"),
    }
    scope_ready = all(str(value or "").strip() for value in scope_fields.values())
    map_ready = bool(readiness.get("map_configured")) and bool(readiness.get("device_registry_configured"))
    responder_ready = bool(readiness.get("responder_groups_configured"))
    steps = [
        {
            "step_id": "customer_scope",
            "label": "Customer scope",
            "status": "ready" if scope_ready else "blocked",
            "evidence": " / ".join(str(value or "-") for value in scope_fields.values()),
            "next_step": "Set tenant, namespace, customer, project, and site identifiers.",
        },
        {
            "step_id": "managed_objects",
            "label": "Managed object catalog",
            "status": "ready" if objects else "blocked",
            "evidence": f"{len(objects)} object(s), {len(_string_list(managed_objects.get('scenario_ids')))} scenario(s).",
            "next_step": "Define the customer's real assets, areas, visitor services, and response ownership.",
        },
        {
            "step_id": "runtime_bindings",
            "label": "Runtime bindings",
            "status": "ready" if objects and binding_missing_count == 0 else "manual_check" if objects else "blocked",
            "evidence": (
                f"{len(vision_models)} vision model(s), {len(sensor_protocols)} sensor protocol(s), "
                f"{len(skill_packages)} skill package(s), {binding_missing_count} missing binding(s)."
            ),
            "next_step": "Bind every managed object to model, protocol, skill, and acceptance references.",
        },
        {
            "step_id": "site_map_devices",
            "label": "Site map and devices",
            "status": "ready" if report.get("status") == "passed" and map_ready else "blocked",
            "evidence": (
                f"{summary.get('zone_count', 0)} zone(s), {summary.get('device_count', 0)} device(s), "
                f"{summary.get('help_point_count', 0)} help point(s)."
            ),
            "next_step": "Complete map zones, help points, device registry, and source coverage.",
        },
        {
            "step_id": "responder_credentials",
            "label": "Responder and credentials",
            "status": "manual_check" if env_missing else "ready" if responder_ready else "blocked",
            "evidence": f"{len(env_missing)} missing required environment value(s).",
            "next_step": "Configure DingTalk responders, robot callbacks, and signed device secrets.",
        },
        {
            "step_id": "acceptance_evidence",
            "label": "Acceptance evidence",
            "status": str(acceptance.get("overall_status") or "blocked"),
            "evidence": (
                f"{acceptance.get('ready_object_count', 0)} ready / "
                f"{acceptance.get('manual_check_object_count', 0)} manual / "
                f"{acceptance.get('blocked_object_count', 0)} blocked object(s)."
            ),
            "next_step": "Run repository evidence and onsite smoke tests before customer signoff.",
        },
        {
            "step_id": "handoff_package",
            "label": "Handoff package",
            "status": "manual_check",
            "evidence": "Export package and acceptance dossier after the above steps are reviewed.",
            "next_step": "Export the customer package, acceptance report, and printable dossier.",
        },
    ]
    blocked_count = sum(1 for item in steps if item["status"] == "blocked")
    manual_count = sum(1 for item in steps if item["status"] == "manual_check")
    ready_count = sum(1 for item in steps if item["status"] == "ready")
    overall_status = "blocked" if blocked_count else "manual_check" if manual_count else "ready"
    next_step = next(
        (str(item.get("next_step") or "") for item in steps if item.get("status") != "ready"),
        "项目交付流程已具备最终交接复核条件。",
    )
    return {
        "overall_status": overall_status,
        "ready_count": ready_count,
        "manual_check_count": manual_count,
        "blocked_count": blocked_count,
        "steps": steps,
        "next_step": next_step,
        "customer_status": _customer_project_delivery_status(overall_status, blocked_count, manual_count),
        "release_claim": "该流程支持交付复核和客户试点交接；生产上线仍需要现场硬件、凭证、安全和人工接管验收。",
    }


def _site_deployment_stage(
    status: str,
    warnings: list[Any],
    env_missing: list[dict[str, Any]],
    *,
    check_env: bool,
) -> str:
    if status != "passed":
        return "blocked"
    if check_env and (warnings or env_missing):
        return "site_config_ready"
    return "production_ready" if check_env else "site_config_ready"


def _site_customer_status(stage: str) -> str:
    return {
        "production_ready": "Production credentials configured",
        "site_config_ready": "Site configuration ready; live credentials still need validation",
        "blocked": "Blocked by missing site configuration",
    }.get(stage, "Unknown")


def _site_profile_next_step(
    status: str,
    deployment_stage: str,
    errors: list[Any],
    warnings: list[Any],
    env_missing: list[dict[str, Any]],
) -> str:
    if status != "passed":
        first = str(errors[0]) if errors else "missing required site profile fields"
        return f"Fix site profile validation error: {first}"
    if env_missing:
        return "Fill deployment environment variables for DingTalk responders and signed devices."
    if warnings:
        return "Resolve site profile warnings, then rerun readiness with environment checks."
    if deployment_stage == "production_ready":
        return "Run live field demo and device bridge smoke against the deployed service."
    return "Run readiness with check_env=true before customer production acceptance."


def _site_catalog_next_step(sites: list[dict[str, Any]], *, check_env: bool) -> str:
    if not sites:
        return "Create at least one site profile under deploy/site-profiles."
    blocked = [item for item in sites if item.get("status") != "passed"]
    if blocked:
        return "Fix blocked site profiles before scaling rollout."
    missing_env = sum(int(item.get("env_missing_count") or 0) for item in sites)
    if missing_env:
        return "Fill missing responder and device secrets for each configured site."
    if not check_env:
        return "Rerun the catalog with environment checks before production acceptance."
    return "Select a site and run live device, notification, voice, and runtime smokes."


def _customer_project_summary(sites: list[dict[str, Any]]) -> dict[str, Any]:
    customers: set[str] = set()
    projects: set[str] = set()
    tenants: set[str] = set()
    namespaces: set[str] = set()
    industries: set[str] = set()
    object_categories: set[str] = set()
    scenario_ids: set[str] = set()
    object_count = 0
    for site in sites:
        customer = _mapping(site.get("customer"))
        customer_id = str(customer.get("customer_id") or "")
        project_id = str(customer.get("project_id") or "")
        tenant_id = str(customer.get("tenant_id") or DEFAULT_DELIVERY_NAMESPACE)
        delivery_namespace = str(customer.get("delivery_namespace") or DEFAULT_DELIVERY_NAMESPACE)
        industry = str(customer.get("industry") or "")
        tenants.add(tenant_id)
        namespaces.add(delivery_namespace)
        if customer_id:
            customers.add(customer_id)
        if project_id:
            projects.add(project_id)
        if industry:
            industries.add(industry)
        summary = _mapping(site.get("managed_objects_summary"))
        object_count += int(summary.get("object_type_count") or 0)
        object_categories.update(str(item) for item in summary.get("categories") or [] if item)
        scenario_ids.update(str(item) for item in summary.get("scenario_ids") or [] if item)
    return {
        "customer_count": len(customers),
        "project_count": len(projects),
        "tenant_count": len(tenants),
        "delivery_namespace_count": len(namespaces),
        "industry_count": len(industries),
        "managed_object_type_count": object_count,
        "managed_object_categories": sorted(object_categories),
        "covered_scenario_count": len(scenario_ids),
    }


__all__ = [
    "build_customer_project_catalog",
    "build_site_profile_catalog",
    "build_site_profile_report",
    "_customer_project_delivery_workflow",
    "_customer_project_summary",
    "_site_catalog_next_step",
    "_site_customer_status",
    "_site_deployment_stage",
    "_site_profile_catalog_item",
    "_site_profile_next_step",
]
