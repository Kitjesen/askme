"""Customer-project template catalog read model.

This module owns the product-facing template-market listing used by routes and
delivery readiness. Mutation workflows such as release approval and project
creation remain in the template facade until their write closures are split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def list_customer_project_templates(
    root: Path,
    *,
    tenant_id: str = "",
    delivery_namespace: str = "",
    industry: str = "",
    publish_status: str = "",
    product_status: str = "",
    template_id: str = "",
    release_channel: str = "",
    owner: str = "",
) -> dict[str, Any]:
    """Return reusable industry templates for solution-provider rollout."""
    adapters = _site_profile_adapters()
    root = Path(root)
    templates = []
    filters = _customer_project_template_filters(
        tenant_id=tenant_id,
        delivery_namespace=delivery_namespace,
        industry=industry,
        publish_status=publish_status,
        product_status=product_status,
        template_id=template_id,
        release_channel=release_channel,
        owner=owner,
    )
    blueprints_payload = adapters["runtime_blueprints_payload"]()
    for path in adapters["site_profile_paths"](root, pattern="*.yaml"):
        try:
            profile = adapters["load_field_site_profile"](path)
            report = adapters["validate_field_site_profile"](profile)
            template = _mapping(profile.get("template"))
            customer = _mapping(profile.get("customer"))
            tenant = adapters["delivery_tenant_id"](customer)
            namespace = adapters["delivery_namespace"](customer)
            managed_objects = adapters["managed_object_catalog_from_site_profile"](profile)
            delivery_summary = adapters["template_delivery_summary"](
                template=template,
                customer=customer,
                managed_objects=managed_objects,
                report=report,
            )
            customer_delivery = adapters["customer_delivery_surface"](
                profile=profile,
                template=template,
                customer=customer,
                managed_objects=managed_objects,
                report=report,
                surface="template",
            )
            delivery_summary = delivery_summary | customer_delivery
            runtime_blueprint_binding = adapters["template_runtime_blueprint_binding"](
                template=template,
                customer=customer,
                delivery_summary=delivery_summary,
                blueprints_payload=blueprints_payload,
            )
            template_package = adapters["template_package_summary"](
                profile=profile,
                template=template,
                path=path,
                report=report,
                delivery_summary=delivery_summary,
            )
            templates.append(
                {
                    "template_id": str(template.get("template_id") or path.stem),
                    "display_name": str(template.get("display_name") or path.stem),
                    "tenant_id": tenant,
                    "delivery_namespace": namespace,
                    "industry": str(
                        customer.get("industry")
                        or template.get("industry")
                        or "unspecified"
                    ),
                    "template_version": template_package["version"],
                    "publish_status": template_package["publish_status"],
                    "release_channel": template_package["release_channel"],
                    "owner": template_package["owner"],
                    "product_status": template_package["product_status"],
                    "template_path": str(path),
                    "status": report.get("status"),
                    "errors": report.get("errors") or [],
                    "warnings": report.get("warnings") or [],
                    "managed_objects_summary": managed_objects | {
                        "objects": managed_objects["objects"][:8],
                        "objects_by_id": {},
                    },
                    "delivery_summary": delivery_summary,
                    "applicability_scope": customer_delivery["applicability_scope"],
                    "out_of_scope": customer_delivery["out_of_scope"],
                    "customer_prerequisites": customer_delivery["customer_prerequisites"],
                    "scenario_acceptance_criteria": customer_delivery[
                        "scenario_acceptance_criteria"
                    ],
                    "dependency_matrix": customer_delivery["dependency_matrix"],
                    "delivery_checklist": adapters["template_delivery_checklist"](
                        delivery_summary
                    ),
                    "template_package": template_package,
                    "runtime_blueprint_binding": runtime_blueprint_binding,
                    "customer_claim": str(
                        template.get("customer_claim")
                        or "Reusable customer project starter for this industry."
                    ),
                    "next_step": str(
                        template.get("next_step")
                        or "Create a customer project from this template, then bind real devices and credentials."
                    ),
                }
            )
        except Exception as exc:
            templates.append(
                {
                    "template_id": path.stem,
                    "display_name": path.stem,
                    "tenant_id": adapters["default_delivery_namespace"],
                    "delivery_namespace": adapters["default_delivery_namespace"],
                    "industry": "unknown",
                    "template_version": "0.0.0",
                    "publish_status": "blocked",
                    "release_channel": "blocked",
                    "owner": "unassigned",
                    "product_status": "blocked",
                    "template_path": str(path),
                    "status": "failed",
                    "errors": [str(exc)],
                    "warnings": [],
                    "managed_objects_summary": {},
                    "delivery_summary": adapters["template_delivery_summary"](
                        template={},
                        customer={},
                        managed_objects={},
                        report={
                            "status": "failed",
                            "errors": [str(exc)],
                            "warnings": [],
                        },
                    ),
                    "runtime_blueprint_binding": adapters["template_runtime_blueprint_binding"](
                        template={},
                        customer={},
                        delivery_summary={},
                        blueprints_payload=blueprints_payload,
                    ),
                    "applicability_scope": adapters["customer_delivery_applicability_scope"](
                        template={},
                        customer={},
                        managed_objects={},
                        surface="template",
                    ),
                    "out_of_scope": adapters["customer_delivery_out_of_scope"]("template"),
                    "customer_prerequisites": [],
                    "scenario_acceptance_criteria": [],
                    "dependency_matrix": [],
                    "delivery_checklist": adapters["template_delivery_checklist"](
                        adapters["template_delivery_summary"](
                            template={},
                            customer={},
                            managed_objects={},
                            report={
                                "status": "failed",
                                "errors": [str(exc)],
                                "warnings": [],
                            },
                        )
                    ),
                    "template_package": adapters["template_package_summary"](
                        profile={},
                        template={
                            "template_id": path.stem,
                            "version": "0.0.0",
                            "publish_status": "blocked",
                        },
                        path=path,
                        report={
                            "status": "failed",
                            "errors": [str(exc)],
                            "warnings": [],
                        },
                        delivery_summary={},
                    ),
                    "customer_claim": "模板校验错误修复前不能使用。",
                    "next_step": "Fix template YAML and rerun validation.",
                }
            )
    if filters:
        templates = [
            item
            for item in templates
            if _customer_project_template_matches_filters(item, filters)
        ]
    templates.sort(
        key=lambda item: (
            str(item.get("industry") or ""),
            str(item.get("template_id") or ""),
        )
    )
    summary = customer_project_template_summary_from_items(templates)
    if filters:
        summary["filtered"] = True
        summary["filters"] = filters
    return {
        "root": str(root),
        "filters": filters,
        "templates": templates,
        "summary": summary,
        "customer_claim": (
            "Industry templates let delivery teams start a new customer project without custom code."
        ),
    }


def customer_project_template_summary_from_items(
    templates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return template-market summary for the current filtered result set."""

    return {
        "template_count": len(templates),
        "valid_count": len([item for item in templates if item.get("status") == "passed"]),
        "product_ready_count": len(
            [
                item
                for item in templates
                if _mapping(item.get("template_package")).get("product_status") == "ready"
            ]
        ),
        "manual_check_count": len(
            [
                item
                for item in templates
                if _mapping(item.get("template_package")).get("product_status")
                == "manual_check"
            ]
        ),
        "blocked_count": len(
            [
                item
                for item in templates
                if _mapping(item.get("template_package")).get("product_status")
                == "blocked"
            ]
        ),
        "tenant_count": len(
            {str(item.get("tenant_id") or "") for item in templates if item.get("tenant_id")}
        ),
        "delivery_namespace_count": len(
            {
                str(item.get("delivery_namespace") or "")
                for item in templates
                if item.get("delivery_namespace")
            }
        ),
        "industry_count": len(
            {str(item.get("industry") or "") for item in templates if item.get("industry")}
        ),
        "publish_statuses": sorted(
            {
                str(item.get("publish_status") or "")
                for item in templates
                if item.get("publish_status")
            }
        ),
        "product_statuses": sorted(
            {
                str(
                    _mapping(item.get("template_package")).get("product_status")
                    or item.get("product_status")
                    or ""
                )
                for item in templates
                if _mapping(item.get("template_package")).get("product_status")
                or item.get("product_status")
            }
        ),
        "managed_object_type_count": sum(
            int(_mapping(item.get("managed_objects_summary")).get("object_type_count") or 0)
            for item in templates
        ),
        "runtime_blueprint_bound_count": len(
            [
                item
                for item in templates
                if _mapping(
                    _mapping(item.get("runtime_blueprint_binding")).get("selected_blueprint")
                ).get("name")
            ]
        ),
        "runtime_blueprint_ready_count": len(
            [
                item
                for item in templates
                if _mapping(item.get("runtime_blueprint_binding")).get("status") == "ready"
            ]
        ),
        "runtime_blueprint_manual_check_count": len(
            [
                item
                for item in templates
                if _mapping(item.get("runtime_blueprint_binding")).get("status")
                == "manual_check"
            ]
        ),
        "runtime_blueprint_blocked_count": len(
            [
                item
                for item in templates
                if _mapping(item.get("runtime_blueprint_binding")).get("status") == "blocked"
            ]
        ),
    }


def _customer_project_template_filters(**values: str) -> dict[str, str]:
    return {
        key: str(value or "").strip()
        for key, value in values.items()
        if str(value or "").strip()
    }


def _customer_project_template_matches_filters(
    template: dict[str, Any],
    filters: dict[str, str],
) -> bool:
    package = _mapping(template.get("template_package"))
    for key, expected in filters.items():
        value = str(package.get(key) or template.get(key) or "")
        if not _text_filter_matches(value, expected):
            return False
    return True


def _text_filter_matches(value: str, expected: str) -> bool:
    expected = str(expected or "").strip().lower()
    if not expected:
        return True
    options = [item.strip() for item in expected.split(",") if item.strip()]
    if not options:
        return True
    lower_value = str(value or "").strip().lower()
    return any(option in lower_value for option in options)


def _site_profile_adapters() -> dict[str, Any]:
    from askme.pipeline.field.customer_project_managed_objects import (
        managed_object_catalog_from_site_profile,
    )
    from askme.pipeline.field.customer_project_template_delivery import (
        _customer_delivery_applicability_scope,
        _customer_delivery_out_of_scope,
        _customer_delivery_surface,
        _template_delivery_checklist,
        _template_delivery_summary,
        _template_package_summary,
        _template_runtime_blueprint_binding,
    )
    from askme.pipeline.field.customer_project_template_support import (
        DEFAULT_DELIVERY_NAMESPACE,
        _delivery_namespace,
        _delivery_tenant_id,
        _site_profile_paths,
        load_field_site_profile,
    )
    from askme.pipeline.field.field_site_validation import (
        validate_field_site_profile,
    )

    return {
        "customer_delivery_applicability_scope": _customer_delivery_applicability_scope,
        "customer_delivery_out_of_scope": _customer_delivery_out_of_scope,
        "customer_delivery_surface": _customer_delivery_surface,
        "default_delivery_namespace": DEFAULT_DELIVERY_NAMESPACE,
        "delivery_namespace": _delivery_namespace,
        "delivery_tenant_id": _delivery_tenant_id,
        "load_field_site_profile": load_field_site_profile,
        "managed_object_catalog_from_site_profile": managed_object_catalog_from_site_profile,
        "site_profile_paths": _site_profile_paths,
        "template_delivery_checklist": _template_delivery_checklist,
        "template_delivery_summary": _template_delivery_summary,
        "template_package_summary": _template_package_summary,
        "template_runtime_blueprint_binding": _template_runtime_blueprint_binding,
        "validate_field_site_profile": validate_field_site_profile,
        "runtime_blueprints_payload": _runtime_blueprints_payload,
    }


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _runtime_blueprints_payload() -> dict[str, Any]:
    try:
        from askme.blueprints import catalog_payload

        payload = catalog_payload()
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


__all__ = [
    "customer_project_template_summary_from_items",
    "list_customer_project_templates",
]
