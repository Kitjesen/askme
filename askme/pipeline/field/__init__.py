"""Field operations, incident handling, and delivery readiness pipeline modules.

New field-delivery code should depend on the public product boundaries listed
in ``__all__`` instead of importing from the large ``field_site_profile`` module
directly. That module remains as the compatibility and implementation holder
until its function clusters are physically migrated.
"""

from __future__ import annotations

__all__ = [
    "customer_project_acceptance",
    "customer_project_acceptance_registry",
    "alert_dispatcher",
    "customer_project_artifact_manifests",
    "customer_project_evidence_inventory",
    "customer_project_execution_bindings",
    "customer_project_package_assessment",
    "customer_project_package_html",
    "customer_project_package_rules",
    "customer_project_managed_objects",
    "customer_project_profile_operations",
    "customer_project_profiles",
    "customer_project_resource_catalog",
    "customer_project_scope",
    "customer_project_implementation_handoff",
    "customer_project_template_catalog",
    "customer_project_template_delivery",
    "customer_project_template_release",
    "customer_project_template_support",
    "customer_project_artifacts",
    "customer_project_templates",
    "customer_projects",
    "delivery_resource_governance",
    "delivery_resource_registry",
    "delivery_resources",
    "field_deployment_readiness",
    "field_ingest_adapters",
    "field_ingest_bridge",
    "field_operations",
    "field_scenarios",
    "field_site_catalog",
    "field_site_runtime_config",
    "field_site_validation",
    "field_site_profile",
    "incident_alerts",
    "paths",
    "product_launch_readiness",
    "solution_delivery_readiness",
]
