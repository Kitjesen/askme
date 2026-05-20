"""Public customer-project catalog, profile, object, and acceptance API.

New API and service code should import customer project catalog,
managed-object, evidence, acceptance, history, rollback, and archive functions
from this product facade.
"""

from __future__ import annotations

from askme.pipeline.field.customer_project_acceptance import (
    customer_project_acceptance_closure,
    customer_project_acceptance_report,
    list_customer_project_customer_signoffs,
    list_customer_project_onsite_evidence,
    register_customer_project_acceptance_review,
    register_customer_project_customer_signoff,
    register_customer_project_onsite_evidence,
)
from askme.pipeline.field.customer_project_acceptance_registry import (
    build_customer_project_acceptance_registry,
)
from askme.pipeline.field.customer_project_execution_bindings import (
    build_customer_project_execution_bindings,
)
from askme.pipeline.field.customer_project_profile_operations import (
    delete_managed_object,
    get_customer_project_profile,
    rollback_customer_project_profile,
    upsert_customer_project_profile,
    upsert_managed_object,
)
from askme.pipeline.field.customer_project_profiles import (
    archive_customer_project_profile,
    customer_project_catalog_acceptance_gate,
    customer_project_catalog_summary_from_projects,
    list_customer_project_revisions,
)
from askme.pipeline.field.customer_project_resource_catalog import (
    build_customer_project_resource_catalog,
)
from askme.pipeline.field.field_site_catalog import (
    build_customer_project_catalog,
    build_site_profile_catalog,
    build_site_profile_report,
)
from askme.pipeline.field.solution_delivery_readiness import (
    build_solution_delivery_readiness,
)

__all__ = [
    "archive_customer_project_profile",
    "build_customer_project_acceptance_registry",
    "build_customer_project_catalog",
    "build_customer_project_execution_bindings",
    "build_customer_project_resource_catalog",
    "build_site_profile_catalog",
    "build_site_profile_report",
    "build_solution_delivery_readiness",
    "customer_project_acceptance_closure",
    "customer_project_acceptance_report",
    "customer_project_catalog_acceptance_gate",
    "customer_project_catalog_summary_from_projects",
    "delete_managed_object",
    "get_customer_project_profile",
    "list_customer_project_customer_signoffs",
    "list_customer_project_onsite_evidence",
    "list_customer_project_revisions",
    "register_customer_project_acceptance_review",
    "register_customer_project_customer_signoff",
    "register_customer_project_onsite_evidence",
    "rollback_customer_project_profile",
    "upsert_customer_project_profile",
    "upsert_managed_object",
]
