"""Public customer-project template catalog and release-governance API.

New API and service code should import template catalog, project creation,
release requests, and release notes from this product facade.
"""

from __future__ import annotations

from askme.pipeline.field.customer_project_profile_operations import (
    create_customer_project_from_template,
)
from askme.pipeline.field.customer_project_template_catalog import (
    customer_project_template_summary_from_items,
    list_customer_project_templates,
)
from askme.pipeline.field.customer_project_template_release import (
    create_customer_project_template_release_request,
    customer_project_template_release_notes,
    export_customer_project_template_release_notes_bundle,
    list_customer_project_template_release_requests,
    list_customer_project_template_revisions,
    review_customer_project_template_release_request,
    update_customer_project_template_release,
)

__all__ = [
    "create_customer_project_from_template",
    "create_customer_project_template_release_request",
    "customer_project_template_release_notes",
    "customer_project_template_summary_from_items",
    "export_customer_project_template_release_notes_bundle",
    "list_customer_project_template_release_requests",
    "list_customer_project_template_revisions",
    "list_customer_project_templates",
    "review_customer_project_template_release_request",
    "update_customer_project_template_release",
]
