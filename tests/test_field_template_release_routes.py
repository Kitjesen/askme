"""HTTP tests for field template release route boundaries."""

import shutil
from pathlib import Path

from askme.pipeline.field_site_profile import (
    DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT,
    create_customer_project_template_release_request,
    review_customer_project_template_release_request,
    update_customer_project_template_release,
)
from fastapi.testclient import TestClient

from tests.support.field_route_app import (
    field_route_test_app as _field_route_test_app,
)
from tests.support.field_route_app import (
    scoped_project_authorize as _scoped_project_authorize,
)


def test_field_template_release_endpoint_uses_effective_publish_status(
    tmp_path: Path,
) -> None:
    template_root = tmp_path / "customer-project-templates"
    shutil.copytree(DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT, template_root)
    site_profile_root = tmp_path / "site-profiles"
    site_profile_root.mkdir()

    promoted = update_customer_project_template_release(
        template_root,
        "factory-inspection",
        {"version": "0.1.1", "publish_status": "published", "release_channel": "stable"},
        operator_id="product.reviewer",
        reason="Seed a published template for route authorization regression.",
        allow_published=True,
        approval_request_id="seed-approved-request",
    )
    assert promoted["accepted"] is True

    client = TestClient(
        _field_route_test_app(
            site_profile_root,
            customer_project_template_root=template_root,
        )
    )
    response = client.post(
        "/api/field/customer-project-templates/factory-inspection/release",
        json={
            "operator_id": "product.owner",
            "release": {
                "release_note": (
                    "Attempt to edit a published template without request approval."
                )
            },
        },
    )
    assert response.status_code == 409
    assert response.json()["reason"] == "published_release_requires_approval_request"


def test_field_template_release_read_surfaces_respect_project_scope(
    tmp_path: Path,
) -> None:
    template_root = tmp_path / "customer-project-templates"
    shutil.copytree(DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT, template_root)
    site_profile_root = tmp_path / "site-profiles"
    site_profile_root.mkdir()

    tenant_template = (template_root / "factory-inspection.yaml").read_text(
        encoding="utf-8"
    )
    tenant_template = tenant_template.replace(
        "template_id: factory-inspection",
        "template_id: tenant-factory",
        1,
    )
    tenant_template = tenant_template.replace(
        "display_name: Factory inspection starter",
        "display_name: Tenant factory starter",
        1,
    )
    tenant_template = tenant_template.replace(
        "customer:\n  customer_id: template-customer",
        (
            "customer:\n"
            "  tenant_id: tenant-a\n"
            "  delivery_namespace: pilot\n"
            "  customer_id: tenant-customer"
        ),
        1,
    )
    tenant_template = tenant_template.replace(
        "project_id: factory-inspection-project",
        "project_id: tenant-factory-project",
        1,
    )
    tenant_template = tenant_template.replace(
        "site_id: template-factory-site",
        "site_id: tenant-factory-site",
        1,
    )
    (template_root / "tenant-factory.yaml").write_text(
        tenant_template,
        encoding="utf-8",
    )

    request_result = create_customer_project_template_release_request(
        template_root,
        "tenant-factory",
        {"version": "0.1.1", "publish_status": "published", "release_channel": "stable"},
        operator_id="product.owner",
        reason="Seed tenant-specific release governance data.",
    )
    assert request_result["accepted"] is True
    review_result = review_customer_project_template_release_request(
        template_root,
        request_result["request"]["request_id"],
        decision="approve",
        operator_id="product.reviewer",
    )
    assert review_result["accepted"] is True

    client = TestClient(
        _field_route_test_app(
            site_profile_root,
            customer_project_template_root=template_root,
            authorize_callback=_scoped_project_authorize(
                {
                    "tenant_ids": ["default"],
                    "delivery_namespaces": ["default"],
                }
            ),
        )
    )

    history = client.get("/api/field/customer-project-templates/tenant-factory/history")
    assert history.status_code == 403
    assert history.json()["reason"] == "project_scope_not_allowed"

    release_requests = client.get("/api/field/customer-project-template-release-requests")
    assert release_requests.status_code == 200
    assert release_requests.json()["summary"]["scope_filtered"] is True
    assert all(
        item["template_id"] != "tenant-factory"
        for item in release_requests.json()["requests"]
    )

    release_notes = client.get("/api/field/customer-project-template-release-notes")
    assert release_notes.status_code == 200
    assert release_notes.json()["summary"]["scope_filtered"] is True
    assert all(
        item["template_id"] != "tenant-factory"
        for item in release_notes.json()["notes"]
    )

    release_notes_bundle = client.post(
        "/api/field/customer-project-template-release-notes/export",
        json={"customer_context": {"project_name": "Scoped Proposal"}},
    )
    assert release_notes_bundle.status_code == 200
    bundle = release_notes_bundle.json()["bundle"]
    assert release_notes_bundle.json()["summary"]["scope_filtered"] is True
    assert all(
        item["template_id"] != "tenant-factory"
        for item in bundle["release_notes"]
    )
    assert "tenant-factory" not in bundle["html"]

    denied_release_request = client.post(
        "/api/field/customer-project-templates/tenant-factory/release-requests",
        json={
            "release": {"version": "0.1.2", "publish_status": "published"},
            "reason": "Out-of-scope release request should fail.",
        },
    )
    assert denied_release_request.status_code == 403
    assert denied_release_request.json()["reason"] == "project_scope_not_allowed"

    denied_review = client.post(
        (
            "/api/field/customer-project-template-release-requests/"
            f"{request_result['request']['request_id']}/review"
        ),
        json={"decision": "reject", "reason": "Out-of-scope review should fail."},
    )
    assert denied_review.status_code == 403
    assert denied_review.json()["reason"] == "project_scope_not_allowed"

    denied_release = client.post(
        "/api/field/customer-project-templates/tenant-factory/release",
        json={"release": {"version": "0.1.2", "publish_status": "pilot"}},
    )
    assert denied_release.status_code == 403
    assert denied_release.json()["reason"] == "project_scope_not_allowed"
