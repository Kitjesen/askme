from __future__ import annotations

from askme.pipeline.field.customer_project_scope import (
    _customer_delivery_filename_parts,
    _customer_project_profile_diff,
    _delivery_scope_payload,
    _same_delivery_project_scope,
)


def test_delivery_scope_uses_customer_project_before_site_fallback() -> None:
    scope = _delivery_scope_payload(
        {
            "customer": {
                "tenant_id": "tenant-a",
                "delivery_namespace": "pilot",
                "customer_id": "fanmu",
                "project_id": "phase-one",
            },
            "site": {"site_id": "site-01"},
        }
    )

    assert scope == {
        "tenant_id": "tenant-a",
        "delivery_namespace": "pilot",
        "customer_id": "fanmu",
        "project_id": "phase-one",
        "site_id": "site-01",
    }
    assert _customer_delivery_filename_parts({"tenant_id": "tenant-a", "delivery_namespace": "pilot"}) == [
        "tenant-a",
        "pilot",
    ]


def test_delivery_scope_identity_requires_tenant_and_namespace() -> None:
    base = {
        "tenant_id": "tenant-a",
        "delivery_namespace": "pilot",
        "customer_id": "fanmu",
        "project_id": "phase-one",
        "site_id": "site-01",
    }

    assert _same_delivery_project_scope(base, dict(base)) is True
    assert _same_delivery_project_scope(base, {**base, "delivery_namespace": "prod"}) is False


def test_customer_project_profile_diff_hashes_changed_top_level_sections() -> None:
    diff = _customer_project_profile_diff(
        {
            "customer": {"customer_id": "fanmu"},
            "site": {"site_id": "site-01"},
            "devices": {"camera": {"enabled": True}},
        },
        {
            "customer": {"customer_id": "fanmu"},
            "site": {"site_id": "site-02"},
            "devices": {"camera": {"enabled": False}},
        },
    )

    assert [item["path"] for item in diff] == ["site", "devices"]
    assert all(item["current_sha256"] != item["incoming_sha256"] for item in diff)
