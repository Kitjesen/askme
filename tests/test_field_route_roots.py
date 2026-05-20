from pathlib import Path

from askme.api.services.field_route_roots import (
    build_field_route_roots,
    field_operations_path_roots,
)
from askme.pipeline.field.paths import (
    DEFAULT_CUSTOMER_PROJECT_ACCEPTANCE_DOSSIER_ROOT,
    DEFAULT_CUSTOMER_PROJECT_PACKAGE_ROOT,
    DEFAULT_CUSTOMER_PROJECT_PROPOSAL_ROOT,
    DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT,
    DEFAULT_DELIVERY_RESOURCE_ROOT,
    DEFAULT_SITE_PROFILE_ROOT,
)


def test_field_route_roots_use_repo_defaults_without_overrides() -> None:
    roots = build_field_route_roots()

    assert roots.site_profile_root() == DEFAULT_SITE_PROFILE_ROOT
    assert roots.template_root() == DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT
    assert roots.delivery_resource_root() == DEFAULT_DELIVERY_RESOURCE_ROOT
    assert roots.package_output_root() == DEFAULT_CUSTOMER_PROJECT_PACKAGE_ROOT
    assert (
        roots.acceptance_dossier_output_root()
        == DEFAULT_CUSTOMER_PROJECT_ACCEPTANCE_DOSSIER_ROOT
    )
    assert roots.proposal_output_root() == DEFAULT_CUSTOMER_PROJECT_PROPOSAL_ROOT


def test_field_route_roots_anchor_to_customer_deploy_root() -> None:
    site_root = Path("deploy/customer-a/site-profiles")
    roots = build_field_route_roots(site_profile_root=site_root)

    assert roots.site_profile_root() == site_root
    assert roots.deploy_root() == Path("deploy/customer-a")
    assert roots.template_root() == Path("deploy/customer-a/customer-project-templates")
    assert roots.delivery_resource_root() == Path("deploy/customer-a/delivery-resources")
    assert roots.package_output_root() == Path(
        "deploy/customer-a/artifacts/customer-project-packages"
    )
    assert roots.acceptance_dossier_output_root() == Path(
        "deploy/customer-a/artifacts/customer-project-acceptance-dossiers"
    )
    assert roots.proposal_output_root() == Path(
        "deploy/customer-a/artifacts/customer-project-proposals"
    )


def test_field_route_roots_keep_repo_artifacts_for_default_deploy_root() -> None:
    roots = build_field_route_roots(site_profile_root=DEFAULT_SITE_PROFILE_ROOT)

    assert roots.deploy_root() == DEFAULT_SITE_PROFILE_ROOT.parent
    assert roots.template_root() == DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT
    assert roots.delivery_resource_root() == DEFAULT_DELIVERY_RESOURCE_ROOT
    assert roots.package_output_root() == DEFAULT_CUSTOMER_PROJECT_PACKAGE_ROOT
    assert (
        roots.acceptance_dossier_output_root()
        == DEFAULT_CUSTOMER_PROJECT_ACCEPTANCE_DOSSIER_ROOT
    )
    assert roots.proposal_output_root() == DEFAULT_CUSTOMER_PROJECT_PROPOSAL_ROOT


def test_field_route_roots_explicit_overrides_win(tmp_path: Path) -> None:
    roots = build_field_route_roots(
        site_profile_root=tmp_path / "deploy" / "customer-a" / "site-profiles",
        customer_project_template_root=tmp_path / "templates",
        delivery_resource_root=tmp_path / "resources",
        customer_project_package_root=tmp_path / "packages",
        customer_project_acceptance_dossier_root=tmp_path / "dossiers",
        customer_project_proposal_root=tmp_path / "proposals",
    )

    assert roots.template_root() == tmp_path / "templates"
    assert roots.delivery_resource_root() == tmp_path / "resources"
    assert roots.package_output_root() == tmp_path / "packages"
    assert roots.acceptance_dossier_output_root() == tmp_path / "dossiers"
    assert roots.proposal_output_root() == tmp_path / "proposals"


def test_field_operations_path_roots_resolve_legacy_site_profile_path(tmp_path: Path) -> None:
    roots = field_operations_path_roots(
        {
            "field_operations": {
                "site_profile_path": "deploy/customer-a/site-profiles/site-a.yaml",
                "customer_project_package_root": str(tmp_path / "packages"),
            }
        },
        project_root=tmp_path,
    )

    assert roots["site_profile_root"] == tmp_path / "deploy" / "customer-a" / "site-profiles"
    assert roots["customer_project_template_root"] == DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT
    assert roots["delivery_resource_root"] == DEFAULT_DELIVERY_RESOURCE_ROOT
    assert roots["customer_project_package_root"] == tmp_path / "packages"
    assert (
        roots["customer_project_acceptance_dossier_root"]
        == DEFAULT_CUSTOMER_PROJECT_ACCEPTANCE_DOSSIER_ROOT
    )
    assert roots["customer_project_proposal_root"] == DEFAULT_CUSTOMER_PROJECT_PROPOSAL_ROOT
