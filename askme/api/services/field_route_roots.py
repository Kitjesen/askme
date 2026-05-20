"""Route-root providers for the Field API composition layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from askme.pipeline.field.paths import (
    DEFAULT_CUSTOMER_PROJECT_ACCEPTANCE_DOSSIER_ROOT,
    DEFAULT_CUSTOMER_PROJECT_PACKAGE_ROOT,
    DEFAULT_CUSTOMER_PROJECT_PROPOSAL_ROOT,
    DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT,
    DEFAULT_DELIVERY_RESOURCE_ROOT,
    DEFAULT_SITE_PROFILE_ROOT,
)


@dataclass(frozen=True)
class FieldRouteRoots:
    """Resolve customer-project route roots without coupling routes to defaults."""

    site_profile_root_override: Path | None = None
    customer_project_template_root_override: Path | None = None
    delivery_resource_root_override: Path | None = None
    customer_project_package_root_override: Path | None = None
    customer_project_acceptance_dossier_root_override: Path | None = None
    customer_project_proposal_root_override: Path | None = None

    def site_profile_root(self) -> Path:
        return self.site_profile_root_override or DEFAULT_SITE_PROFILE_ROOT

    def deploy_root(self) -> Path:
        root = self.site_profile_root()
        if root.name == "site-profiles":
            return root.parent
        return root.parent if root.parent.name == "deploy" else root

    def template_root(self) -> Path:
        if self.customer_project_template_root_override is not None:
            return self.customer_project_template_root_override
        if self.site_profile_root_override is None:
            return DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT
        return self.deploy_root() / "customer-project-templates"

    def delivery_resource_root(self) -> Path:
        if self.delivery_resource_root_override is not None:
            return self.delivery_resource_root_override
        if self.site_profile_root_override is None:
            return DEFAULT_DELIVERY_RESOURCE_ROOT
        return self.deploy_root() / "delivery-resources"

    def artifact_output_root(self, root_name: str) -> Path:
        deploy_root = self.deploy_root()
        base = deploy_root.parent if deploy_root.name == "deploy" else deploy_root
        return base / "artifacts" / root_name

    def package_output_root(self) -> Path:
        if self.customer_project_package_root_override is not None:
            return self.customer_project_package_root_override
        if self.site_profile_root_override is None:
            return DEFAULT_CUSTOMER_PROJECT_PACKAGE_ROOT
        return self.artifact_output_root(DEFAULT_CUSTOMER_PROJECT_PACKAGE_ROOT.name)

    def acceptance_dossier_output_root(self) -> Path:
        if self.customer_project_acceptance_dossier_root_override is not None:
            return self.customer_project_acceptance_dossier_root_override
        if self.site_profile_root_override is None:
            return DEFAULT_CUSTOMER_PROJECT_ACCEPTANCE_DOSSIER_ROOT
        return self.artifact_output_root(DEFAULT_CUSTOMER_PROJECT_ACCEPTANCE_DOSSIER_ROOT.name)

    def proposal_output_root(self) -> Path:
        if self.customer_project_proposal_root_override is not None:
            return self.customer_project_proposal_root_override
        if self.site_profile_root_override is None:
            return DEFAULT_CUSTOMER_PROJECT_PROPOSAL_ROOT
        return self.artifact_output_root(DEFAULT_CUSTOMER_PROJECT_PROPOSAL_ROOT.name)


def build_field_route_roots(
    *,
    site_profile_root: Path | None = None,
    customer_project_template_root: Path | None = None,
    delivery_resource_root: Path | None = None,
    customer_project_package_root: Path | None = None,
    customer_project_acceptance_dossier_root: Path | None = None,
    customer_project_proposal_root: Path | None = None,
) -> FieldRouteRoots:
    """Build route-root providers from optional app-level overrides."""

    return FieldRouteRoots(
        site_profile_root_override=site_profile_root,
        customer_project_template_root_override=customer_project_template_root,
        delivery_resource_root_override=delivery_resource_root,
        customer_project_package_root_override=customer_project_package_root,
        customer_project_acceptance_dossier_root_override=customer_project_acceptance_dossier_root,
        customer_project_proposal_root_override=customer_project_proposal_root,
    )


def config_path(value: Any, *, default: Path, project_root: Path) -> Path:
    """Resolve configured repo-relative paths against an explicit project root."""

    if value in (None, ""):
        return default
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


def field_operations_path_roots(
    app_config: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Path]:
    """Resolve Field API storage roots from app config."""

    field_cfg = app_config.get("field_operations", {}) if isinstance(app_config, dict) else {}
    if not isinstance(field_cfg, dict):
        field_cfg = {}
    site_profile_root_value = field_cfg.get("site_profile_root") or field_cfg.get(
        "site_profiles_root"
    )
    if not site_profile_root_value and field_cfg.get("site_profile_path"):
        site_profile_root_value = Path(str(field_cfg["site_profile_path"])).parent
    return {
        "site_profile_root": config_path(
            site_profile_root_value,
            default=DEFAULT_SITE_PROFILE_ROOT,
            project_root=project_root,
        ),
        "customer_project_template_root": config_path(
            field_cfg.get("customer_project_template_root"),
            default=DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT,
            project_root=project_root,
        ),
        "delivery_resource_root": config_path(
            field_cfg.get("delivery_resource_root"),
            default=DEFAULT_DELIVERY_RESOURCE_ROOT,
            project_root=project_root,
        ),
        "customer_project_package_root": config_path(
            field_cfg.get("customer_project_package_root"),
            default=DEFAULT_CUSTOMER_PROJECT_PACKAGE_ROOT,
            project_root=project_root,
        ),
        "customer_project_acceptance_dossier_root": config_path(
            field_cfg.get("customer_project_acceptance_dossier_root"),
            default=DEFAULT_CUSTOMER_PROJECT_ACCEPTANCE_DOSSIER_ROOT,
            project_root=project_root,
        ),
        "customer_project_proposal_root": config_path(
            field_cfg.get("customer_project_proposal_root"),
            default=DEFAULT_CUSTOMER_PROJECT_PROPOSAL_ROOT,
            project_root=project_root,
        ),
    }


__all__ = [
    "FieldRouteRoots",
    "build_field_route_roots",
    "config_path",
    "field_operations_path_roots",
]
