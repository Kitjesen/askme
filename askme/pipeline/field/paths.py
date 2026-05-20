"""Repository-anchored path defaults for field delivery modules."""

from __future__ import annotations

from pathlib import Path

from askme.config import project_root

PROJECT_ROOT = project_root()
DEFAULT_SITE_PROFILE_ROOT = PROJECT_ROOT / "deploy" / "site-profiles"
DEFAULT_CUSTOMER_PROJECT_TEMPLATE_ROOT = PROJECT_ROOT / "deploy" / "customer-project-templates"
DEFAULT_DELIVERY_RESOURCE_ROOT = PROJECT_ROOT / "deploy" / "delivery-resources"
DEFAULT_CUSTOMER_PROJECT_PACKAGE_ROOT = PROJECT_ROOT / "artifacts" / "customer-project-packages"
DEFAULT_CUSTOMER_PROJECT_ACCEPTANCE_DOSSIER_ROOT = (
    PROJECT_ROOT / "artifacts" / "customer-project-acceptance-dossiers"
)
DEFAULT_CUSTOMER_PROJECT_PROPOSAL_ROOT = PROJECT_ROOT / "artifacts" / "customer-project-proposals"


def project_path(path: str | Path) -> Path:
    """Resolve a repo-relative field path against the askme project root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate
