"""Compatibility helpers for package migrations."""

from askme.compat.imports import install_legacy_aliases
from askme.compat.legacy_facades import (
    LEGACY_FACADE_BY_PATH,
    LEGACY_FACADES,
    LegacyFacade,
    legacy_facade_for,
)

__all__ = [
    "LEGACY_FACADE_BY_PATH",
    "LEGACY_FACADES",
    "LegacyFacade",
    "install_legacy_aliases",
    "legacy_facade_for",
]
