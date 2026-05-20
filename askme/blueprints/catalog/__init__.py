"""Blueprint catalog and readiness metadata."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "ALIASES": ("askme.blueprints.catalog.data", "ALIASES"),
    "BLUEPRINTS": ("askme.blueprints.catalog.data", "BLUEPRINTS"),
    "BlueprintSpec": ("askme.blueprints.catalog.models", "BlueprintSpec"),
    "EDGE_ROBOT_MODULES": ("askme.blueprints.catalog.data", "EDGE_ROBOT_MODULES"),
    "LINGTU_VOICE_MODULES": (
        "askme.blueprints.catalog.data",
        "LINGTU_VOICE_MODULES",
    ),
    "MCP_MODULES": ("askme.blueprints.catalog.data", "MCP_MODULES"),
    "TEXT_MODULES": ("askme.blueprints.catalog.data", "TEXT_MODULES"),
    "VOICE_MODULES": ("askme.blueprints.catalog.data", "VOICE_MODULES"),
    "VOICE_PERCEPTION_MODULES": (
        "askme.blueprints.catalog.data",
        "VOICE_PERCEPTION_MODULES",
    ),
    "blueprint_delivery_package": (
        "askme.blueprints.catalog.catalog",
        "blueprint_delivery_package",
    ),
    "blueprint_configuration_summary": (
        "askme.blueprints.catalog.catalog",
        "blueprint_configuration_summary",
    ),
    "blueprint_readiness": (
        "askme.blueprints.catalog.catalog",
        "blueprint_readiness",
    ),
    "catalog_payload": ("askme.blueprints.catalog.catalog", "catalog_payload"),
    "get_blueprint_spec": ("askme.blueprints.catalog.catalog", "get_blueprint_spec"),
    "inspect_blueprint": ("askme.blueprints.catalog.catalog", "inspect_blueprint"),
    "list_blueprints": ("askme.blueprints.catalog.catalog", "list_blueprints"),
    "load_runtime_blueprint_for_modes": (
        "askme.blueprints.catalog.catalog",
        "load_runtime_blueprint_for_modes",
    ),
    "load_blueprint_runtime": (
        "askme.blueprints.catalog.catalog",
        "load_blueprint_runtime",
    ),
    "resolve_runtime_blueprint_for_modes": (
        "askme.blueprints.catalog.catalog",
        "resolve_runtime_blueprint_for_modes",
    ),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve blueprint catalog contracts on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
