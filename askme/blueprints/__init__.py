"""Product runtime blueprint package.

Owner subpackages:
- ``catalog``: customer-visible blueprint catalog and readiness metadata.
- ``presets``: concrete runtime blueprint presets.
- ``runner``: CLI/runtime launch helper.

Historical imports such as ``askme.blueprints.voice`` remain available for
compatibility. New code should import from the owner subpackage.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from askme.compat.imports import install_legacy_aliases

_LEGACY_MODULE_ALIASES = {
    "askme.blueprints._runner": "askme.blueprints.runner.runner",
    "askme.blueprints.edge_robot": "askme.blueprints.presets.edge_robot",
    "askme.blueprints.lingtu_voice": "askme.blueprints.presets.lingtu_voice",
    "askme.blueprints.mcp": "askme.blueprints.presets.mcp",
    "askme.blueprints.text": "askme.blueprints.presets.text",
    "askme.blueprints.voice": "askme.blueprints.presets.voice",
    "askme.blueprints.voice_perception": "askme.blueprints.presets.voice_perception",
}

_LAZY_EXPORTS = {
    "BLUEPRINTS": ("askme.blueprints.catalog.data", "BLUEPRINTS"),
    "BlueprintSpec": ("askme.blueprints.catalog.models", "BlueprintSpec"),
    "blueprint_configuration_summary": (
        "askme.blueprints.catalog.catalog",
        "blueprint_configuration_summary",
    ),
    "blueprint_delivery_package": (
        "askme.blueprints.catalog.catalog",
        "blueprint_delivery_package",
    ),
    "blueprint_readiness": ("askme.blueprints.catalog.catalog", "blueprint_readiness"),
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


install_legacy_aliases(__name__, _LEGACY_MODULE_ALIASES)


def __getattr__(name: str) -> Any:
    legacy_module = _LEGACY_MODULE_ALIASES.get(f"{__name__}.{name}")
    if legacy_module:
        value = import_module(legacy_module)
        globals()[name] = value
        return value

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
