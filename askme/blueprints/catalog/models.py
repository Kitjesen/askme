"""Blueprint catalog data models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BlueprintSpec:
    """Customer and delivery metadata for one runtime blueprint."""

    name: str
    title: str
    description: str
    import_path: str
    object_name: str
    startup_command: str
    product_stage: str
    primary_loop: str
    customer_visible: bool
    deployment_targets: tuple[str, ...]
    modules: tuple[str, ...]
    capabilities: tuple[str, ...]
    scenarios: tuple[str, ...]
    required_config: tuple[str, ...]
    external_services: tuple[str, ...]
    safety_boundaries: tuple[str, ...]
    validation_commands: tuple[str, ...]
    config_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, tuple):
                payload[key] = list(value)
            if key == "config_aliases" and isinstance(value, dict):
                payload[key] = {
                    alias: list(paths) if isinstance(paths, tuple) else paths
                    for alias, paths in value.items()
                }
        return payload
