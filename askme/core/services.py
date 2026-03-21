"""Service registry for dependency injection.

Modules register themselves as named services. Tools with ``requires``
can request injection of these services at registration time.

Usage::

    # During app init:
    services.register("vision_bridge", self.vision)
    services.register("audio_agent", self.audio)

    # During tool registration:
    deps = services.resolve(tool_spec.requires)
    # → {"vision_bridge": <VisionBridge instance>}
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_services: dict[str, Any] = {}


def register(name: str, instance: Any) -> None:
    """Register a service instance by name."""
    _services[name] = instance
    logger.debug("[Services] Registered: %s", name)


def get(name: str) -> Any | None:
    """Get a registered service by name. Returns None if not found."""
    return _services.get(name)


def resolve(names: list[str]) -> dict[str, Any]:
    """Resolve multiple service names to instances.

    Returns a dict of name → instance. Logs warning for missing services.
    """
    result = {}
    for name in names:
        svc = _services.get(name)
        if svc is None:
            logger.warning("[Services] Required service '%s' not registered", name)
        else:
            result[name] = svc
    return result


def clear() -> None:
    """Clear all registered services (for testing)."""
    _services.clear()
