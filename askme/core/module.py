"""Module base class — self-describing, lifecycle-managed components.

Every major askme subsystem (vision, voice, perception, memory, agent)
inherits from Module. Modules declare their dependencies, register
themselves, and have standard lifecycle hooks.

Inspired by DimOS Module but adapted to askme's async architecture.

Usage::

    class VisionModule(Module):
        name = "vision"
        depends_on = ["frame_daemon"]

        async def start(self):
            self.vision_bridge = VisionBridge()

        async def stop(self):
            pass
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


class Module:
    """Base class for all askme subsystem modules.

    Subclasses set class-level ``name`` and ``depends_on``, then
    implement ``start()`` and optionally ``stop()``.
    """

    # Override in subclass
    name: ClassVar[str] = ""
    depends_on: ClassVar[list[str]] = []

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._started = False

    async def start(self) -> None:
        """Initialize the module. Called after dependencies are started."""
        pass

    async def stop(self) -> None:
        """Graceful shutdown. Called in reverse dependency order."""
        pass

    @property
    def is_started(self) -> bool:
        return self._started

    def __repr__(self) -> str:
        status = "started" if self._started else "stopped"
        return f"<{self.__class__.__name__}({self.name}) {status}>"


# Global module registry
_MODULE_REGISTRY: dict[str, type[Module]] = {}


def module(name: str, depends_on: list[str] | None = None):
    """Class decorator to register a Module subclass.

    Usage::

        @module("vision", depends_on=["frame_daemon"])
        class VisionModule(Module):
            async def start(self):
                ...
    """
    def decorator(cls):
        cls.name = name
        cls.depends_on = depends_on or []
        _MODULE_REGISTRY[name] = cls
        return cls
    return decorator


def get_registered_modules() -> dict[str, type[Module]]:
    return dict(_MODULE_REGISTRY)
