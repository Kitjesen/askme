"""Orchestrator — topological module startup, lifecycle management.

Replaces the monolithic app.py wiring with dependency-aware module
initialization. Modules declare dependencies, orchestrator resolves
the startup order automatically.

Usage::

    orch = Orchestrator(config)
    orch.register(VisionModule)
    orch.register(PerceptionModule)  # depends_on=["vision"]
    orch.register(AgentModule)       # depends_on=["perception", "voice"]
    await orch.start_all()
    ...
    await orch.stop_all()
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from askme.core.module import Module, get_registered_modules
from askme.core import services

logger = logging.getLogger(__name__)


class Orchestrator:
    """Manages module lifecycle with dependency-aware ordering."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._module_classes: dict[str, type[Module]] = {}
        self._instances: dict[str, Module] = {}
        self._start_order: list[str] = []

    def register(self, module_cls: type[Module]) -> None:
        """Register a module class for lifecycle management."""
        name = module_cls.name
        if not name:
            raise ValueError(f"{module_cls.__name__} has no 'name' set")
        self._module_classes[name] = module_cls
        logger.debug("[Orchestrator] Registered module: %s", name)

    def register_all(self) -> None:
        """Register all @module-decorated classes found by auto-discovery."""
        for name, cls in get_registered_modules().items():
            if name not in self._module_classes:
                self.register(cls)

    async def start_all(self) -> None:
        """Start all modules in topological (dependency) order."""
        self._start_order = self._topological_sort()
        logger.info(
            "[Orchestrator] Starting %d modules: %s",
            len(self._start_order), " → ".join(self._start_order),
        )

        for name in self._start_order:
            cls = self._module_classes[name]
            # Extract module-specific config section
            mod_config = self._config.get(name, {})
            instance = cls(config=mod_config)
            self._instances[name] = instance

            # Register as service for dependency injection
            services.register(name, instance)

            try:
                await instance.start()
                instance._started = True
                logger.info("[Orchestrator] ✓ %s started", name)
            except Exception as exc:
                logger.error("[Orchestrator] ✗ %s failed to start: %s", name, exc)
                raise

    async def stop_all(self) -> None:
        """Stop all modules in reverse startup order."""
        for name in reversed(self._start_order):
            instance = self._instances.get(name)
            if instance and instance.is_started:
                try:
                    await instance.stop()
                    instance._started = False
                    logger.info("[Orchestrator] ✓ %s stopped", name)
                except Exception as exc:
                    logger.warning("[Orchestrator] ✗ %s stop error: %s", name, exc)

    def get(self, name: str) -> Module | None:
        """Get a started module instance by name."""
        return self._instances.get(name)

    def _topological_sort(self) -> list[str]:
        """Sort modules by dependency order (Kahn's algorithm).

        Raises ValueError if circular dependency detected.
        """
        # Build adjacency and in-degree
        in_degree: dict[str, int] = {name: 0 for name in self._module_classes}
        dependents: dict[str, list[str]] = {name: [] for name in self._module_classes}

        for name, cls in self._module_classes.items():
            for dep in cls.depends_on:
                if dep not in self._module_classes:
                    logger.warning(
                        "[Orchestrator] Module '%s' depends on '%s' which is not registered — skipping",
                        name, dep,
                    )
                    continue
                dependents[dep].append(name)
                in_degree[name] += 1

        # Start with modules that have no dependencies
        queue = [name for name, deg in in_degree.items() if deg == 0]
        result: list[str] = []

        while queue:
            # Sort for deterministic order
            queue.sort()
            node = queue.pop(0)
            result.append(node)

            for dependent in dependents[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._module_classes):
            missing = set(self._module_classes) - set(result)
            raise ValueError(f"Circular dependency detected among: {missing}")

        return result

    @property
    def started_modules(self) -> list[str]:
        return [n for n, m in self._instances.items() if m.is_started]
