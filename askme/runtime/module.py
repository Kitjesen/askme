"""Declarative module system for askme runtime.

Inspired by DimOS Blueprint but lighter — no automatic stream wiring,
just declarative dependency resolution and one-line composition.

Usage::

    from askme.runtime.module import Module, Runtime

    class MyPerception(Module):
        name = "perception"
        depends_on = ("pulse",)
        provides = ("detections", "world_state")

        def build(self, cfg, services):
            self.detector = ChangeDetector(cfg, pulse=services.pulse)

        async def start(self):
            await self.detector.start()

        async def stop(self):
            await self.detector.stop()

    # Compose:
    runtime = Runtime.use(Pulse) + Runtime.use(MyPerception)

    # Swap:
    runtime = runtime.replace(Pulse, MockPulse)

    # Build and run:
    app = await runtime.build(cfg)
    await app.start()
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class Module(ABC):
    """Base class for all askme runtime modules.

    Subclasses declare their identity and dependencies as class attributes,
    then implement build/start/stop/health.
    """

    # ── Override these in subclasses ──
    name: str = ""
    description: str = ""
    depends_on: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.name and cls.__name__ != "Module":
            # Auto-name from class: MyPerception → my_perception
            cls.name = _camel_to_snake(cls.__name__)

    @abstractmethod
    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        """Construct internal state from config. Called once before start."""

    async def start(self) -> None:
        """Start the module. Override if needed."""

    async def stop(self) -> None:
        """Stop the module. Override if needed."""

    def health(self) -> dict[str, Any]:
        """Return health snapshot. Override for richer data."""
        return {"status": "ok"}

    def capabilities(self) -> dict[str, Any]:
        """Return capabilities snapshot."""
        return {}


class ModuleRegistry:
    """Holds built modules — passed to build() so modules can find dependencies."""

    def __init__(self) -> None:
        self._modules: dict[str, Module] = {}

    def register(self, module: Module) -> None:
        self._modules[module.name] = module

    def get(self, name: str) -> Module | None:
        return self._modules.get(name)

    def __getattr__(self, name: str) -> Module:
        if name.startswith("_"):
            raise AttributeError(name)
        mod = self._modules.get(name)
        if mod is None:
            raise AttributeError(f"Module '{name}' not registered")
        return mod

    def __contains__(self, name: str) -> bool:
        return name in self._modules

    def items(self) -> Any:
        return self._modules.items()

    def keys(self) -> Any:
        return self._modules.keys()


@dataclass(frozen=True)
class Runtime:
    """Declarative runtime composition — add modules with + operator.

    Example::

        rt = Runtime.use(Pulse) + Runtime.use(VoiceIO) + Runtime.use(Perception)
        app = await rt.build(cfg)
        await app.start()
    """

    _module_classes: tuple[type[Module], ...] = ()

    @classmethod
    def use(cls, module_class: type[Module]) -> Runtime:
        """Create a Runtime with a single module."""
        return cls(_module_classes=(module_class,))

    def __add__(self, other: Runtime) -> Runtime:
        """Compose two runtimes — merge their module lists."""
        # Deduplicate by name, keep last (allows override)
        seen: dict[str, type[Module]] = {}
        for mc in self._module_classes + other._module_classes:
            seen[mc.name] = mc
        return Runtime(_module_classes=tuple(seen.values()))

    def replace(self, old: type[Module], new: type[Module]) -> Runtime:
        """Replace a module class — same name, different implementation."""
        replaced = tuple(
            new if mc.name == old.name else mc
            for mc in self._module_classes
        )
        return Runtime(_module_classes=replaced)

    def without(self, module_class: type[Module]) -> Runtime:
        """Remove a module by class."""
        filtered = tuple(
            mc for mc in self._module_classes if mc.name != module_class.name
        )
        return Runtime(_module_classes=filtered)

    async def build(self, cfg: dict[str, Any] | None = None) -> RuntimeApp:
        """Resolve dependencies, build all modules, return a startable app."""
        cfg = cfg or {}
        registry = ModuleRegistry()

        # Instantiate all modules
        instances: dict[str, Module] = {}
        for mc in self._module_classes:
            instance = mc()
            instances[instance.name] = instance

        # Topological sort by depends_on
        order = _topo_sort(instances)

        # Build in dependency order
        for name in order:
            mod = instances[name]
            registry.register(mod)
            mod.build(cfg.get(name, cfg), registry)
            logger.debug("Module built: %s", name)

        return RuntimeApp(modules=instances, start_order=order)


@dataclass
class RuntimeApp:
    """A built runtime — start/stop in dependency order."""

    modules: dict[str, Module]
    start_order: list[str]
    _started: bool = field(default=False, init=False)

    async def start(self) -> None:
        if self._started:
            return
        for name in self.start_order:
            await self.modules[name].start()
            logger.info("Module started: %s", name)
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        for name in reversed(self.start_order):
            try:
                await self.modules[name].stop()
            except Exception as e:
                logger.warning("Module %s stop error: %s", name, e)
        self._started = False

    def health(self) -> dict[str, dict]:
        return {name: mod.health() for name, mod in self.modules.items()}

    def get(self, name: str) -> Module | None:
        return self.modules.get(name)

    def __getattr__(self, name: str) -> Module:
        if name.startswith("_") or name in ("modules", "start_order"):
            raise AttributeError(name)
        mod = self.modules.get(name)
        if mod is None:
            raise AttributeError(f"Module '{name}' not in runtime")
        return mod


def _topo_sort(modules: dict[str, Module]) -> list[str]:
    """Topological sort of modules by depends_on."""
    in_degree: dict[str, int] = {n: 0 for n in modules}
    dependents: dict[str, list[str]] = {n: [] for n in modules}

    for name, mod in modules.items():
        for dep in mod.depends_on:
            if dep not in modules:
                continue  # optional dependency
            in_degree[name] += 1
            dependents[dep].append(name)

    queue: deque[str] = deque(n for n, d in in_degree.items() if d == 0)
    order: list[str] = []
    while queue:
        cur = queue.popleft()
        order.append(cur)
        for dep in dependents[cur]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    if len(order) != len(modules):
        raise ValueError(f"Cycle: {set(modules) - set(order)}")
    return order


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    result = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            result.append("_")
        result.append(ch.lower())
    return "".join(result)
