"""Declarative module system for askme runtime.

Modules declare typed ports via ``In[T]`` / ``Out[T]`` class annotations.
The runtime auto-wires matching ports by name + type during ``build()``.

Usage::

    from askme.runtime.module import Module, Runtime, In, Out

    class MyBus(Module):
        detections: Out[DetectionFrame]   # I produce detections
        estop: Out[EstopState]            # I produce estop state

        def build(self, cfg, registry):
            ...

    class MyPerception(Module):
        detections: In[DetectionFrame]    # I need detections

        def build(self, cfg, registry):
            # self.detections is auto-wired to MyBus.detections
            print(self.detections)  # → the Out port from MyBus

    # Compose — ports auto-connect:
    runtime = Runtime.use(MyBus) + Runtime.use(MyPerception)

    # Swap — same ports, different implementation:
    runtime = runtime.replace(MyBus, MockBus)

    # Build and run:
    app = await runtime.build(cfg)
    await app.start()
"""

from __future__ import annotations

import asyncio
import logging
import sys
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, get_args, get_origin, get_type_hints

logger = logging.getLogger(__name__)

# ── Typed port descriptors ────────────────────────────────────────────

T = TypeVar("T")


class Out(Generic[T]):
    """Output port — this module produces data of type T.

    Declare as a class annotation::

        class MyBus(Module):
            detections: Out[DetectionFrame]
    """


class In(Generic[T]):
    """Input port — this module consumes data of type T.

    Declare as a class annotation::

        class MyPerception(Module):
            detections: In[DetectionFrame]

    After build(), ``self.detections`` is auto-wired to the matching Out port.
    """


class Required(Generic[T]):
    """Required input port — build() fails if no matching Out exists.

    Same as In[T] but enforced at build time::

        class Planner(Module):
            scene: Required[SceneGraph]   # must have a provider
            weather: In[WeatherInfo]      # optional, None if missing
    """


@dataclass(frozen=True)
class PortInfo:
    """Metadata about a declared port on a module."""

    name: str
    direction: str  # "in", "out", or "required_in"
    data_type: type
    module_name: str


def _scan_ports(module_class: type[Module]) -> list[PortInfo]:
    """Scan a Module subclass for In[T]/Out[T]/Required[T] annotations."""
    ports: list[PortInfo] = []
    try:
        globalns: dict[str, Any] = {}
        for c in reversed(module_class.__mro__):
            if c.__module__ in sys.modules:
                globalns.update(sys.modules[c.__module__].__dict__)
        hints = get_type_hints(module_class, globalns=globalns)
    except Exception:
        hints = getattr(module_class, "__annotations__", {})

    for attr_name, annotation in hints.items():
        origin = get_origin(annotation)
        if origin is Out:
            args = get_args(annotation)
            data_type = args[0] if args else Any
            ports.append(PortInfo(attr_name, "out", data_type, module_class.name))
        elif origin is Required:
            args = get_args(annotation)
            data_type = args[0] if args else Any
            ports.append(PortInfo(attr_name, "required_in", data_type, module_class.name))
        elif origin is In:
            args = get_args(annotation)
            data_type = args[0] if args else Any
            ports.append(PortInfo(attr_name, "in", data_type, module_class.name))
    return ports


@dataclass
class WireResult:
    """Result of auto-wiring: connections made + diagnostics."""

    wired: list[tuple[str, str, str]]  # (in_mod, port_name, out_mod)
    unwired_optional: list[tuple[str, str, str]]  # (mod, port, type_name)
    orphan_outs: list[tuple[str, str, str]]  # (mod, port, type_name) — no subscriber
    semantic_matches: list[tuple[str, str, str, str]]  # (in_mod, in_port, out_mod, out_port)


def _auto_wire(instances: dict[str, Module]) -> WireResult:
    """Match In/Required ports to Out ports. Supports:

    1. **Exact match**: name + type both match (highest priority)
    2. **Semantic match**: type matches but name differs — only when the type has
       exactly one Out and one In across all modules (unambiguous)
    3. **Required enforcement**: Required[T] ports with no match raise ValueError
    4. **Orphan detection**: Out ports with no subscriber are flagged
    """
    # Collect all Out ports
    out_by_name_type: dict[tuple[str, type], tuple[str, Module]] = {}
    out_by_type: dict[type, list[tuple[str, str, Module]]] = {}  # type → [(port_name, mod_name, mod)]

    for mod_name, mod in instances.items():
        for port in _scan_ports(type(mod)):
            if port.direction == "out":
                key = (port.name, port.data_type)
                if key in out_by_name_type:
                    raise ValueError(
                        f"Ambiguous Out port '{port.name}' ({port.data_type.__name__}): "
                        f"provided by both '{out_by_name_type[key][0]}' and '{mod_name}'"
                    )
                out_by_name_type[key] = (mod_name, mod)
                out_by_type.setdefault(port.data_type, []).append(
                    (port.name, mod_name, mod)
                )

    wired: list[tuple[str, str, str]] = []
    unwired_optional: list[tuple[str, str, str]] = []
    semantic_matches: list[tuple[str, str, str, str]] = []
    consumed_outs: set[tuple[str, type]] = set()

    # Collect all In/Required ports
    in_ports: list[tuple[str, Module, PortInfo]] = []
    for mod_name, mod in instances.items():
        for port in _scan_ports(type(mod)):
            if port.direction in ("in", "required_in"):
                in_ports.append((mod_name, mod, port))

    for mod_name, mod, port in in_ports:
        # 1. Try exact match (name + type)
        key = (port.name, port.data_type)
        match = out_by_name_type.get(key)

        if match:
            out_mod_name, out_mod = match
            setattr(mod, port.name, out_mod)
            wired.append((mod_name, port.name, out_mod_name))
            consumed_outs.add(key)
            logger.debug("Wired (exact): %s.%s ← %s.%s", mod_name, port.name, out_mod_name, port.name)
            continue

        # 2. Try semantic match (same type, different name, unambiguous)
        type_providers = out_by_type.get(port.data_type, [])
        if len(type_providers) == 1:
            out_port_name, out_mod_name, out_mod = type_providers[0]
            setattr(mod, port.name, out_mod)
            wired.append((mod_name, port.name, out_mod_name))
            consumed_outs.add((out_port_name, port.data_type))
            semantic_matches.append((mod_name, port.name, out_mod_name, out_port_name))
            logger.info(
                "Wired (semantic): %s.%s ← %s.%s (%s) — names differ but type unique",
                mod_name, port.name, out_mod_name, out_port_name,
                port.data_type.__name__,
            )
            continue

        # 3. No match
        if port.direction == "required_in":
            raise ValueError(
                f"Required port '{mod_name}.{port.name}' ({port.data_type.__name__}) "
                f"has no matching Out — cannot build"
            )

        setattr(mod, port.name, None)
        unwired_optional.append((mod_name, port.name, port.data_type.__name__))
        logger.debug("Unwired (optional): %s.%s (%s)", mod_name, port.name, port.data_type.__name__)

    # 4. Detect orphan Outs (no subscriber)
    orphan_outs: list[tuple[str, str, str]] = []
    for (port_name, data_type), (mod_name, _) in out_by_name_type.items():
        if (port_name, data_type) not in consumed_outs:
            orphan_outs.append((mod_name, port_name, data_type.__name__))
            logger.warning("Orphan Out: %s.%s (%s) — no subscriber", mod_name, port_name, data_type.__name__)

    return WireResult(
        wired=wired,
        unwired_optional=unwired_optional,
        orphan_outs=orphan_outs,
        semantic_matches=semantic_matches,
    )


def _validate_topology(instances: dict[str, Module], wire_result: WireResult) -> list[str]:
    """Validate the wired topology. Returns list of warnings (empty = healthy).

    Checks:
    1. No Required ports left unwired (already enforced in _auto_wire)
    2. Orphan Outs flagged
    3. Reachability: every module with In ports has at least one wired connection
    """
    warnings: list[str] = []

    for mod_name, port_name, type_name in wire_result.orphan_outs:
        warnings.append(f"Orphan Out: {mod_name}.{port_name} ({type_name}) has no subscriber")

    # Check modules with In ports have at least one wired
    wired_consumers = {w[0] for w in wire_result.wired}
    for mod_name, mod in instances.items():
        in_ports = [p for p in _scan_ports(type(mod)) if p.direction in ("in", "required_in")]
        if in_ports and mod_name not in wired_consumers:
            all_optional = all(p.direction == "in" for p in in_ports)
            if not all_optional:
                warnings.append(f"Module '{mod_name}' has input ports but none are wired")

    return warnings


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
        """Resolve dependencies, auto-wire ports, validate topology, build."""
        cfg = cfg or {}
        registry = ModuleRegistry()

        # Instantiate all modules
        instances: dict[str, Module] = {}
        for mc in self._module_classes:
            instance = mc()
            instances[instance.name] = instance

        # Auto-wire In[T]/Required[T] ← Out[T] (exact + semantic match)
        wire_result = _auto_wire(instances)

        # Validate topology
        warnings = _validate_topology(instances, wire_result)
        for w in warnings:
            logger.warning("Topology: %s", w)

        # Topological sort by depends_on
        order = _topo_sort(instances)

        # Build in dependency order
        for name in order:
            mod = instances[name]
            registry.register(mod)
            mod.build(cfg.get(name, cfg), registry)
            logger.debug("Module built: %s", name)

        return RuntimeApp(
            modules=instances,
            start_order=order,
            wired_ports=wire_result.wired,
            wire_result=wire_result,
        )


@dataclass
class RuntimeApp:
    """A built runtime — start/stop in dependency order."""

    modules: dict[str, Module]
    start_order: list[str]
    wired_ports: list[tuple[str, str, str]] = field(default_factory=list)
    wire_result: WireResult | None = None
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

    async def hot_swap(self, module_name: str, new_module_class: type[Module], cfg: dict | None = None) -> None:
        """Hot-swap a running module. Not yet implemented.

        Will stop the old module, re-wire In ports, build and start the new one.
        """
        raise NotImplementedError("Hot-swap coming in a future release")

    def flow_stats(self) -> dict:
        """Per-port message flow stats."""
        return {
            "wired_ports": len(self.wired_ports),
            "note": "Detailed per-port stats coming soon",
        }

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
