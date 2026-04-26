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

import logging
import sys
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Annotated, Any, Generic, TypeVar, get_args, get_origin, get_type_hints

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


class AmbiguousPortError(ValueError):
    """Raised when a type has multiple Out providers and no Alias is used.

    When two or more modules declare ``Out[SomeType]`` and a third module
    declares ``In[SomeType]``, type-only (semantic) wiring cannot choose one
    provider without guessing.  Decorate the In port with ``Alias`` to pick
    the intended provider explicitly::

        class MyModule(Module):
            # Ambiguous — raises AmbiguousPortError at build time:
            client: In[LLMClient]

            # Explicit — wires to the module named "primary_llm":
            client: Alias[LLMClient, "primary_llm"]
    """


@dataclass
class _AliasMarker:
    """Internal metadata attached to Alias[T, 'name'] annotations."""
    provider_name: str


class Alias:
    """Explicitly select a named provider when multiple Out[T] exist.

    Usage::

        class MyModule(Module):
            # Two LLMClient providers exist → use the one from "fast_llm" module:
            fast_client: Alias[LLMClient, "fast_llm"]
            slow_client: Alias[LLMClient, "slow_llm"]

    ``Alias[T, "name"]`` produces ``Annotated[T, _AliasMarker("name")]`` so
    standard type-checking tools see the base type ``T``.

    Wiring rules:
      - Finds the module named *name* that exposes ``Out[T]``.
      - Raises ``ValueError`` if no such module/port exists.
      - Raises ``AmbiguousPortError`` if there are multiple providers of T and
        an ``In[T]`` (without Alias) is used anywhere in the runtime.
    """

    def __class_getitem__(cls, params: tuple) -> Any:  # type: ignore[override]
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError(
                f"Alias requires exactly two parameters: Alias[Type, 'name'], got {params!r}"
            )
        t, name = params
        if not isinstance(name, str):
            raise TypeError(
                f"Alias provider name must be a string literal, got {name!r}"
            )
        return Annotated[t, _AliasMarker(name)]


@dataclass(frozen=True)
class PortInfo:
    """Metadata about a declared port on a module."""

    name: str
    direction: str  # "in", "out", "required_in", or "alias_in"
    data_type: type
    module_name: str
    alias_name: str | None = None  # set for Alias[T, "name"] ports


def _scan_ports(module_class: type[Module]) -> list[PortInfo]:
    """Scan a Module subclass for In[T]/Out[T]/Required[T] annotations."""
    ports: list[PortInfo] = []
    try:
        globalns: dict[str, Any] = {}
        for c in reversed(module_class.__mro__):
            if c.__module__ in sys.modules:
                globalns.update(sys.modules[c.__module__].__dict__)
        # include_extras=True preserves Annotated[T, metadata] so Alias markers
        # are not stripped; without it Annotated[T, ...] collapses to T.
        hints = get_type_hints(module_class, globalns=globalns, include_extras=True)
    except Exception:
        hints = getattr(module_class, "__annotations__", {})

    for attr_name, annotation in hints.items():
        origin = get_origin(annotation)

        # Alias[T, "name"] is encoded as Annotated[T, _AliasMarker("name")]
        if origin is Annotated:
            ann_args = get_args(annotation)
            base_type = ann_args[0] if ann_args else Any
            alias_marker = next(
                (a for a in ann_args[1:] if isinstance(a, _AliasMarker)), None
            )
            if alias_marker is not None:
                ports.append(
                    PortInfo(attr_name, "alias_in", base_type, module_class.name,
                             alias_name=alias_marker.provider_name)
                )
                continue

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
    """Match In/Required/Alias ports to Out ports. Supports:

    1. **Exact match**: name + type both match (highest priority).
       Raises ``AmbiguousPortError`` when two modules share the same Out port
       name and type — use ``Alias`` to select one.
    2. **Semantic match**: type matches but name differs — only when exactly one
       Out[T] provider exists (unambiguous).
    3. **Alias match**: ``Alias[T, "module_name"]`` — explicitly selects Out[T]
       from the named module; safe when multiple providers exist.
    4. **Ambiguity guard**: ``In[T]`` / ``Required[T]`` when ≥2 providers exist
       raises ``AmbiguousPortError`` instead of silently doing nothing.
    5. **Required enforcement**: ``Required[T]`` with no match raises ValueError.
    6. **Orphan detection**: Out ports with no subscriber are flagged.
    """
    # Index Out ports by (port_name, data_type) → list[(mod_name, mod)]
    # Allows multiple providers; ambiguity is flagged at wiring time, not here.
    out_by_name_type: dict[tuple[str, type], list[tuple[str, Module]]] = {}
    # Index by data_type → list[(port_name, mod_name, mod)] for semantic match
    out_by_type: dict[type, list[tuple[str, str, Module]]] = {}
    # Index by (module_name, data_type) → (port_name, mod) for Alias lookup
    out_by_mod_type: dict[tuple[str, type], tuple[str, Module]] = {}

    for mod_name, mod in instances.items():
        for port in _scan_ports(type(mod)):
            if port.direction == "out":
                key = (port.name, port.data_type)
                out_by_name_type.setdefault(key, []).append((mod_name, mod))
                out_by_type.setdefault(port.data_type, []).append(
                    (port.name, mod_name, mod)
                )
                out_by_mod_type[(mod_name, port.data_type)] = (port.name, mod)

    wired: list[tuple[str, str, str]] = []
    unwired_optional: list[tuple[str, str, str]] = []
    semantic_matches: list[tuple[str, str, str, str]] = []
    consumed_outs: set[tuple[str, type]] = set()

    # Collect all In/Required/Alias ports
    in_ports: list[tuple[str, Module, PortInfo]] = []
    for mod_name, mod in instances.items():
        for port in _scan_ports(type(mod)):
            if port.direction in ("in", "required_in", "alias_in"):
                in_ports.append((mod_name, mod, port))

    for mod_name, mod, port in in_ports:

        # ── 0. Alias[T, "provider_name"] — explicit named lookup ──────────
        if port.direction == "alias_in":
            provider_name = port.alias_name  # guaranteed non-None for alias_in
            alias_match = out_by_mod_type.get((provider_name, port.data_type))
            if alias_match:
                out_port_name, out_mod = alias_match
                setattr(mod, port.name, out_mod)
                wired.append((mod_name, port.name, provider_name))
                consumed_outs.add((out_port_name, port.data_type))
                logger.debug(
                    "Wired (alias): %s.%s ← %s.%s (%s)",
                    mod_name, port.name, provider_name, out_port_name,
                    port.data_type.__name__,
                )
            else:
                raise ValueError(
                    f"Alias port '{mod_name}.{port.name}' ({port.data_type.__name__}) "
                    f"references provider '{provider_name}' but no matching "
                    f"Out[{port.data_type.__name__}] was found on that module — cannot build"
                )
            continue

        # ── 1. Exact match (name + type) ──────────────────────────────────
        key = (port.name, port.data_type)
        exact_providers = out_by_name_type.get(key, [])

        if len(exact_providers) == 1:
            out_mod_name, out_mod = exact_providers[0]
            setattr(mod, port.name, out_mod)
            wired.append((mod_name, port.name, out_mod_name))
            consumed_outs.add(key)
            logger.debug("Wired (exact): %s.%s ← %s.%s", mod_name, port.name, out_mod_name, port.name)
            continue

        if len(exact_providers) > 1:
            provider_list = ", ".join(f"'{mn}'" for mn, _ in exact_providers)
            raise AmbiguousPortError(
                f"Ambiguous port '{mod_name}.{port.name}' ({port.data_type.__name__}): "
                f"{len(exact_providers)} modules export the same port name: {provider_list}. "
                f"Use Alias[{port.data_type.__name__}, 'module_name'] to select one explicitly."
            )

        # ── 2. Semantic match (same type, unambiguous) ────────────────────
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

        # ── 3. Ambiguity guard ────────────────────────────────────────────
        # Multiple type-only (semantic) providers exist with NO matching name.
        # Unlike exact-name conflicts, we skip rather than raise: the In port
        # simply stays None.  Log a warning so developers know to use Alias.
        if len(type_providers) > 1:
            provider_list = ", ".join(
                f"'{mn}.{pn}'" for pn, mn, _ in type_providers
            )
            logger.warning(
                "[autowire] Ambiguous semantic match for %s.%s (%s): %d providers [%s]. "
                "Port left unset — use Alias[%s, 'module_name'] to wire explicitly.",
                mod_name, port.name, port.data_type.__name__,
                len(type_providers), provider_list, port.data_type.__name__,
            )
            # Fall through to "No match" path — port stays None

        # ── 4. No match ───────────────────────────────────────────────────
        if port.direction == "required_in":
            raise ValueError(
                f"Required port '{mod_name}.{port.name}' ({port.data_type.__name__}) "
                f"has no matching Out — cannot build"
            )

        setattr(mod, port.name, None)
        unwired_optional.append((mod_name, port.name, port.data_type.__name__))
        logger.debug("Unwired (optional): %s.%s (%s)", mod_name, port.name, port.data_type.__name__)

    # ── 5. Detect orphan Outs (no subscriber) ────────────────────────────
    orphan_outs: list[tuple[str, str, str]] = []
    for (port_name, data_type), providers in out_by_name_type.items():
        for mod_name, _ in providers:
            if (port_name, data_type) not in consumed_outs:
                orphan_outs.append((mod_name, port_name, data_type.__name__))
                logger.warning(
                    "Orphan Out: %s.%s (%s) — no subscriber",
                    mod_name, port_name, data_type.__name__,
                )

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

    def __init__(self) -> None:
        # Pre-initialise every declared In/Required/Alias port to None so that
        # direct attribute access is safe both (a) under Runtime.build(), where
        # _auto_wire() later overwrites the attribute with the provider module,
        # and (b) under standalone instantiation (tests, REPL) where auto-wiring
        # never runs.  Without this, subclass build() methods have to guard
        # every port access with ``getattr(self, "port_name", None)``.
        for port in _scan_ports(type(self)):
            if port.direction in ("in", "required_in", "alias_in"):
                if not hasattr(self, port.name):
                    setattr(self, port.name, None)

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
            mod.build(cfg, registry)
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
        """Hot-swap a running module.

        Replaces *module_name* with a freshly built instance of *new_module_class*
        without restarting the entire pipeline.  Steps:

        1. Stop the existing module gracefully.
        2. Build a new instance via ``new_module_class.build(cfg, registry)``.
        3. Re-wire all In-ports that were wired to/from the old module.
        4. Start the new module.

        .. note::
            Not yet implemented — contributions welcome.
            Callers may use :meth:`stop` / re-:meth:`start` as a workaround.
        """
        raise NotImplementedError(
            f"hot_swap('{module_name}') is not yet implemented. "
            "Use stop() followed by start() as a workaround."
        )

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
