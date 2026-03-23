"""Runtime component abstractions for askme assembly."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable


def _normalise_snapshot(
    payload: dict[str, Any] | None,
    *,
    default_status: str,
) -> dict[str, Any]:
    data = dict(payload or {})
    data.setdefault("status", default_status)
    return data


async def _call_maybe_async(func: Callable[[], Any] | None) -> Any:
    if func is None:
        return None
    result = func()
    if inspect.isawaitable(result):
        return await result
    return result


def resolve_start_order(
    components: dict[str, RuntimeComponent],
) -> list[str]:
    """Topological sort of components by *depends_on*.

    Returns component names in an order that satisfies all declared
    dependencies.  Components without dependencies come first.
    Raises ``ValueError`` on cycles.
    """
    # Build adjacency: name -> set of dependents
    in_degree: dict[str, int] = {name: 0 for name in components}
    dependents: dict[str, list[str]] = {name: [] for name in components}

    for name, comp in components.items():
        for dep in getattr(comp, "depends_on", ()):
            if dep not in components:
                # Dependency not registered — skip silently (optional dep).
                continue
            in_degree[name] += 1
            dependents[dep].append(name)

    queue: deque[str] = deque(
        name for name, degree in in_degree.items() if degree == 0
    )
    ordered: list[str] = []
    while queue:
        current = queue.popleft()
        ordered.append(current)
        for dependent in dependents[current]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(ordered) != len(components):
        missing = set(components) - set(ordered)
        raise ValueError(f"Dependency cycle detected among components: {missing}")

    return ordered


class RuntimeComponent(ABC):
    """Uniform lifecycle + introspection surface for assembled services."""

    name: str
    description: str

    @abstractmethod
    async def start(self) -> None:
        """Start or warm up the component."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the component and release background work."""

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Return the component health snapshot."""

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """Return the capability/introspection snapshot."""

    def snapshot(self) -> dict[str, Any]:
        """Return a combined serializable view."""
        return {
            "name": self.name,
            "description": self.description,
            "health": self.health(),
            "capabilities": self.capabilities(),
        }


@dataclass
class CallableComponent(RuntimeComponent):
    """Small adapter that turns callbacks into a runtime component."""

    name: str
    description: str
    start_hook: Callable[[], Any] | None = None
    stop_hook: Callable[[], Any] | None = None
    health_hook: Callable[[], dict[str, Any] | None] | None = None
    capabilities_hook: Callable[[], dict[str, Any] | None] | None = None
    default_status: str = "ok"
    depends_on: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    profiles: tuple[str, ...] = ("*",)

    async def start(self) -> None:
        await _call_maybe_async(self.start_hook)

    async def stop(self) -> None:
        await _call_maybe_async(self.stop_hook)

    def health(self) -> dict[str, Any]:
        if self.health_hook is None:
            return {"status": self.default_status}
        return _normalise_snapshot(
            self.health_hook(),
            default_status=self.default_status,
        )

    def capabilities(self) -> dict[str, Any]:
        return dict(self.capabilities_hook() or {})
