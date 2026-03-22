"""Runtime component abstractions for askme assembly."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
