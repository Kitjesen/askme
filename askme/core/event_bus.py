"""EventBus — lightweight pub/sub for inter-module communication.

Replaces file-based communication between modules with an in-process
async event bus. Faster than file I/O, type-safe, zero-latency.

Can coexist with file-based communication (ChangeDetector still writes
JSONL for external consumers, but publishes to EventBus for internal).

Usage::

    bus = EventBus()

    # Subscribe
    bus.subscribe("person_appeared", my_handler)
    bus.subscribe("*", catch_all_handler)  # wildcard

    # Publish
    await bus.publish("person_appeared", {"class": "person", "confidence": 0.9})

    # In modules:
    @module("perception")
    class PerceptionModule(Module):
        async def start(self):
            bus = services.get("event_bus")
            bus.subscribe("detection_changed", self.on_detection)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# Type for event handlers — both sync and async supported
EventHandler = Callable[["Event"], Any]


@dataclass
class Event:
    """A typed event on the bus."""

    topic: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    source: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class EventBus:
    """Async-compatible publish/subscribe event bus.

    Handlers can be sync or async. Async handlers are awaited,
    sync handlers are called directly. Errors in handlers are
    logged but don't break the publish chain.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._event_count: int = 0

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        """Subscribe a handler to a topic. Use "*" for all topics."""
        self._handlers[topic].append(handler)
        logger.debug("[EventBus] Subscribed to '%s': %s", topic, handler.__name__)

    def unsubscribe(self, topic: str, handler: EventHandler) -> None:
        """Remove a handler from a topic."""
        handlers = self._handlers.get(topic, [])
        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, topic: str, data: dict[str, Any] | None = None,
                      source: str = "") -> int:
        """Publish an event. Returns number of handlers notified.

        Notifies topic-specific handlers + wildcard ("*") handlers.
        """
        event = Event(topic=topic, data=data or {}, source=source)
        self._event_count += 1

        notified = 0

        # Topic-specific handlers
        for handler in self._handlers.get(topic, []):
            notified += await self._call_handler(handler, event)

        # Wildcard handlers
        if topic != "*":
            for handler in self._handlers.get("*", []):
                notified += await self._call_handler(handler, event)

        return notified

    async def _call_handler(self, handler: EventHandler, event: Event) -> int:
        """Call a handler safely. Returns 1 on success, 0 on error."""
        try:
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
            return 1
        except Exception as exc:
            logger.warning(
                "[EventBus] Handler '%s' failed on '%s': %s",
                handler.__name__, event.topic, exc,
            )
            return 0

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def topic_count(self) -> int:
        return len(self._handlers)

    def clear(self) -> None:
        """Remove all subscriptions."""
        self._handlers.clear()
        self._event_count = 0
