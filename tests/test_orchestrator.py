"""Tests for Module, Orchestrator, and EventBus."""

from __future__ import annotations

import asyncio
import pytest

from askme.core.module import Module, _MODULE_REGISTRY, module
from askme.core.orchestrator import Orchestrator
from askme.core.event_bus import EventBus, Event
from askme.core import services


@pytest.fixture(autouse=True)
def clean_state():
    _MODULE_REGISTRY.clear()
    services.clear()
    yield
    _MODULE_REGISTRY.clear()
    services.clear()


# ---------- Module ----------

class TestModule:
    def test_basic_module(self):
        class MyModule(Module):
            name = "test"

        m = MyModule()
        assert m.name == "test"
        assert not m.is_started

    def test_module_decorator(self):
        @module("vision", depends_on=["config"])
        class VisionModule(Module):
            pass

        assert VisionModule.name == "vision"
        assert VisionModule.depends_on == ["config"]
        assert "vision" in _MODULE_REGISTRY


# ---------- Orchestrator ----------

class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_start_single_module(self):
        started = []

        class SimpleModule(Module):
            name = "simple"
            async def start(self):
                started.append(self.name)

        orch = Orchestrator()
        orch.register(SimpleModule)
        await orch.start_all()

        assert started == ["simple"]
        assert orch.started_modules == ["simple"]

    @pytest.mark.asyncio
    async def test_dependency_order(self):
        order = []

        class A(Module):
            name = "a"
            depends_on = []
            async def start(self):
                order.append("a")

        class B(Module):
            name = "b"
            depends_on = ["a"]
            async def start(self):
                order.append("b")

        class C(Module):
            name = "c"
            depends_on = ["b"]
            async def start(self):
                order.append("c")

        orch = Orchestrator()
        orch.register(C)  # register out of order
        orch.register(A)
        orch.register(B)
        await orch.start_all()

        assert order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_stop_reverse_order(self):
        order = []

        class A(Module):
            name = "a"
            async def stop(self):
                order.append("a")

        class B(Module):
            name = "b"
            depends_on = ["a"]
            async def stop(self):
                order.append("b")

        orch = Orchestrator()
        orch.register(A)
        orch.register(B)
        await orch.start_all()
        await orch.stop_all()

        assert order == ["b", "a"]  # reverse

    @pytest.mark.asyncio
    async def test_circular_dependency_raises(self):
        class A(Module):
            name = "a"
            depends_on = ["b"]

        class B(Module):
            name = "b"
            depends_on = ["a"]

        orch = Orchestrator()
        orch.register(A)
        orch.register(B)

        with pytest.raises(ValueError, match="Circular dependency"):
            await orch.start_all()

    @pytest.mark.asyncio
    async def test_services_registered(self):
        class A(Module):
            name = "svc_test"

        orch = Orchestrator()
        orch.register(A)
        await orch.start_all()

        assert services.get("svc_test") is not None
        assert isinstance(services.get("svc_test"), A)

    @pytest.mark.asyncio
    async def test_get_module(self):
        class A(Module):
            name = "getter"

        orch = Orchestrator()
        orch.register(A)
        await orch.start_all()

        assert orch.get("getter") is not None
        assert orch.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_config_passed(self):
        received = {}

        class A(Module):
            name = "cfg_test"
            async def start(self):
                received.update(self.config)

        orch = Orchestrator(config={"cfg_test": {"key": "value"}})
        orch.register(A)
        await orch.start_all()

        assert received == {"key": "value"}


# ---------- EventBus ----------

class TestEventBus:
    @pytest.mark.asyncio
    async def test_basic_publish_subscribe(self):
        bus = EventBus()
        received = []

        def handler(event: Event):
            received.append(event.data)

        bus.subscribe("test", handler)
        await bus.publish("test", {"msg": "hello"})

        assert len(received) == 1
        assert received[0]["msg"] == "hello"

    @pytest.mark.asyncio
    async def test_async_handler(self):
        bus = EventBus()
        received = []

        async def handler(event: Event):
            received.append(event.topic)

        bus.subscribe("async_test", handler)
        await bus.publish("async_test")

        assert received == ["async_test"]

    @pytest.mark.asyncio
    async def test_wildcard(self):
        bus = EventBus()
        received = []

        def handler(event: Event):
            received.append(event.topic)

        bus.subscribe("*", handler)
        await bus.publish("topic_a")
        await bus.publish("topic_b")

        assert received == ["topic_a", "topic_b"]

    @pytest.mark.asyncio
    async def test_multiple_handlers(self):
        bus = EventBus()
        count = [0]

        def h1(e): count[0] += 1
        def h2(e): count[0] += 10

        bus.subscribe("multi", h1)
        bus.subscribe("multi", h2)
        notified = await bus.publish("multi")

        assert notified == 2
        assert count[0] == 11

    @pytest.mark.asyncio
    async def test_handler_error_doesnt_break(self):
        bus = EventBus()
        received = []

        def bad_handler(e):
            raise ValueError("boom")

        def good_handler(e):
            received.append("ok")

        bus.subscribe("err", bad_handler)
        bus.subscribe("err", good_handler)
        notified = await bus.publish("err")

        assert notified == 1  # good_handler succeeded
        assert received == ["ok"]

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        bus = EventBus()
        received = []

        def handler(e):
            received.append(1)

        bus.subscribe("unsub", handler)
        await bus.publish("unsub")
        bus.unsubscribe("unsub", handler)
        await bus.publish("unsub")

        assert received == [1]  # only first publish

    @pytest.mark.asyncio
    async def test_event_count(self):
        bus = EventBus()
        bus.subscribe("x", lambda e: None)
        await bus.publish("x")
        await bus.publish("x")
        assert bus.event_count == 2

    @pytest.mark.asyncio
    async def test_event_source(self):
        bus = EventBus()
        received = []

        def handler(event: Event):
            received.append(event.source)

        bus.subscribe("src", handler)
        await bus.publish("src", source="change_detector")

        assert received == ["change_detector"]
