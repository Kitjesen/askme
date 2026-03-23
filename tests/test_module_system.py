"""Tests for the declarative module composition system."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from askme.runtime.module import Module, ModuleRegistry, Runtime, RuntimeApp


# ── Test modules ─────────────────────────────────────


class FakeBus(Module):
    name = "pulse"
    provides = ("detections", "estop")

    def build(self, cfg, registry):
        self.data = {"connected": True}

    async def start(self):
        self.data["started"] = True

    async def stop(self):
        self.data["started"] = False

    def health(self):
        return {"status": "ok", "connected": self.data.get("connected")}


class MockBus(Module):
    """Drop-in replacement for FakeBus — same name, different impl."""
    name = "pulse"
    provides = ("detections", "estop")

    def build(self, cfg, registry):
        self.data = {"mock": True}

    def health(self):
        return {"status": "ok", "mock": True}


class FakePerception(Module):
    name = "perception"
    depends_on = ("pulse",)
    provides = ("world_state",)

    def build(self, cfg, registry):
        self.bus = registry.pulse

    def health(self):
        return {"status": "ok"}


class FakeVoice(Module):
    name = "voice_io"
    depends_on = ("pulse",)

    def build(self, cfg, registry):
        self.bus = registry.pulse


class FakeSafety(Module):
    name = "safety"
    depends_on = ("pulse",)

    def build(self, cfg, registry):
        pass


class Standalone(Module):
    name = "standalone"

    def build(self, cfg, registry):
        self.cfg_val = cfg.get("key", "default")


# ── Runtime.use + compose ────────────────────────────


def test_use_creates_runtime():
    rt = Runtime.use(FakeBus)
    assert len(rt._module_classes) == 1


def test_add_composes():
    rt = Runtime.use(FakeBus) + Runtime.use(FakePerception)
    assert len(rt._module_classes) == 2


def test_add_deduplicates_by_name():
    rt = Runtime.use(FakeBus) + Runtime.use(MockBus)
    assert len(rt._module_classes) == 1
    assert rt._module_classes[0] is MockBus  # last wins


def test_replace():
    rt = Runtime.use(FakeBus) + Runtime.use(FakePerception)
    rt2 = rt.replace(FakeBus, MockBus)
    names = [mc.name for mc in rt2._module_classes]
    assert "pulse" in names
    # Verify it's the mock
    assert any(mc is MockBus for mc in rt2._module_classes)


def test_without():
    rt = Runtime.use(FakeBus) + Runtime.use(FakePerception) + Runtime.use(FakeVoice)
    rt2 = rt.without(FakeVoice)
    names = [mc.name for mc in rt2._module_classes]
    assert "voice_io" not in names
    assert "pulse" in names


# ── build + start + stop ─────────────────────────────


async def test_build_resolves_dependencies():
    rt = Runtime.use(FakeBus) + Runtime.use(FakePerception)
    app = await rt.build()
    assert app.start_order.index("pulse") < app.start_order.index("perception")


async def test_start_stop_lifecycle():
    rt = Runtime.use(FakeBus) + Runtime.use(FakePerception)
    app = await rt.build()
    await app.start()
    assert app.modules["pulse"].data["started"] is True
    await app.stop()
    assert app.modules["pulse"].data["started"] is False


async def test_double_start_is_noop():
    rt = Runtime.use(Standalone)
    app = await rt.build()
    await app.start()
    await app.start()  # no error
    await app.stop()


async def test_double_stop_is_noop():
    rt = Runtime.use(Standalone)
    app = await rt.build()
    await app.start()
    await app.stop()
    await app.stop()  # no error


# ── replace in practice ──────────────────────────────


async def test_replace_swaps_implementation():
    rt = Runtime.use(FakeBus) + Runtime.use(FakePerception)
    rt_mock = rt.replace(FakeBus, MockBus)
    app = await rt_mock.build()
    assert app.modules["pulse"].health()["mock"] is True
    # Perception still works — got MockBus as its dependency
    assert app.modules["perception"].bus is app.modules["pulse"]


# ── config passthrough ───────────────────────────────


async def test_config_passed_to_build():
    rt = Runtime.use(Standalone)
    app = await rt.build({"standalone": {"key": "hello"}})
    assert app.modules["standalone"].cfg_val == "hello"


# ── health ───────────────────────────────────────────


async def test_app_health():
    rt = Runtime.use(FakeBus) + Runtime.use(FakePerception)
    app = await rt.build()
    h = app.health()
    assert "pulse" in h
    assert "perception" in h
    assert h["pulse"]["status"] == "ok"


# ── attribute access ─────────────────────────────────


async def test_app_getattr():
    rt = Runtime.use(FakeBus)
    app = await rt.build()
    assert app.pulse is app.modules["pulse"]


async def test_app_getattr_missing():
    rt = Runtime.use(FakeBus)
    app = await rt.build()
    with pytest.raises(AttributeError):
        _ = app.nonexistent


# ── ModuleRegistry ───────────────────────────────────


def test_registry_contains():
    reg = ModuleRegistry()
    m = FakeBus()
    m.build({}, reg)
    reg.register(m)
    assert "pulse" in reg
    assert "missing" not in reg


def test_registry_getattr_missing():
    reg = ModuleRegistry()
    with pytest.raises(AttributeError):
        _ = reg.nonexistent


# ── auto name from class ─────────────────────────────


class MyCustomModule(Module):
    def build(self, cfg, registry):
        pass

def test_auto_name():
    assert MyCustomModule.name == "my_custom_module"


# ── cycle detection ──────────────────────────────────


class CycleA(Module):
    name = "cycle_a"
    depends_on = ("cycle_b",)
    def build(self, cfg, registry): pass

class CycleB(Module):
    name = "cycle_b"
    depends_on = ("cycle_a",)
    def build(self, cfg, registry): pass


async def test_cycle_raises():
    rt = Runtime.use(CycleA) + Runtime.use(CycleB)
    with pytest.raises(ValueError, match="Cycle"):
        await rt.build()


# ── complex composition ──────────────────────────────


async def test_four_module_diamond():
    """pulse → perception + voice, both depend on pulse."""
    rt = (
        Runtime.use(FakeBus)
        + Runtime.use(FakePerception)
        + Runtime.use(FakeVoice)
        + Runtime.use(FakeSafety)
    )
    app = await rt.build()
    # pulse must start before all others
    assert app.start_order[0] == "pulse"
    await app.start()
    await app.stop()
