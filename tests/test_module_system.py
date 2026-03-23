"""Tests for the declarative module composition system."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from askme.runtime.module import In, Module, ModuleRegistry, Out, Runtime, RuntimeApp, _scan_ports


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


# ── In[T] / Out[T] typed ports ──────────────────────


class DetectionData:
    """Dummy typed message for port tests."""
    pass


class EstopData:
    """Dummy typed message for port tests."""
    pass


class SensorBus(Module):
    name = "sensor_bus"
    detections: Out[DetectionData]
    estop: Out[EstopData]

    def build(self, cfg, registry):
        self.value = "real_bus"


class MockSensorBus(Module):
    name = "sensor_bus"
    detections: Out[DetectionData]
    estop: Out[EstopData]

    def build(self, cfg, registry):
        self.value = "mock_bus"


class DetectionConsumer(Module):
    name = "det_consumer"
    detections: In[DetectionData]

    def build(self, cfg, registry):
        pass


class EstopConsumer(Module):
    name = "estop_consumer"
    estop: In[EstopData]

    def build(self, cfg, registry):
        pass


class MultiConsumer(Module):
    name = "multi_consumer"
    detections: In[DetectionData]
    estop: In[EstopData]

    def build(self, cfg, registry):
        pass


class NoPortModule(Module):
    name = "no_ports"

    def build(self, cfg, registry):
        pass


# ── scan_ports ───────────────────────────────────────


def test_scan_out_ports():
    ports = _scan_ports(SensorBus)
    out_names = [p.name for p in ports if p.direction == "out"]
    assert "detections" in out_names
    assert "estop" in out_names


def test_scan_in_ports():
    ports = _scan_ports(DetectionConsumer)
    in_names = [p.name for p in ports if p.direction == "in"]
    assert "detections" in in_names


def test_scan_no_ports():
    ports = _scan_ports(NoPortModule)
    assert len(ports) == 0


def test_scan_port_types():
    ports = _scan_ports(SensorBus)
    det_port = [p for p in ports if p.name == "detections"][0]
    assert det_port.data_type is DetectionData
    assert det_port.direction == "out"


# ── auto-wiring ─────────────────────────────────────


async def test_auto_wire_single_port():
    rt = Runtime.use(SensorBus) + Runtime.use(DetectionConsumer)
    app = await rt.build()
    consumer = app.modules["det_consumer"]
    assert consumer.detections is app.modules["sensor_bus"]


async def test_auto_wire_multiple_ports():
    rt = Runtime.use(SensorBus) + Runtime.use(MultiConsumer)
    app = await rt.build()
    mc = app.modules["multi_consumer"]
    assert mc.detections is app.modules["sensor_bus"]
    assert mc.estop is app.modules["sensor_bus"]


async def test_auto_wire_unmatched_in_is_none():
    """In port with no matching Out → set to None."""
    rt = Runtime.use(DetectionConsumer)  # no SensorBus
    app = await rt.build()
    assert app.modules["det_consumer"].detections is None


async def test_auto_wire_recorded_in_wired_ports():
    rt = Runtime.use(SensorBus) + Runtime.use(DetectionConsumer)
    app = await rt.build()
    assert len(app.wired_ports) >= 1
    assert any(
        w[0] == "det_consumer" and w[1] == "detections" and w[2] == "sensor_bus"
        for w in app.wired_ports
    )


async def test_replace_preserves_wiring():
    """Replace SensorBus with MockSensorBus — wiring still works."""
    rt = Runtime.use(SensorBus) + Runtime.use(DetectionConsumer)
    rt = rt.replace(SensorBus, MockSensorBus)
    app = await rt.build()
    consumer = app.modules["det_consumer"]
    bus = app.modules["sensor_bus"]
    assert consumer.detections is bus
    assert bus.value == "mock_bus"


async def test_type_mismatch_not_wired():
    """In[X] does not match Out[Y] even if name is same."""

    class OtherType:
        pass

    class WrongTypeProducer(Module):
        name = "wrong_producer"
        detections: Out[OtherType]
        def build(self, cfg, registry): pass

    rt = Runtime.use(WrongTypeProducer) + Runtime.use(DetectionConsumer)
    app = await rt.build()
    # detections: In[DetectionData] should NOT match Out[OtherType]
    assert app.modules["det_consumer"].detections is None


async def test_ambiguous_out_raises():
    """Two modules with same Out name+type → ValueError."""

    class Bus2(Module):
        name = "bus2"
        detections: Out[DetectionData]
        def build(self, cfg, registry): pass

    rt = Runtime.use(SensorBus) + Runtime.use(Bus2) + Runtime.use(DetectionConsumer)
    with pytest.raises(ValueError, match="Ambiguous"):
        await rt.build()
