"""Tests for BackendRegistry + interfaces."""

from __future__ import annotations

from typing import Any

import pytest

from askme.runtime.registry import BackendRegistry
from askme.interfaces.llm import LLMBackend, llm_registry
from askme.interfaces.asr import ASRBackend, asr_registry
from askme.interfaces.tts import TTSBackend, tts_registry
from askme.interfaces.detector import DetectorBackend, detector_registry
from askme.interfaces.navigator import NavigatorBackend, navigator_registry
from askme.interfaces.bus import BusBackend, bus_registry


# ── BackendRegistry core ─────────────────────────────


class DummyInterface:
    pass


class DummyImpl(DummyInterface):
    def __init__(self, cfg):
        self.cfg = cfg


def test_register_and_create():
    reg = BackendRegistry("test", DummyInterface, default="impl_a")
    reg.register("impl_a")(DummyImpl)
    instance = reg.create({"backend": "impl_a", "key": "val"})
    assert isinstance(instance, DummyImpl)
    assert instance.cfg["key"] == "val"


def test_default_backend():
    reg = BackendRegistry("test", DummyInterface, default="impl_a")
    reg.register("impl_a")(DummyImpl)
    instance = reg.create({})  # no "backend" key → uses default
    assert isinstance(instance, DummyImpl)


def test_unknown_backend_raises():
    reg = BackendRegistry("test", DummyInterface, default="")
    with pytest.raises(KeyError, match="Unknown"):
        reg.create({"backend": "nonexistent"})


def test_no_backend_no_default_raises():
    reg = BackendRegistry("test", DummyInterface, default="")
    with pytest.raises(ValueError, match="No backend specified"):
        reg.create({})


def test_wrong_type_raises():
    reg = BackendRegistry("test", DummyInterface)

    class NotDummy:
        pass

    with pytest.raises(TypeError, match="must implement"):
        reg.register("bad")(NotDummy)


def test_available():
    reg = BackendRegistry("test", DummyInterface)
    reg.register("b")(type("B", (DummyInterface,), {"__init__": lambda s, c: None}))
    reg.register("a")(type("A", (DummyInterface,), {"__init__": lambda s, c: None}))
    assert reg.available() == ["a", "b"]  # sorted


def test_contains():
    reg = BackendRegistry("test", DummyInterface)
    reg.register("x")(type("X", (DummyInterface,), {"__init__": lambda s, c: None}))
    assert "x" in reg
    assert "y" not in reg


def test_len():
    reg = BackendRegistry("test", DummyInterface)
    assert len(reg) == 0
    reg.register("x")(type("X", (DummyInterface,), {"__init__": lambda s, c: None}))
    assert len(reg) == 1


def test_get_class():
    reg = BackendRegistry("test", DummyInterface)
    cls = type("X", (DummyInterface,), {"__init__": lambda s, c: None})
    reg.register("x")(cls)
    assert reg.get_class("x") is cls
    assert reg.get_class("missing") is None


# ── Interface registries exist ───────────────────────


def test_llm_registry_exists():
    assert llm_registry.name == "llm"
    assert llm_registry.interface is LLMBackend


def test_asr_registry_exists():
    assert asr_registry.name == "asr"
    assert asr_registry.interface is ASRBackend


def test_tts_registry_exists():
    assert tts_registry.name == "tts"
    assert tts_registry.interface is TTSBackend


def test_detector_registry_exists():
    assert detector_registry.name == "detector"
    assert detector_registry.interface is DetectorBackend


def test_navigator_registry_exists():
    assert navigator_registry.name == "navigator"
    assert navigator_registry.interface is NavigatorBackend


def test_bus_registry_exists():
    assert bus_registry.name == "bus"
    assert bus_registry.interface is BusBackend


# ── Defaults ─────────────────────────────────────────


def test_llm_default_is_minimax():
    assert llm_registry.default == "minimax"


def test_asr_default_is_sherpa():
    assert asr_registry.default == "sherpa"


def test_tts_default_is_minimax():
    assert tts_registry.default == "minimax"


def test_detector_default_is_bpu_yolo():
    assert detector_registry.default == "bpu_yolo"


def test_navigator_default_is_lingtu():
    assert navigator_registry.default == "lingtu"


def test_bus_default_is_pulse():
    assert bus_registry.default == "pulse"


# ── repr ─────────────────────────────────────────────


def test_repr():
    reg = BackendRegistry("test", DummyInterface)
    assert "test" in repr(reg)
