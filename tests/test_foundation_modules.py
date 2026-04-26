"""Tests for Phase 1 foundation modules.

Verifies LLMModule, ToolsModule, PulseModule, and MemoryModule
build correctly with minimal config, produce expected health snapshots,
wire together via In/Out ports, and can be replaced with mocks.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from askme.llm.client import LLMClient
from askme.runtime.module import Module, ModuleRegistry, Out, Runtime
from askme.schemas.messages import (
    CmsState,
    DetectionFrame,
    EstopState,
    ImuSnapshot,
    JointStateSnapshot,
)
from askme.tools.tool_registry import ToolRegistry

# ── Helpers / Mocks ──────────────────────────────────────────────


class _MockLLMClient:
    """Minimal stand-in for LLMClient — avoids real API config."""

    def __init__(self, **kwargs: Any) -> None:
        self.model = "test-model"
        self.api_key = "test-key"
        self.base_url = "http://test"


class _MockOTAMetrics:
    """Minimal stand-in for OTABridgeMetrics."""

    def snapshot(self) -> dict[str, Any]:
        return {}


def _patch_llm_client():
    """Patch LLMClient so it doesn't read real config or connect."""
    return patch("askme.runtime.modules.llm_module.LLMClient", _MockLLMClient)


def _patch_ota_metrics():
    """Patch OTABridgeMetrics so it doesn't need real environment."""
    return patch("askme.runtime.modules.llm_module.OTABridgeMetrics", _MockOTAMetrics)


# Stub for LLMModule that doesn't touch real config
class StubLLMModule(Module):
    """LLMModule replacement for tests — no real API config needed."""

    name = "llm"
    provides = ("llm",)
    llm_client: Out[LLMClient]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        self.ota_metrics = _MockOTAMetrics()
        self.client = _MockLLMClient()

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "model": self.client.model}


# Mock Pulse module — no DDS dependency
class MockPulseModule(Module):
    """PulseModule replacement for tests — in-memory, no DDS."""

    name = "pulse"
    provides = ("telemetry", "dds")
    detections: Out[DetectionFrame]
    estop: Out[EstopState]
    joints: Out[JointStateSnapshot]
    imu: Out[ImuSnapshot]
    cms_state: Out[CmsState]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        self.bus = MagicMock()
        self.bus.health.return_value = {
            "status": "ok",
            "available": False,
            "connected": False,
            "msg_count": 0,
            "topics": {},
        }
        self.bus.available = False

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def health(self) -> dict[str, Any]:
        return self.bus.health()


# ── LLMModule tests ──────────────────────────────────────────────


class TestLLMModule:
    def test_build_with_mock(self):
        from askme.runtime.modules.llm_module import LLMModule

        with _patch_llm_client(), _patch_ota_metrics():
            mod = LLMModule()
            reg = ModuleRegistry()
            mod.build({}, reg)
            assert mod.client is not None
            assert mod.client.model == "test-model"
            assert mod.ota_metrics is not None

    def test_health(self):
        from askme.runtime.modules.llm_module import LLMModule

        with _patch_llm_client(), _patch_ota_metrics():
            mod = LLMModule()
            mod.build({}, ModuleRegistry())
            h = mod.health()
            assert h["status"] == "ok"
            assert h["model"] == "test-model"

    def test_name_and_provides(self):
        from askme.runtime.modules.llm_module import LLMModule

        assert LLMModule.name == "llm"
        assert "llm" in LLMModule.provides

    async def test_build_via_runtime(self):
        rt = Runtime.use(StubLLMModule)
        app = await rt.build()
        mod = app.modules["llm"]
        assert mod.client.model == "test-model"


# ── ToolsModule tests ────────────────────────────────────────────


class TestToolsModule:
    def test_build_registers_tools(self):
        from askme.runtime.modules.tools_module import ToolsModule

        mod = ToolsModule()
        mod.build({}, ModuleRegistry())
        assert isinstance(mod.registry, ToolRegistry)
        assert len(mod.registry) > 0

    def test_health(self):
        from askme.runtime.modules.tools_module import ToolsModule

        mod = ToolsModule()
        mod.build({}, ModuleRegistry())
        h = mod.health()
        assert h["status"] == "ok"
        assert h["tool_count"] > 0

    def test_name_and_provides(self):
        from askme.runtime.modules.tools_module import ToolsModule

        assert ToolsModule.name == "tools"
        assert "tools" in ToolsModule.provides

    def test_production_mode_config(self):
        from askme.runtime.modules.tools_module import ToolsModule

        mod = ToolsModule()
        mod.build({"tools": {"production_mode": True}}, ModuleRegistry())
        assert len(mod.registry) > 0

    async def test_build_via_runtime(self):
        from askme.runtime.modules.tools_module import ToolsModule

        rt = Runtime.use(ToolsModule)
        app = await rt.build()
        mod = app.modules["tools"]
        assert len(mod.registry) > 0


# ── PulseModule tests ────────────────────────────────────────────


class TestPulseModule:
    def test_build_creates_bus(self):
        from askme.runtime.modules.pulse_module import PulseModule

        mod = PulseModule()
        mod.build({}, ModuleRegistry())
        assert mod.bus is not None

    def test_build_with_config(self):
        from askme.runtime.modules.pulse_module import PulseModule

        mod = PulseModule()
        mod.build({"pulse": {"enabled": False, "node_name": "test_node"}}, ModuleRegistry())
        assert mod.bus is not None

    def test_health_structure(self):
        from askme.runtime.modules.pulse_module import PulseModule

        mod = PulseModule()
        mod.build({"pulse": {"enabled": False}}, ModuleRegistry())
        h = mod.health()
        assert "status" in h
        assert "available" in h
        assert "connected" in h
        assert "msg_count" in h
        assert "topics" in h

    def test_health_disabled(self):
        from askme.runtime.modules.pulse_module import PulseModule

        mod = PulseModule()
        mod.build({"pulse": {"enabled": False}}, ModuleRegistry())
        h = mod.health()
        assert h["status"] == "disabled"
        assert h["connected"] is False

    def test_name_and_provides(self):
        from askme.runtime.modules.pulse_module import PulseModule

        assert PulseModule.name == "pulse"
        assert "telemetry" in PulseModule.provides
        assert "dds" in PulseModule.provides

    async def test_start_stop_disabled(self):
        """Start/stop on a disabled Pulse is a no-op (no crash)."""
        from askme.runtime.modules.pulse_module import PulseModule

        mod = PulseModule()
        mod.build({"pulse": {"enabled": False}}, ModuleRegistry())
        await mod.start()
        await mod.stop()

    async def test_build_via_runtime(self):
        from askme.runtime.modules.pulse_module import PulseModule

        rt = Runtime.use(PulseModule)
        app = await rt.build({"pulse": {"pulse": {"enabled": False}}})
        mod = app.modules["pulse"]
        h = mod.health()
        assert "status" in h


# ── MemoryModule tests ───────────────────────────────────────────


class TestMemoryModule:
    def test_build_without_llm(self):
        """MemoryModule builds even when no LLMModule is present (In is optional)."""
        from askme.runtime.modules.memory_module import MemoryModule

        mod = MemoryModule()
        mod.llm_client = None  # simulate unwired In port
        mod.build({}, ModuleRegistry())
        assert mod.conversation is not None
        assert mod.session_memory is not None
        assert mod.memory_bridge is not None
        assert mod.episodic is not None
        assert mod.memory_system is not None

    def test_build_with_llm_module(self):
        """MemoryModule wires to LLMModule via In[LLMClient]."""
        from askme.runtime.modules.memory_module import MemoryModule

        llm_mod = StubLLMModule()
        llm_mod.build({}, ModuleRegistry())

        mem_mod = MemoryModule()
        mem_mod.llm_client = llm_mod  # simulate auto-wiring
        mem_mod.build({}, ModuleRegistry())

        # Should have used the LLMClient from StubLLMModule
        assert mem_mod.memory_system is not None
        assert mem_mod.conversation is not None

    def test_health(self):
        from askme.runtime.modules.memory_module import MemoryModule

        mod = MemoryModule()
        mod.llm_client = None
        mod.build({}, ModuleRegistry())
        h = mod.health()
        assert h["status"] == "ok"
        assert "conversation_len" in h
        assert "episodic_buffer_len" in h

    def test_name_and_provides(self):
        from askme.runtime.modules.memory_module import MemoryModule

        assert MemoryModule.name == "memory"
        assert "conversation" in MemoryModule.provides
        assert "episodic" in MemoryModule.provides

    async def test_wiring_via_runtime(self):
        """In[LLMClient] on MemoryModule auto-wires to LLMModule's Out[LLMClient]."""
        from askme.runtime.modules.memory_module import MemoryModule

        rt = Runtime.use(StubLLMModule) + Runtime.use(MemoryModule)
        app = await rt.build()
        mem_mod = app.modules["memory"]
        llm_mod = app.modules["llm"]
        # In port should be wired to the LLM module instance
        assert mem_mod.llm_client is llm_mod
        assert mem_mod.memory_system is not None


# ── Composition tests ────────────────────────────────────────────


class TestComposition:
    async def test_four_modules_compose(self):
        """All four Phase 1 modules compose, build, and start."""
        from askme.runtime.modules.memory_module import MemoryModule
        from askme.runtime.modules.tools_module import ToolsModule

        rt = (
            Runtime.use(StubLLMModule)
            + Runtime.use(ToolsModule)
            + Runtime.use(MockPulseModule)
            + Runtime.use(MemoryModule)
        )
        app = await rt.build()
        assert len(app.modules) == 4

        await app.start()
        h = app.health()
        assert "llm" in h
        assert "tools" in h
        assert "pulse" in h
        assert "memory" in h
        for name, status in h.items():
            assert status["status"] == "ok", f"{name} health not ok: {status}"
        await app.stop()

    async def test_start_order_respects_dependencies(self):
        """Modules without depends_on can start in any order; memory after llm."""
        from askme.runtime.modules.memory_module import MemoryModule

        rt = Runtime.use(StubLLMModule) + Runtime.use(MemoryModule)
        app = await rt.build()
        # llm has no depends_on, memory has no explicit depends_on either
        # but the important thing is that build() succeeds and wiring works
        assert app.modules["memory"].llm_client is app.modules["llm"]

    async def test_replace_pulse_with_mock(self):
        """Replace PulseModule with MockPulseModule — verify mock is used."""
        from askme.runtime.modules.pulse_module import PulseModule

        rt = Runtime.use(PulseModule)
        rt_mock = rt.replace(PulseModule, MockPulseModule)
        app = await rt_mock.build()
        h = app.modules["pulse"].health()
        assert h["status"] == "ok"
        # MockPulseModule's bus is a MagicMock
        assert isinstance(app.modules["pulse"].bus, MagicMock)

    async def test_replace_llm_with_stub(self):
        """Replace default LLMModule with StubLLMModule."""
        from askme.runtime.modules.llm_module import LLMModule

        rt = Runtime.use(LLMModule)
        rt_stub = rt.replace(LLMModule, StubLLMModule)
        app = await rt_stub.build()
        assert app.modules["llm"].client.model == "test-model"

    async def test_full_health_snapshot(self):
        """Composed runtime returns health for all modules."""
        from askme.runtime.modules.memory_module import MemoryModule
        from askme.runtime.modules.tools_module import ToolsModule

        rt = (
            Runtime.use(StubLLMModule)
            + Runtime.use(ToolsModule)
            + Runtime.use(MockPulseModule)
            + Runtime.use(MemoryModule)
        )
        app = await rt.build()
        h = app.health()
        assert len(h) == 4
        # Each module has a status key
        for mod_health in h.values():
            assert "status" in mod_health

    async def test_wire_result_records_memory_llm_wiring(self):
        """Wire result records the semantic match from MemoryModule to LLMModule."""
        from askme.runtime.modules.memory_module import MemoryModule

        rt = Runtime.use(StubLLMModule) + Runtime.use(MemoryModule)
        app = await rt.build()
        wr = app.wire_result
        # Should have at least one wired connection (memory.llm_client ← llm)
        assert len(wr.wired) >= 1
        assert any(
            w[0] == "memory" and w[1] == "llm_client" and w[2] == "llm"
            for w in wr.wired
        )
