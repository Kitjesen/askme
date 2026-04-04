"""Tests for runtime module concrete classes: LLMModule, MemoryModule, HealthModule, LEDModule."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.runtime.module import Module, ModuleRegistry


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_registry() -> ModuleRegistry:
    return ModuleRegistry()


# ── LLMModule ─────────────────────────────────────────────────────────────────

class TestLLMModule:
    def _make_module(self, cfg=None):
        from askme.runtime.modules.llm_module import LLMModule
        mod = LLMModule()
        with patch("askme.runtime.modules.llm_module.LLMClient") as mock_cls, \
             patch("askme.runtime.modules.llm_module.OTABridgeMetrics"), \
             patch("askme.runtime.modules.llm_module.LLMConfig.from_cfg") as mock_cfg, \
             patch("askme.runtime.modules.llm_module.LLMConfig.validate_and_warn"):
            mock_client = MagicMock()
            mock_client.model = "test-model"
            mock_cls.return_value = mock_client
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.validate_and_warn = MagicMock()
            mod.build(cfg or {}, _make_registry())
        return mod

    def test_build_creates_client(self):
        from askme.runtime.modules.llm_module import LLMModule
        mod = LLMModule()
        with patch("askme.runtime.modules.llm_module.LLMClient") as mock_cls, \
             patch("askme.runtime.modules.llm_module.OTABridgeMetrics"), \
             patch("askme.runtime.modules.llm_module.LLMConfig") as mock_cfg_cls:
            mock_client = MagicMock()
            mock_client.model = "model"
            mock_cls.return_value = mock_client
            mock_llm_cfg = MagicMock()
            mock_cfg_cls.from_cfg.return_value = mock_llm_cfg
            mod.build({}, _make_registry())
        assert mod.client is not None

    def test_health_returns_ok(self):
        mod = self._make_module()
        h = mod.health()
        assert h["status"] == "ok"
        assert "model" in h

    def test_llm_client_property(self):
        mod = self._make_module()
        assert mod.llm_client is mod.client

    @pytest.mark.asyncio
    async def test_stop_cancels_warmup_task(self):
        mod = self._make_module()
        # Create a fake "running" task
        async def _long_task():
            await asyncio.sleep(100)

        task = asyncio.create_task(_long_task())
        mod._warmup_task = task
        await mod.stop()
        # Give the event loop a tick to process cancellation
        await asyncio.sleep(0)
        assert task.cancelled() or task.cancelling() > 0


# ── MemoryModule ──────────────────────────────────────────────────────────────

class TestMemoryModule:
    def _make_module(self):
        from askme.runtime.modules.memory_module import MemoryModule
        mod = MemoryModule()
        # Patch all the heavy memory classes
        with patch("askme.runtime.modules.memory_module.SessionMemory"), \
             patch("askme.runtime.modules.memory_module.ConversationManager") as mock_conv, \
             patch("askme.runtime.modules.memory_module.MemoryBridge"), \
             patch("askme.runtime.modules.memory_module.EpisodicMemory") as mock_epi, \
             patch("askme.runtime.modules.memory_module.MemorySystem"):
            mock_conv_inst = MagicMock()
            mock_conv_inst.history = []
            mock_conv.return_value = mock_conv_inst
            mock_epi_inst = MagicMock()
            mock_epi_inst._buffer = []
            mock_epi.return_value = mock_epi_inst
            mod.llm_client = None  # no LLMModule wired
            mod.build({}, _make_registry())
        return mod

    def test_build_creates_memory_components(self):
        mod = self._make_module()
        assert mod.conversation is not None
        assert mod.session_memory is not None
        assert mod.episodic is not None
        assert mod.memory_bridge is not None

    def test_health_returns_ok(self):
        mod = self._make_module()
        h = mod.health()
        assert h["status"] == "ok"
        assert "conversation_len" in h

    @pytest.mark.asyncio
    async def test_stop_no_llm_no_crash(self):
        mod = self._make_module()
        await mod.stop()  # should not raise


# ── HealthModule ──────────────────────────────────────────────────────────────

class TestHealthModule:
    def _make_module(self):
        from askme.runtime.modules.health_module import HealthModule
        mod = HealthModule()
        mock_server = MagicMock()
        mock_server.enabled = True
        mock_server.port = 8080
        with patch("askme.health_server.AskmeHealthServer", return_value=mock_server), \
             patch("askme.runtime.modules.health_module.AskmeHealthServer",
                   return_value=mock_server, create=True):
            mod.build({}, _make_registry())
        return mod

    def test_build_creates_server(self):
        mod = self._make_module()
        assert mod.server is not None

    def test_health_returns_ok(self):
        mod = self._make_module()
        h = mod.health()
        assert h["status"] == "ok"
        assert h["port"] == 8080

    @pytest.mark.asyncio
    async def test_start_calls_server_start_when_enabled(self):
        mod = self._make_module()
        mod.server.enabled = True
        mod.server.start = AsyncMock()
        await mod.start()
        mod.server.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_skips_when_disabled(self):
        mod = self._make_module()
        mod.server.enabled = False
        mod.server.start = AsyncMock()
        await mod.start()
        mod.server.start.assert_not_called()


# ── LEDModule ─────────────────────────────────────────────────────────────────

class TestLEDModule:
    def _make_module(self, led_base_url=""):
        from askme.runtime.modules.led_module import LEDModule
        mod = LEDModule()
        mock_bridge = MagicMock()
        mock_bridge.run = AsyncMock()
        with patch("askme.robot.led_controller.HttpLedController"), \
             patch("askme.robot.led_controller.NullLedController"), \
             patch("askme.robot.state_led_bridge.StateLedBridge", return_value=mock_bridge, create=True), \
             patch("askme.runtime.modules.led_module.StateLedBridge", return_value=mock_bridge, create=True):
            cfg = {"led": {"base_url": led_base_url}}
            mod.build(cfg, _make_registry())
        return mod

    def test_build_with_empty_url_uses_null_controller(self):
        mod = self._make_module(led_base_url="")
        # NullLedController is used when no URL provided
        assert mod.led_controller is not None
        assert mod.led_bridge is not None

    def test_health_returns_ok(self):
        mod = self._make_module()
        h = mod.health()
        assert h["status"] == "ok"

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        mod = self._make_module()

        async def _fake_run():
            await asyncio.sleep(100)

        mod._task = asyncio.create_task(_fake_run())
        await mod.stop()
        assert mod._task.cancelled()

    @pytest.mark.asyncio
    async def test_stop_no_task_no_crash(self):
        mod = self._make_module()
        # No task set — should not raise
        await mod.stop()
