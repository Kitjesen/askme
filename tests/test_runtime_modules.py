"""Tests for runtime module concrete classes: LLMModule, MemoryModule, HealthModule, LEDModule."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.runtime.module import ModuleRegistry

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

    def test_build_passes_runtime_config_to_memory_bridge(self):
        from askme.runtime.modules.memory_module import MemoryModule

        cfg = {"memory": {"enabled": False, "backend": "vector"}}
        mod = MemoryModule()
        with patch("askme.runtime.modules.memory_module.SessionMemory"), \
             patch("askme.runtime.modules.memory_module.ConversationManager") as mock_conv, \
             patch("askme.runtime.modules.memory_module.MemoryBridge") as mock_bridge, \
             patch("askme.runtime.modules.memory_module.EpisodicMemory") as mock_epi, \
             patch("askme.runtime.modules.memory_module.MemorySystem"):
            mock_conv.return_value.history = []
            mock_epi.return_value._buffer = []
            mod.llm_client = None
            mod.build(cfg, _make_registry())

        mock_bridge.assert_called_once_with(config=cfg)

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

class TestMissionModule:
    def _make_module(self):
        from askme.runtime.modules.mission_module import MissionModule

        mod = MissionModule()
        mod.build({}, _make_registry())
        return mod

    def test_build_creates_service(self):
        mod = self._make_module()
        assert mod.mission_service is mod.service

    def test_health_returns_ok(self):
        mod = self._make_module()
        h = mod.health()
        assert h["status"] == "ok"
        assert h["submit_enabled"] is False

    def test_capabilities_expose_http_paths(self):
        mod = self._make_module()
        capabilities = mod.capabilities()
        assert capabilities["dry_run_default"] is True
        assert "POST /api/missions/draft" in capabilities["http_paths"]


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

    def test_build_wires_runtime_http_providers(self):
        from askme.runtime.modules.health_module import HealthModule

        class FakeTextModule:
            name = "text"

            def __init__(self):
                self.text_loop = MagicMock()
                self.text_loop.process_turn = AsyncMock(return_value="reply")

            def health(self):
                return {"status": "ok"}

            def capabilities(self):
                return {"mode": "text"}

        class FakeMemoryModule:
            name = "memory"
            conversation = MagicMock(
                history=[{"role": "user", "content": "hello"}],
            )

            def health(self):
                return {"status": "ok"}

            def capabilities(self):
                return {}

        class FakeSkillManager:
            def get_contracts(self):
                return [MagicMock(source="code"), MagicMock(source="legacy")]

            def openapi_document(self):
                return {
                    "info": {"title": "askme", "version": "1.0"},
                    "paths": {"/skills/test": {}},
                }

            def get_all(self):
                return {"test": object()}

            def get_enabled(self):
                return {"test": object()}

            def get_contract_catalog(self):
                return [{"name": "test"}]

        class FakeSkillModule:
            name = "skill"
            skill_manager = FakeSkillManager()

            def health(self):
                return {"status": "ok"}

            def capabilities(self):
                return {"skills": True}

        class FakeMissionModule:
            name = "mission"
            mission_service = MagicMock()

            def health(self):
                return {"status": "ok"}

            def capabilities(self):
                return {"dry_run_default": True}

        registry = _make_registry()
        text_mod = FakeTextModule()
        mission_mod = FakeMissionModule()
        registry.register(text_mod)
        registry.register(FakeMemoryModule())
        registry.register(FakeSkillModule())
        registry.register(mission_mod)

        mod = HealthModule()
        mock_server = MagicMock()
        mock_server.enabled = True
        mock_server.port = 8080
        with patch("askme.health_server.AskmeHealthServer", return_value=mock_server):
            mod.build({"app": {"name": "askme-test", "version": "9.9"}}, registry)

        mock_server.set_chat_handler.assert_called_once_with(
            text_mod.text_loop.process_turn,
        )
        mock_server.set_capabilities_provider.assert_called_once()
        mock_server.set_conversation_provider.assert_called_once()
        mock_server.set_mission_handler.assert_called_once_with(
            mission_mod.mission_service,
        )

        capabilities = mock_server.set_capabilities_provider.call_args.args[0]()
        assert capabilities["app"]["name"] == "askme-test"
        assert capabilities["app"]["version"] == "9.9"
        assert capabilities["app"]["voice_mode"] is False
        assert capabilities["app"]["robot_mode"] is False
        assert capabilities["profile"]["name"] == "text"
        assert capabilities["profile"]["primary_loop"] == "text"
        assert capabilities["components"]["text"]["capabilities"] == {"mode": "text"}
        assert capabilities["mission_adapter"] == {"dry_run_default": True}
        assert capabilities["skills"]["contract_count"] == 2
        assert capabilities["skills"]["code_contract_count"] == 1
        assert capabilities["openapi"]["path_count"] == 1

        conversation = mock_server.set_conversation_provider.call_args.args[0]()
        assert conversation == [{"role": "user", "content": "hello"}]

    def test_build_falls_back_to_pipeline_chat_when_text_missing(self):
        from askme.runtime.modules.health_module import HealthModule

        class FakePipelineModule:
            name = "pipeline"

            def __init__(self):
                self.brain_pipeline = MagicMock()
                self.brain_pipeline.process = AsyncMock(return_value="reply")

            def health(self):
                return {"status": "ok"}

            def capabilities(self):
                return {}

        registry = _make_registry()
        pipeline_mod = FakePipelineModule()
        registry.register(pipeline_mod)

        mod = HealthModule()
        mock_server = MagicMock()
        mock_server.enabled = True
        mock_server.port = 8080
        with patch("askme.health_server.AskmeHealthServer", return_value=mock_server):
            mod.build({}, registry)

        mock_server.set_chat_handler.assert_called_once_with(
            pipeline_mod.brain_pipeline.process,
        )

    def test_runtime_profile_infers_mcp_and_edge_robot_modes(self):
        from askme.runtime.modules.health_module import HealthModule

        class FakeModule:
            def __init__(self, name):
                self.name = name

        mod = HealthModule()

        mcp_registry = _make_registry()
        for name in ("voice", "control", "executor", "safety"):
            mcp_registry.register(FakeModule(name))

        mcp_profile = mod._runtime_profile(mcp_registry)
        assert mcp_profile.name == "mcp"
        assert mcp_profile.primary_loop == "mcp"
        assert mcp_profile.http_chat is False

        edge_registry = _make_registry()
        for name in ("voice", "text", "control", "perception", "led"):
            edge_registry.register(FakeModule(name))

        edge_profile = mod._runtime_profile(edge_registry)
        assert edge_profile.name == "edge_robot"
        assert edge_profile.primary_loop == "voice"
        assert edge_profile.http_chat is True


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
