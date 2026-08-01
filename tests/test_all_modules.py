"""Tests for all Phase 2-5 runtime modules and compositions.

Each module is tested with minimal config and mock dependencies.
Composition tests verify that VOICE_RUNTIME, TEXT_RUNTIME, and EDGE_ROBOT_RUNTIME
build without errors when backed by mock modules.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from askme.runtime.module import Module, ModuleRegistry, Runtime, RuntimeApp

# ── Minimal mock modules for composition testing ──────────────────────


class _MockLLMModule(Module):
    """Mock LLM module that creates a fake client."""

    name = "llm"
    provides = ("llm",)

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        self.client = MagicMock()
        self.client.model = "test-model"
        self.client.chat = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="hi", tool_calls=None))]
            )
        )
        self.llm_client = self.client
        self.ota_metrics = MagicMock()
        self.ota_metrics.snapshot.return_value = {}


class _MockToolsModule(Module):
    """Mock Tools module that creates a fake registry."""

    name = "tools"
    provides = ("tools",)

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        self.registry = MagicMock()
        self.registry.__len__ = lambda self: 0
        self.registry.get_tool_schemas.return_value = []
        self.registry.get_tool.return_value = None


class _MockPulseModule(Module):
    """Mock Pulse module."""

    name = "pulse"
    provides = ("telemetry", "dds")

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        self.bus = MagicMock()
        self.bus.available = False
        self.bus.health.return_value = {"status": "ok"}


class _MockMemoryModule(Module):
    """Mock Memory module."""

    name = "memory"
    provides = ("conversation", "episodic", "vector_memory", "session_memory")

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        self.conversation = MagicMock()
        self.conversation.history = []
        self.memory_bridge = MagicMock()
        self.session_memory = MagicMock()
        self.episodic = MagicMock()
        self.episodic._buffer = []
        self.memory_system = MagicMock()


# ── Helpers ────────────────────────────────────────────────────────────


def _base_registry() -> ModuleRegistry:
    """Build a registry with the four foundation mock modules."""
    registry = ModuleRegistry()
    cfg: dict[str, Any] = {}
    for cls in (_MockLLMModule, _MockToolsModule, _MockPulseModule, _MockMemoryModule):
        mod = cls()
        registry.register(mod)
        mod.build(cfg, registry)
    return registry


def _cfg() -> dict[str, Any]:
    """Minimal config dict for tests."""
    return {
        "brain": {"model": "test-model"},
        "voice": {},
        "pulse": {},
        "runtime": {"dog_safety": {}, "dog_control": {}},
        "tools": {},
        "led": {},
        "proactive": {"enabled": False},
        "health_server": {"enabled": False, "port": 19876},
    }


# ══════════════════════════════════════════════════════════════════════
# Phase 2: PerceptionModule
# ══════════════════════════════════════════════════════════════════════


class TestPerceptionModule:
    def test_build_minimal(self):
        from askme.runtime.modules.perception_module import PerceptionModule

        registry = _base_registry()
        mod = PerceptionModule()
        mod.build(_cfg(), registry)
        assert mod.vision_bridge is not None

    def test_health(self):
        from askme.runtime.modules.perception_module import PerceptionModule

        registry = _base_registry()
        mod = PerceptionModule()
        mod.build(_cfg(), registry)
        h = mod.health()
        assert h["status"] == "ok"
        assert "change_detector_active" in h

    def test_name_and_provides(self):
        from askme.runtime.modules.perception_module import PerceptionModule

        assert PerceptionModule.name == "perception"
        assert "vision" in PerceptionModule.provides

    def test_depends_on_pulse(self):
        from askme.runtime.modules.perception_module import PerceptionModule

        assert "pulse" in PerceptionModule.depends_on


# ══════════════════════════════════════════════════════════════════════
# Phase 2: SafetyModule
# ══════════════════════════════════════════════════════════════════════


class TestSafetyModule:
    def test_build_minimal(self):
        from askme.runtime.modules.safety_module import SafetyModule

        registry = _base_registry()
        mod = SafetyModule()
        mod.build(_cfg(), registry)
        assert mod.client is not None

    def test_health_not_configured(self):
        from askme.runtime.modules.safety_module import SafetyModule

        registry = _base_registry()
        mod = SafetyModule()
        mod.build(_cfg(), registry)
        h = mod.health()
        assert h["status"] == "ok"
        assert h["configured"] is False

    def test_name_and_provides(self):
        from askme.runtime.modules.safety_module import SafetyModule

        assert SafetyModule.name == "safety"
        assert "dog_safety" in SafetyModule.provides

    def test_depends_on_pulse(self):
        from askme.runtime.modules.safety_module import SafetyModule

        assert "pulse" in SafetyModule.depends_on


# ══════════════════════════════════════════════════════════════════════
# Phase 3: PipelineModule
# ══════════════════════════════════════════════════════════════════════


class TestPipelineModule:
    def test_build_minimal(self):
        from askme.runtime.modules.pipeline_module import PipelineModule

        registry = _base_registry()
        # Add safety module
        from askme.runtime.modules.safety_module import SafetyModule

        safety = SafetyModule()
        registry.register(safety)
        safety.build(_cfg(), registry)

        mod = PipelineModule()
        mod.build(_cfg(), registry)
        assert mod.brain_pipeline is not None

    def test_health(self):
        from askme.runtime.modules.pipeline_module import PipelineModule

        registry = _base_registry()
        mod = PipelineModule()
        mod.build(_cfg(), registry)
        h = mod.health()
        assert h["status"] == "ok"

    def test_name_and_depends(self):
        from askme.runtime.modules.pipeline_module import PipelineModule

        assert PipelineModule.name == "pipeline"
        assert "llm" in PipelineModule.depends_on
        assert "memory" in PipelineModule.depends_on


# ══════════════════════════════════════════════════════════════════════
# Phase 3: SkillModule
# ══════════════════════════════════════════════════════════════════════


class TestSkillModule:
    def test_build_minimal(self):
        from askme.runtime.modules.pipeline_module import PipelineModule
        from askme.runtime.modules.skill_module import SkillModule

        registry = _base_registry()
        pipeline = PipelineModule()
        registry.register(pipeline)
        pipeline.build(_cfg(), registry)

        mod = SkillModule()
        mod.build(_cfg(), registry)
        assert mod.skill_manager is not None
        assert mod.skill_dispatcher is not None

    def test_health(self):
        from askme.runtime.modules.pipeline_module import PipelineModule
        from askme.runtime.modules.skill_module import SkillModule

        registry = _base_registry()
        pipeline = PipelineModule()
        registry.register(pipeline)
        pipeline.build(_cfg(), registry)

        mod = SkillModule()
        mod.build(_cfg(), registry)
        h = mod.health()
        assert h["status"] == "ok"
        assert "skill_count" in h

    def test_name_and_depends(self):
        from askme.runtime.modules.skill_module import SkillModule

        assert SkillModule.name == "skill"
        assert "pipeline" in SkillModule.depends_on


# ══════════════════════════════════════════════════════════════════════
# Phase 3: ExecutorModule
# ══════════════════════════════════════════════════════════════════════


class TestExecutorModule:
    def test_build_minimal(self):
        from askme.runtime.modules.executor_module import ExecutorModule

        registry = _base_registry()
        mod = ExecutorModule()
        mod.build(_cfg(), registry)
        assert mod.shell is not None

    def test_health(self):
        from askme.runtime.modules.executor_module import ExecutorModule

        registry = _base_registry()
        mod = ExecutorModule()
        mod.build(_cfg(), registry)
        h = mod.health()
        assert h["status"] == "ok"

    def test_wires_into_pipeline(self):
        from askme.runtime.modules.executor_module import ExecutorModule
        from askme.runtime.modules.pipeline_module import PipelineModule

        registry = _base_registry()
        pipeline = PipelineModule()
        registry.register(pipeline)
        pipeline.build(_cfg(), registry)

        mod = ExecutorModule()
        mod.pipeline_in = pipeline  # simulate auto-wire
        mod.build(_cfg(), registry)
        assert pipeline.brain_pipeline._agent_shell is mod.shell

    def test_name_and_depends(self):
        from askme.runtime.modules.executor_module import ExecutorModule

        assert ExecutorModule.name == "executor"
        assert "llm" in ExecutorModule.depends_on


# ══════════════════════════════════════════════════════════════════════
# Phase 4: VoiceModule
# ══════════════════════════════════════════════════════════════════════


class TestVoiceModule:
    @patch("askme.voice.audio_agent.AudioAgent.__init__", return_value=None)
    @patch("askme.voice.audio_agent.AudioAgent.set_runtime_switch_callback")
    @patch("askme.voice.audio_agent.AudioAgent.speak", return_value=None)
    def test_build_minimal(self, mock_speak, mock_set_runtime_switch_callback, mock_init):
        from askme.runtime.modules.executor_module import ExecutorModule
        from askme.runtime.modules.pipeline_module import PipelineModule
        from askme.runtime.modules.skill_module import SkillModule
        from askme.runtime.modules.voice_module import VoiceModule

        registry = _base_registry()
        pipeline = PipelineModule()
        registry.register(pipeline)
        pipeline.build(_cfg(), registry)

        skill = SkillModule()
        registry.register(skill)
        skill.build(_cfg(), registry)

        executor = ExecutorModule()
        registry.register(executor)
        executor.build(_cfg(), registry)

        mod = VoiceModule()
        mod.pipeline_in = pipeline  # simulate Runtime auto-wiring for the required port
        mod.build(_cfg(), registry)
        assert mod.voice_loop is not None
        assert mod.router is not None
        mock_set_runtime_switch_callback.assert_called_once()

    def test_name_and_depends(self):
        from askme.runtime.modules.voice_module import VoiceModule

        assert VoiceModule.name == "voice"
        assert "pipeline" in VoiceModule.depends_on
        assert "skill" in VoiceModule.depends_on


# ══════════════════════════════════════════════════════════════════════
# Phase 4: TextModule
# ══════════════════════════════════════════════════════════════════════


class TestTextModule:
    @patch("askme.voice.audio_agent.AudioAgent.__init__", return_value=None)
    def test_build_minimal_without_voice(self, mock_init):
        from askme.runtime.modules.pipeline_module import PipelineModule
        from askme.runtime.modules.skill_module import SkillModule
        from askme.runtime.modules.text_module import TextModule

        registry = _base_registry()
        pipeline = PipelineModule()
        registry.register(pipeline)
        pipeline.build(_cfg(), registry)

        skill = SkillModule()
        registry.register(skill)
        skill.build(_cfg(), registry)

        mod = TextModule()
        mod.build(_cfg(), registry)
        assert mod.text_loop is not None
        assert mod.commands is not None

    def test_health(self):
        from askme.runtime.modules.text_module import TextModule

        # health() should work even before build
        mod = TextModule()
        h = mod.health()
        assert h["status"] == "ok"

    def test_name_and_depends(self):
        from askme.runtime.modules.text_module import TextModule

        assert TextModule.name == "text"
        assert "pipeline" in TextModule.depends_on


# ══════════════════════════════════════════════════════════════════════
# Phase 5: ControlModule
# ══════════════════════════════════════════════════════════════════════


class TestControlModule:
    def test_build_minimal(self):
        from askme.runtime.modules.control_module import ControlModule

        registry = _base_registry()
        mod = ControlModule()
        mod.build(_cfg(), registry)
        assert mod.client is not None

    def test_health_not_configured(self):
        from askme.runtime.modules.control_module import ControlModule

        registry = _base_registry()
        mod = ControlModule()
        mod.build(_cfg(), registry)
        h = mod.health()
        assert h["status"] == "ok"
        assert h["configured"] is False

    def test_name_and_depends(self):
        from askme.runtime.modules.control_module import ControlModule

        assert ControlModule.name == "control"
        assert "pulse" in ControlModule.depends_on


# ══════════════════════════════════════════════════════════════════════
# Phase 5: LEDModule
# ══════════════════════════════════════════════════════════════════════


class TestLEDModule:
    def test_build_minimal(self):
        from askme.runtime.modules.led_module import LEDModule

        registry = _base_registry()

        # Add safety module dependency
        from askme.runtime.modules.safety_module import SafetyModule

        safety = SafetyModule()
        registry.register(safety)
        safety.build(_cfg(), registry)

        mod = LEDModule()
        mod.build(_cfg(), registry)
        assert mod.led_bridge is not None
        assert mod.led_controller is not None

    def test_health(self):
        from askme.runtime.modules.led_module import LEDModule

        registry = _base_registry()
        from askme.runtime.modules.safety_module import SafetyModule

        safety = SafetyModule()
        registry.register(safety)
        safety.build(_cfg(), registry)

        mod = LEDModule()
        mod.build(_cfg(), registry)
        h = mod.health()
        assert h["status"] == "ok"

    def test_name_and_depends(self):
        from askme.runtime.modules.led_module import LEDModule

        assert LEDModule.name == "led"
        assert "safety" in LEDModule.depends_on
        assert "skill" in LEDModule.depends_on


# ══════════════════════════════════════════════════════════════════════
# Phase 5: ProactiveModule
# ══════════════════════════════════════════════════════════════════════


class TestProactiveModule:
    def test_build_minimal(self):
        from askme.runtime.modules.pipeline_module import PipelineModule
        from askme.runtime.modules.proactive_module import ProactiveModule

        registry = _base_registry()
        pipeline = PipelineModule()
        registry.register(pipeline)
        pipeline.build(_cfg(), registry)

        mod = ProactiveModule()
        mod.build(_cfg(), registry)
        assert mod.agent is not None

    def test_health_disabled(self):
        from askme.runtime.modules.pipeline_module import PipelineModule
        from askme.runtime.modules.proactive_module import ProactiveModule

        registry = _base_registry()
        pipeline = PipelineModule()
        registry.register(pipeline)
        pipeline.build(_cfg(), registry)

        mod = ProactiveModule()
        mod.build(_cfg(), registry)
        h = mod.health()
        assert h["status"] == "ok"
        assert h["enabled"] is False

    def test_name_and_depends(self):
        from askme.runtime.modules.proactive_module import ProactiveModule

        assert ProactiveModule.name == "proactive"
        assert "pipeline" in ProactiveModule.depends_on
        assert "memory" in ProactiveModule.depends_on


# ══════════════════════════════════════════════════════════════════════
# Phase 5: HealthModule
# ══════════════════════════════════════════════════════════════════════


class TestHealthModule:
    def test_build_minimal(self):
        from askme.runtime.modules.health_module import HealthModule

        registry = _base_registry()
        mod = HealthModule()
        mod.build(_cfg(), registry)
        assert mod.server is not None
        assert mod.server.enabled is False

    def test_health(self):
        from askme.runtime.modules.health_module import HealthModule

        registry = _base_registry()
        mod = HealthModule()
        mod.build(_cfg(), registry)
        h = mod.health()
        assert h["status"] == "ok"
        assert h["enabled"] is False

    def test_name_and_provides(self):
        from askme.runtime.modules.health_module import HealthModule

        assert HealthModule.name == "health"
        assert "health_http" in HealthModule.provides


# ══════════════════════════════════════════════════════════════════════
# Composition tests
# ══════════════════════════════════════════════════════════════════════


class TestCompositions:
    def test_voice_runtime_builds(self):
        """voice blueprint includes cognition and fake runtime handoff."""
        from askme.blueprints.voice import voice

        from askme.runtime.modules.warm_session_module import WarmSessionModule

        names = [mc.name for mc in voice._module_classes]
        assert "cognition" in names
        assert "runtime_handoff" in names
        assert "executor" in names
        assert "warm_sessions" in names
        assert WarmSessionModule in voice._module_classes
        assert names.index("executor") < names.index("voice")
        assert len(voice._module_classes) == 13

    def test_voice_perception_runtime_builds(self):
        """voice_perception declares its own complete module list."""
        from askme.blueprints.voice_perception import voice_perception

        from askme.runtime.modules.warm_session_module import WarmSessionModule

        names = [mc.name for mc in voice_perception._module_classes]
        assert "cognition" in names
        assert "runtime_handoff" in names
        assert "executor" in names
        assert "warm_sessions" in names
        assert WarmSessionModule in voice_perception._module_classes
        assert len(voice_perception._module_classes) == 17

    def test_text_runtime_has_no_voice(self):
        """text blueprint has no VoiceModule."""
        from askme.blueprints.text import text

        from askme.runtime.modules.warm_session_module import WarmSessionModule

        names = [mc.name for mc in text._module_classes]
        assert "voice" not in names
        assert "text" in names
        assert "mission" in names
        assert "cognition" in names
        assert "runtime_handoff" in names
        assert "warm_sessions" in names
        assert "health" in names
        assert "executor" in names
        assert WarmSessionModule in text._module_classes
        assert names.index("executor") < names.index("text")
        assert len(text._module_classes) == 12

    def test_edge_robot_runtime_builds(self):
        """edge_robot declares its own complete module list."""
        from askme.blueprints.edge_robot import edge_robot

        from askme.runtime.modules.warm_session_module import WarmSessionModule

        names = [mc.name for mc in edge_robot._module_classes]
        assert "control" in names
        assert "led" in names
        assert "mission" in names
        assert "cognition" in names
        assert "runtime_handoff" in names
        assert "warm_sessions" in names
        assert WarmSessionModule in edge_robot._module_classes
        assert len(edge_robot._module_classes) == 20

    def test_replace_on_composition(self):
        """Replacing a module in a blueprint should work."""
        from askme.blueprints.voice import voice

        class MockLLM(Module):
            name = "llm"
            provides = ("llm",)

            def build(self, cfg, registry):
                pass

        replaced = voice.replace(
            type("LLMModule", (), {"name": "llm"}),
            MockLLM,
        )
        for mc in replaced._module_classes:
            if mc.name == "llm":
                assert mc is MockLLM

    def test_without_on_composition(self):
        """Removing a module from blueprint should work."""
        from askme.blueprints.voice import voice

        from askme.runtime.modules.text_module import TextModule
        from askme.runtime.modules.warm_session_module import WarmSessionModule

        smaller = voice.without(TextModule)
        names = [mc.name for mc in smaller._module_classes]
        assert "text" not in names
        assert "cognition" in names
        assert "warm_sessions" in names
        assert WarmSessionModule in smaller._module_classes
        assert len(smaller._module_classes) == 12


# ══════════════════════════════════════════════════════════════════════
# Module __init__ exports
# ══════════════════════════════════════════════════════════════════════


class TestModuleExports:
    def test_all_18_modules_importable(self):
        """All module classes should be importable from the package."""
        from askme.runtime.modules import (
            CognitionModule,
            ControlModule,
            ExecutorModule,
            HealthModule,
            LEDModule,
            LLMModule,
            MemoryModule,
            MissionModule,
            PerceptionModule,
            PipelineModule,
            ProactiveModule,
            PulseModule,
            ReactionModule,
            RuntimeHandoffModule,
            SafetyModule,
            SkillModule,
            TelegramModule,
            TextModule,
            ToolsModule,
            VoiceModule,
            WarmSessionModule,
        )

        modules = [
            CognitionModule,
            LLMModule,
            ToolsModule,
            PulseModule,
            MemoryModule,
            MissionModule,
            PerceptionModule,
            SafetyModule,
            PipelineModule,
            SkillModule,
            ExecutorModule,
            VoiceModule,
            TextModule,
            ControlModule,
            LEDModule,
            ProactiveModule,
            ReactionModule,
            RuntimeHandoffModule,
            HealthModule,
            TelegramModule,
            WarmSessionModule,
        ]
        assert len(modules) == 21
        for mod_cls in modules:
            assert hasattr(mod_cls, "name")
            assert hasattr(mod_cls, "build")

    def test_all_in_dunder_all(self):
        """__all__ should list all module classes."""
        import askme.runtime.modules as pkg

        assert "WarmSessionModule" in pkg.__all__
        assert len(pkg.__all__) == 21


class TestToolsModule:
    async def test_stop_shuts_down_tool_registry_without_blocking(self):
        from askme.runtime.modules.tools_module import ToolsModule

        mod = ToolsModule()
        mod.registry = MagicMock()

        await mod.stop()

        mod.registry.shutdown.assert_called_once_with(
            wait=False,
            cancel_futures=True,
        )

    def test_health_includes_registry_diagnostics(self):
        from askme.runtime.modules.tools_module import ToolsModule

        mod = ToolsModule()
        mod.registry = MagicMock()
        mod.registry.diagnostics.return_value = {
            "tool_count": 3,
            "executor": {"active": True, "max_workers": 4},
            "cooldown_count": 1,
            "pending_approval": False,
            "rate_limit": {"active": {}},
            "circuit_breakers": {"open": {}},
            "background_jobs": {"stored": 0},
        }

        assert mod.health() == {
            "status": "ok",
            "tool_count": 3,
            "executor": {"active": True, "max_workers": 4},
            "cooldown_count": 1,
            "pending_approval": False,
            "rate_limit": {"active": {}},
            "circuit_breakers": {"open": {}},
            "background_jobs": {"stored": 0},
        }


# ══════════════════════════════════════════════════════════════════════
# Async build integration (lightweight — mocked foundation)
# ══════════════════════════════════════════════════════════════════════


class TestAsyncBuild:
    async def test_mock_runtime_build_and_start(self):
        """A minimal composition with mocks should build and start/stop."""
        rt = (
            Runtime.use(_MockLLMModule)
            + Runtime.use(_MockToolsModule)
            + Runtime.use(_MockPulseModule)
            + Runtime.use(_MockMemoryModule)
        )
        app = await rt.build({})
        assert isinstance(app, RuntimeApp)
        assert "llm" in app.modules
        assert "tools" in app.modules
        assert "pulse" in app.modules
        assert "memory" in app.modules
        await app.start()
        await app.stop()

    async def test_mock_runtime_health(self):
        """Health snapshot should include all modules."""
        rt = (
            Runtime.use(_MockLLMModule)
            + Runtime.use(_MockToolsModule)
            + Runtime.use(_MockPulseModule)
            + Runtime.use(_MockMemoryModule)
        )
        app = await rt.build({})
        health = app.health()
        assert "llm" in health
        assert "tools" in health
        assert "pulse" in health
        assert "memory" in health


class TestRuntimeEntrypoint:
    async def test_voice_run_app_waits_for_voice_module_task(self, monkeypatch):
        """VoiceModule.start owns VoiceLoop.run; run_app must not call it twice."""
        from askme import main

        voice_loop = MagicMock()
        voice_loop.run = AsyncMock()
        voice_mod = MagicMock()
        voice_mod.voice_loop = voice_loop
        voice_mod._task = None

        class _App:
            modules = {"voice": voice_mod}

            async def start(self):
                voice_mod._task = asyncio.create_task(voice_loop.run())

            async def stop(self):
                pass

        class _Blueprint:
            async def build(self, cfg):
                return _App()

        monkeypatch.setattr(main, "get_config", lambda: {})
        monkeypatch.setattr(
            main,
            "_select_blueprint",
            lambda *, voice_mode, robot_mode: _Blueprint(),
        )

        await main.run_app(voice_mode=True, robot_mode=False)

        voice_loop.run.assert_awaited_once()

    async def test_text_run_app_still_runs_text_loop_directly(self, monkeypatch):
        """Text mode has no module-owned loop task, so run_app still runs TextLoop."""
        from askme import main

        text_loop = MagicMock()
        text_loop.run = AsyncMock()
        text_mod = MagicMock()
        text_mod.text_loop = text_loop

        class _App:
            modules = {"text": text_mod}

            async def start(self):
                pass

            async def stop(self):
                pass

        class _Blueprint:
            async def build(self, cfg):
                return _App()

        monkeypatch.setattr(main, "get_config", lambda: {})
        monkeypatch.setattr(
            main,
            "_select_blueprint",
            lambda *, voice_mode, robot_mode: _Blueprint(),
        )

        await main.run_app(voice_mode=False, robot_mode=False)

        text_loop.run.assert_awaited_once()
