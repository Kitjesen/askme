from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from askme.robot_interaction import AddressDetector, InteractionGate
from askme.runtime.core.module import ModuleRegistry
from askme.runtime.modules.voice_module import VoiceModule
from askme.runtime.modules.voice_stack import (
    build_runtime_voice_stack,
    runtime_voice_stack_from_module,
)
from askme.voice_gateway import VoiceGatewayService


@pytest.mark.asyncio
async def test_voice_module_uses_audio_input_lifecycle() -> None:
    started = asyncio.Event()

    async def _run() -> None:
        started.set()
        await asyncio.Event().wait()

    mod = VoiceModule()
    mod._audio = MagicMock()
    mod._voice_loop = MagicMock()
    mod._voice_loop.run = _run
    mod._task = None

    await mod.start()
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await mod.stop()

    mod._audio.start_input.assert_called_once_with()
    mod._audio.stop_input.assert_called_once_with()
    mod._audio.shutdown.assert_called_once_with()


def test_runtime_voice_stack_builds_shared_audio_router_and_gateway(monkeypatch) -> None:
    calls: dict[str, object] = {}
    audio = object()
    audio_router = object()
    metrics = object()

    def _build_audio_frontend(config, *, voice_mode, metrics):
        calls["audio_config"] = config
        calls["voice_mode"] = voice_mode
        calls["metrics"] = metrics
        return SimpleNamespace(
            audio=audio,
            audio_router=audio_router,
            asr="asr-provider",
            tts="tts-provider",
        )

    def _build_voice_runtime_bridge(config):
        calls["bridge_config"] = config
        return "runtime-bridge"

    skill_manager = MagicMock()
    skill_manager.get_voice_triggers.return_value = {"hello": "greet"}

    monkeypatch.setattr(
        "askme.runtime.modules.voice_stack.build_audio_frontend",
        _build_audio_frontend,
    )
    monkeypatch.setattr(
        "askme.runtime.modules.voice_stack.build_voice_runtime_bridge",
        _build_voice_runtime_bridge,
    )

    stack = build_runtime_voice_stack(
        {
            "runtime": {"voice_bridge": {"enabled": True}},
            "voice": {
                "address_detection": {"enabled": True, "name_window": 7.0},
                "interaction_gate": {
                    "enabled": True,
                    "min_asr_confidence": 0.72,
                    "max_interaction_distance_m": 2.5,
                },
            },
        },
        voice_mode=False,
        metrics=metrics,
        skill_manager=skill_manager,
    )

    assert calls == {
        "audio_config": {
            "runtime": {"voice_bridge": {"enabled": True}},
            "voice": {
                "address_detection": {"enabled": True, "name_window": 7.0},
                "interaction_gate": {
                    "enabled": True,
                    "min_asr_confidence": 0.72,
                    "max_interaction_distance_m": 2.5,
                },
            },
        },
        "voice_mode": False,
        "metrics": metrics,
        "bridge_config": {"enabled": True},
    }
    assert stack.audio is audio
    assert stack.audio_router is audio_router
    assert stack.asr_provider == "asr-provider"
    assert stack.tts_provider == "tts-provider"
    assert stack.voice_runtime_bridge == "runtime-bridge"
    assert isinstance(stack.voice_gateway, VoiceGatewayService)
    assert stack.voice_gateway.bridge == "runtime-bridge"
    assert isinstance(stack.address_detector, AddressDetector)
    assert stack.address_detector.enabled is True
    assert stack.address_detector._name_window == 7.0
    assert isinstance(stack.interaction_gate, InteractionGate)
    assert stack.interaction_gate.min_asr_confidence == 0.72
    assert stack.interaction_gate.max_interaction_distance_m == 2.5
    skill_manager.get_voice_triggers.assert_called_once_with()


def test_voice_module_injects_runtime_stack_gate_components(monkeypatch) -> None:
    captured: dict[str, object] = {}
    address_detector = object()
    interaction_gate = object()

    class VoiceLoop:
        def __init__(self, **kwargs):
            captured["voice_loop_kwargs"] = kwargs

        def set_address_detector(self, detector):
            captured["address_detector"] = detector

        def set_interaction_gate(self, gate):
            captured["interaction_gate"] = gate

        def set_interaction_perception_provider(self, provider):
            captured["perception_provider"] = provider

        def interaction_status_snapshot(self):
            return {}

    stack = SimpleNamespace(
        audio=MagicMock(),
        audio_router=object(),
        asr_provider="asr-provider",
        tts_provider="tts-provider",
        voice_runtime_bridge=object(),
        voice_gateway=object(),
        router=object(),
        address_detector=address_detector,
        interaction_gate=interaction_gate,
    )

    monkeypatch.setattr(
        "askme.runtime.modules.voice_module.build_runtime_voice_stack",
        MagicMock(return_value=stack),
    )
    monkeypatch.setattr("askme.pipeline.channels.voice_loop.VoiceLoop", VoiceLoop)

    mod = VoiceModule()
    mod.build({}, ModuleRegistry())

    assert mod.address_detector is address_detector
    assert mod.interaction_gate is interaction_gate
    assert captured["address_detector"] is address_detector
    assert captured["interaction_gate"] is interaction_gate
    assert captured["voice_loop_kwargs"] == {
        "router": stack.router,
        "pipeline": None,
        "audio": stack.audio,
        "voice_runtime_bridge": stack.voice_gateway,
        "dispatcher": None,
        "audio_router": stack.audio_router,
    }


def test_runtime_voice_stack_wraps_legacy_raw_bridge_with_gateway() -> None:
    raw_bridge = object()
    voice_mod = SimpleNamespace(
        audio=object(),
        audio_router=object(),
        asr_provider="asr-provider",
        tts_provider="tts-provider",
        router=object(),
        address_detector=object(),
        interaction_gate=object(),
        voice_runtime_bridge=raw_bridge,
    )

    stack = runtime_voice_stack_from_module(voice_mod)

    assert stack.voice_runtime_bridge is raw_bridge
    assert isinstance(stack.voice_gateway, VoiceGatewayService)
    assert stack.voice_gateway.bridge is raw_bridge
