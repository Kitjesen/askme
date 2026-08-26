from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from askme.robot_interaction import AddressDetector, InteractionGate
from askme.runtime.core.module import ModuleRegistry
from askme.runtime.modules.voice_module import (
    VoiceModule,
    _build_mission_context_provider,
    _build_voice_task_lifecycle,
    _build_voice_task_operator_provider,
    _voice_product_readiness,
)
from askme.runtime.modules.voice_stack import (
    build_runtime_voice_stack,
    runtime_voice_stack_from_module,
)
from askme.voice_gateway import VoiceGatewayService


def test_voice_module_resolves_volcengine_tts_control_payload() -> None:
    mod = VoiceModule()
    mod._base_cfg = {
        "voice": {
            "tts": {
                "backend": "minimax",
                "volcengine_tts_api_key": "secret",
                "volcengine_tts_resource_id": "seed-tts-2.0",
                "volcengine_tts_speaker": "speaker-a",
            }
        }
    }

    resolved = mod._resolve_tts_config(
        {
            "backend": "volc",
            "model": "seed-tts-2.0-metadata",
            "voice_id": "speaker-b",
        }
    )

    assert resolved["backend"] == "volcengine"
    assert resolved["volcengine_tts_model"] == "seed-tts-2.0-metadata"
    assert resolved["volcengine_tts_resource_id"] == "seed-tts-2.0-metadata"
    assert resolved["volcengine_tts_speaker"] == "speaker-b"
    assert resolved["volcengine_tts_api_key"] == "secret"


def test_voice_task_operator_provider_requires_turn_bound_or_explicit_single_operator() -> None:
    verified = {
        "operator_id": "shared-device",
        "roles": ["operator"],
        "authenticated": True,
        "source": "speaker_verification",
        "person_id": "person-turn-a",
        "permissions": ["runtime:read", "runtime:submit", "runtime:cancel"],
    }
    lifecycle = MagicMock()
    lifecycle.default_operator_context = SimpleNamespace(person_id="static-person")
    dynamic_audio = MagicMock()
    dynamic_audio.voice_task_operator_context_for_turn.return_value = verified

    dynamic = _build_voice_task_operator_provider(
        {"runtime_handoff": {"voice_task": {"operator": {}}}},
        dynamic_audio,
        lifecycle,
    )

    assert dynamic is not None
    assert dynamic("session-a", "turn-a").person_id == "person-turn-a"

    no_dynamic_audio = SimpleNamespace()
    static_denied = _build_voice_task_operator_provider(
        {
            "runtime_handoff": {
                "voice_task": {"operator": {"source": "speaker_verification"}}
            }
        },
        no_dynamic_audio,
        lifecycle,
    )
    assert static_denied is not None
    assert static_denied("session-a", "turn-a") is None

    static_allowed = _build_voice_task_operator_provider(
        {
            "runtime_handoff": {
                "voice_task": {
                    "operator": {
                        "source": "authenticated_gateway",
                        "session_scope": "single_operator",
                    }
                }
            }
        },
        no_dynamic_audio,
        lifecycle,
    )
    assert static_allowed is not None
    assert static_allowed("session-a", "turn-a").person_id == "static-person"


@pytest.mark.asyncio
async def test_voice_module_persists_volcengine_tts_selection() -> None:
    mod = VoiceModule()
    mod._base_cfg = {
        "voice": {
            "tts": {
                "backend": "minimax",
                "volcengine_tts_api_key": "secret",
                "volcengine_tts_resource_id": "seed-tts-2.0",
                "volcengine_tts_speaker": "speaker-a",
                "volcengine_tts_model": "seed-tts-2.0",
            }
        }
    }
    mod._voice_cfg = {"tts": dict(mod._base_cfg["voice"]["tts"])}
    mod._audio = MagicMock()
    mod._audio.reconfigure_tts.return_value = {"updated": True}
    mod._control_state = {}

    await mod._switch_tts(
        {
            "backend": "volcengine",
            "model": "seed-tts-2.0-runtime",
            "voice_id": "speaker-b",
        }
    )

    assert mod._control_state["tts"] == {
        "backend": "volcengine",
        "model": "seed-tts-2.0-runtime",
        "voice_id": "speaker-b",
    }


def test_voice_module_forwards_tts_activation_callback_to_audio_frontend() -> None:
    mod = VoiceModule()
    mod._audio = MagicMock()
    callback = MagicMock()

    mod.set_tts_activation_callback(callback)

    mod._audio.set_tts_activation_callback.assert_called_once_with(callback)


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


@pytest.mark.asyncio
async def test_voice_module_retries_input_without_crashing_runtime() -> None:
    started = asyncio.Event()

    async def _run() -> None:
        started.set()
        await asyncio.Event().wait()

    mod = VoiceModule()
    mod._audio = MagicMock()
    mod._audio.start_input.side_effect = [RuntimeError("device missing"), None]
    mod._voice_loop = MagicMock()
    mod._voice_loop.run = _run
    mod._task = None
    mod._input_retry_seconds = 0.01

    await mod.start()
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await mod.stop()

    assert mod._audio.start_input.call_count == 2
    assert mod._audio.stop_input.call_count >= 2
    mod._audio.shutdown.assert_called_once_with()


@pytest.mark.asyncio
async def test_voice_module_phrase_prime_is_background_and_harvested_on_stop(
    monkeypatch,
) -> None:
    started = threading.Event()
    released = threading.Event()
    loop_started = asyncio.Event()
    observed: dict[str, object] = {}

    async def _run() -> None:
        loop_started.set()
        await asyncio.Event().wait()

    def _prime(tts_config, entries, *, stop_event):
        observed["tts_config"] = tts_config
        observed["entries"] = entries
        observed["stop_event"] = stop_event
        started.set()
        released.wait(timeout=2.0)
        return []

    monkeypatch.setattr(
        "askme.runtime.modules.voice_module.prime_phrase_cache",
        _prime,
    )

    mod = VoiceModule()
    mod._audio = MagicMock()
    mod._voice_loop = MagicMock(run=_run)
    mod._task = None
    mod._router = MagicMock()
    mod._router._policy.quick_replies = {"好的": "好的。"}
    mod._voice_cfg = {
        "feedback": {
            "spoken_wait_prompt_enabled": True,
            "text": "收到，我来看看。",
            "cache_key": "feedback-waiting",
        },
        "tts": {
            "backend": "edge",
            "phrase_cache_enabled": True,
            "phrase_prime_enabled": True,
            "phrase_prime_list": [
                "好的。",
                {"cache_key": "feedback-waiting", "text": "收到，我来看看。"},
            ],
        },
    }
    mod._phrase_prime_task = None
    mod._phrase_prime_stop = threading.Event()

    await mod.start()
    await asyncio.wait_for(loop_started.wait(), timeout=1.0)
    assert await asyncio.to_thread(started.wait, 1.0)
    assert mod._phrase_prime_task is not None
    assert not mod._phrase_prime_task.done()

    released.set()
    await mod.stop()

    assert mod._phrase_prime_task.done()
    assert observed["stop_event"] is mod._phrase_prime_stop
    assert observed["tts_config"] == mod._voice_cfg["tts"]
    assert [entry.text for entry in observed["entries"]] == ["好的。", "收到，我来看看。"]


@pytest.mark.asyncio
async def test_voice_module_phrase_prime_failure_does_not_fail_startup(monkeypatch) -> None:
    loop_started = asyncio.Event()

    async def _run() -> None:
        loop_started.set()
        await asyncio.Event().wait()

    def _prime(*_args, **_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        "askme.runtime.modules.voice_module.prime_phrase_cache",
        _prime,
    )

    mod = VoiceModule()
    mod._audio = MagicMock()
    mod._voice_loop = MagicMock(run=_run)
    mod._task = None
    mod._router = MagicMock()
    mod._router._policy.quick_replies = {"好的": "好的。"}
    mod._voice_cfg = {
        "tts": {
            "phrase_cache_enabled": True,
            "phrase_prime_enabled": True,
            "phrase_prime_list": ["好的。"],
        }
    }
    mod._phrase_prime_task = None
    mod._phrase_prime_stop = threading.Event()

    await mod.start()
    await asyncio.wait_for(loop_started.wait(), timeout=1.0)
    await mod._phrase_prime_task
    await mod.stop()

    mod._audio.start_input.assert_called_once_with()
    mod._audio.shutdown.assert_called_once_with()


@pytest.mark.asyncio
async def test_voice_module_defers_provider_prewarm_to_warm_session_manager() -> None:
    loop_started = asyncio.Event()

    async def _run() -> None:
        loop_started.set()
        await asyncio.Event().wait()

    class FakeTTS:
        def __init__(self) -> None:
            self.prewarm_calls = 0
            self.cancel_calls = 0

        def prewarm_provider_session(self) -> dict[str, object]:
            self.prewarm_calls += 1
            return {"ok": True, "status": "opened"}

        def cancel_provider_prewarm(self) -> None:
            self.cancel_calls += 1

    tts = FakeTTS()
    mod = VoiceModule()
    mod._audio = MagicMock()
    mod._audio.tts = tts
    mod._voice_loop = MagicMock(run=_run)
    mod._task = None
    mod._router = MagicMock()
    mod._router._policy.quick_replies = {}
    mod._voice_cfg = {"tts": {"phrase_prime_enabled": False}}
    mod._phrase_prime_task = None
    mod._phrase_prime_stop = threading.Event()

    await mod.start()
    await asyncio.wait_for(loop_started.wait(), timeout=1.0)
    await asyncio.sleep(0.02)
    await mod.stop()

    assert tts.prewarm_calls == 0
    assert tts.cancel_calls == 0


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

        def set_mission_context_provider(self, provider):
            captured["mission_context_provider"] = provider

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

    pipeline = MagicMock()
    mod = VoiceModule()
    mod.pipeline_in = SimpleNamespace(brain_pipeline=pipeline)
    mod.build({}, ModuleRegistry())

    assert mod.address_detector is address_detector
    assert mod.interaction_gate is interaction_gate
    assert captured["address_detector"] is address_detector
    assert captured["interaction_gate"] is interaction_gate
    assert callable(captured["mission_context_provider"])
    assert captured["voice_loop_kwargs"] == {
        "router": stack.router,
        "pipeline": pipeline,
        "audio": stack.audio,
        "voice_runtime_bridge": stack.voice_gateway,
        "dispatcher": None,
        "audio_router": stack.audio_router,
        "voice_task_lifecycle": None,
        "voice_task_operator_provider": None,
        "anonymous_encounter_idle_seconds": 25.0,
    }


def test_voice_module_wires_barge_in_to_pipeline_turn_cancel(monkeypatch) -> None:
    pipeline = MagicMock()
    audio = MagicMock()
    stack = SimpleNamespace(
        audio=audio,
        audio_router=object(),
        asr_provider="asr-provider",
        tts_provider="tts-provider",
        voice_runtime_bridge=object(),
        voice_gateway=object(),
        router=object(),
        address_detector=object(),
        interaction_gate=object(),
    )

    monkeypatch.setattr(
        "askme.runtime.modules.voice_module.build_runtime_voice_stack",
        MagicMock(return_value=stack),
    )
    monkeypatch.setattr(
        "askme.pipeline.channels.voice_loop.VoiceLoop",
        MagicMock,
    )

    mod = VoiceModule()
    mod.pipeline_in = SimpleNamespace(brain_pipeline=pipeline)
    mod.build({}, ModuleRegistry())

    audio.set_barge_in_callback.assert_called_once()
    callback = audio.set_barge_in_callback.call_args.args[0]
    callback()
    pipeline.cancel_current_turn.assert_called_once_with(owner="voice")


def test_voice_module_accepts_legacy_audio_and_pipeline_without_barge_api(
    monkeypatch,
) -> None:
    audio = object()
    pipeline = SimpleNamespace(set_audio=MagicMock())
    stack = SimpleNamespace(
        audio=audio,
        audio_router=object(),
        asr_provider="asr-provider",
        tts_provider="tts-provider",
        voice_runtime_bridge=object(),
        voice_gateway=object(),
        router=object(),
        address_detector=object(),
        interaction_gate=object(),
    )

    monkeypatch.setattr(
        "askme.runtime.modules.voice_module.build_runtime_voice_stack",
        MagicMock(return_value=stack),
    )
    monkeypatch.setattr(
        "askme.runtime.modules.voice_module.build_speech_playback",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        "askme.pipeline.channels.voice_loop.VoiceLoop",
        MagicMock,
    )

    mod = VoiceModule()
    mod.pipeline_in = SimpleNamespace(brain_pipeline=pipeline)
    mod.build({}, ModuleRegistry())

    pipeline.set_audio.assert_called_once_with(audio)


def test_mission_context_provider_maps_runtime_and_safety_state() -> None:
    active_run = SimpleNamespace(current_state="paused")
    runtime_service = SimpleNamespace(
        run_service=SimpleNamespace(active_run=lambda: active_run),
    )
    modules = {
        "runtime_handoff": SimpleNamespace(runtime_handoff_service=runtime_service),
        "safety": SimpleNamespace(health=lambda: {"estop_active": False}),
    }
    registry = SimpleNamespace(get=lambda name: modules.get(name))
    provider = _build_mission_context_provider(
        {"voice": {"interaction_gate": {"default_actor_role": "supervisor"}}},
        registry,
    )

    assert provider() == {
        "mission_mode": "paused",
        "actor_role": "supervisor",
        "source": "runtime_handoff",
        "runtime_state": "paused",
    }

    modules["safety"] = SimpleNamespace(health=lambda: {"estop_active": True})
    assert provider()["mission_mode"] == "emergency"
    assert provider()["source"] == "safety"


def test_voice_product_readiness_requires_configured_wake_word() -> None:
    snapshot = _voice_product_readiness(
        {
            "pipeline_ok": True,
            "input_ready": True,
            "output_ready": True,
            "wake_word_enabled": False,
        },
        {"enabled": False, "circuit_open": False},
        {"product_readiness": {"require_wake_word": True}},
    )

    assert snapshot["ready"] is False
    assert snapshot["blockers"] == ["wake_word_not_ready"]


def test_voice_product_readiness_exposes_kws_safety_only_degraded_mode() -> None:
    snapshot = _voice_product_readiness(
        {
            "pipeline_ok": True,
            "input_ready": True,
            "output_ready": True,
            "wake_word_enabled": False,
            "kws_unavailable_safety_only": True,
        },
        {"enabled": False, "circuit_open": False},
        {"product_readiness": {"require_wake_word": True}},
    )

    assert snapshot["ready"] is False
    assert snapshot["degraded_mode"] == "kws_unavailable_safety_only"


def test_voice_product_readiness_can_require_runtime_bridge() -> None:
    snapshot = _voice_product_readiness(
        {
            "pipeline_ok": True,
            "input_ready": True,
            "output_ready": True,
            "wake_word_enabled": True,
        },
        {"enabled": False, "circuit_open": False},
        {"product_readiness": {"require_runtime_bridge": True}},
    )

    assert snapshot["ready"] is False
    assert snapshot["blockers"] == ["runtime_bridge_not_ready"]


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


def test_voice_task_lifecycle_uses_only_explicit_trusted_operator_config() -> None:
    registry = ModuleRegistry()
    runtime_module = SimpleNamespace(
        name="runtime_handoff",
        enabled=True,
        runtime_handoff_service=SimpleNamespace(
            run_service=SimpleNamespace(durable_store_ready=True)
        ),
        external_task_supervisor=object(),
    )
    registry.register(runtime_module)

    untrusted = _build_voice_task_lifecycle(
        {"runtime_handoff": {"voice_task": {"enabled": True}}},
        registry,
    )
    assert untrusted is not None
    assert untrusted._operator_context is None

    per_turn = _build_voice_task_lifecycle(
        {
            "runtime_handoff": {
                "voice_task": {
                    "enabled": True,
                    "approval_ttl_seconds": 45,
                    "operator": {
                        "operator_id": "robot-device-7",
                        "roles": ["operator"],
                        "authenticated": True,
                        "source": "speaker_verification",
                        "person_id": "operator-person-7",
                        "permissions": ["runtime:read", "runtime:submit", "runtime:cancel"],
                    },
                }
            }
        },
        registry,
    )

    assert per_turn is not None
    assert per_turn.default_operator_context is None
    with pytest.raises(PermissionError, match="runtime:read"):
        per_turn.status_snapshot("session-without-turn")

    trusted = _build_voice_task_lifecycle(
        {
            "runtime_handoff": {
                "voice_task": {
                    "enabled": True,
                    "approval_ttl_seconds": 45,
                    "operator": {
                        "operator_id": "robot-device-7",
                        "roles": ["operator"],
                        "authenticated": True,
                        "source": "authenticated_gateway",
                        "session_scope": "single_operator",
                        "person_id": "operator-person-7",
                        "permissions": ["runtime:read", "runtime:submit", "runtime:cancel"],
                    },
                }
            }
        },
        registry,
    )

    assert trusted is not None
    assert trusted._approval_ttl_s == 45.0
    assert trusted._operator_context.operator_id == "robot-device-7"
    assert trusted._operator_context.person_id == "operator-person-7"
    assert trusted._operator_context.allows("runtime:read") is True
    assert trusted._operator_context.allows("runtime:submit") is True
    assert trusted._operator_context.allows("runtime:cancel") is True

    string_false = _build_voice_task_lifecycle(
        {
            "runtime_handoff": {
                "voice_task": {
                    "enabled": True,
                    "operator": {
                        "operator_id": "robot-device-7",
                        "roles": ["operator"],
                        "authenticated": "false",
                        "source": "device_certificate",
                        "session_scope": "single_operator",
                        "permissions": ["runtime:read", "runtime:submit"],
                    },
                }
            }
        },
        registry,
    )
    assert string_false is not None
    assert string_false._operator_context.authenticated is False
    assert string_false._operator_context.allows("runtime:submit") is False


def test_voice_task_lifecycle_rejects_non_durable_runtime_store() -> None:
    registry = ModuleRegistry()
    registry.register(
        SimpleNamespace(
            name="runtime_handoff",
            enabled=True,
            runtime_handoff_service=SimpleNamespace(
                run_service=SimpleNamespace(durable_store_ready=False)
            ),
            external_task_supervisor=object(),
        )
    )

    with pytest.raises(ValueError, match="durable TaskRun store"):
        _build_voice_task_lifecycle(
            {"runtime_handoff": {"voice_task": {"enabled": True}}},
            registry,
        )
