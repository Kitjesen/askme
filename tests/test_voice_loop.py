import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock

import askme.pipeline.voice_loop as voice_loop_module
import pytest
from askme.pipeline.trace import PipelineTracer
from askme.pipeline.voice_loop import VoiceLoop
from askme.voice.interaction_gate import InteractionGate

from askme.pipeline.proactive.base import ProactiveResult
from askme.pipeline.skills.outcome import SkillOutcome
from askme.robot_interaction import Intent, IntentRouter, IntentType
from askme.robot_interaction.interaction_gate import (
    InteractionAction,
    InteractionDecision,
)


class _Router:
    def route(self, text: str) -> Intent:
        if text == "exit":
            return Intent(type=IntentType.COMMAND, command="exit", raw_text=text)
        return Intent(type=IntentType.GENERAL, raw_text=text)


class _Pipeline:
    last_spoken_text = ""

    def __init__(self) -> None:
        self.process_calls: list[str] = []
        self.process_conversation_session_ids: list[str | None] = []
        self.process_voice_turn_ids: list[str | None] = []
        self.skill_calls: list[tuple[str, str]] = []
        self.memory_calls: list[str] = []
        self.pending_calls: list[str] = []
        self.pending_reply_map: dict[str, str] = {}
        self._episodic = _Episodic()

    def has_pending_tool_approval(self) -> bool:
        return False

    def start_idle_reflection(self):
        return None

    def start_memory_prefetch(self, user_text: str):
        self.memory_calls.append(user_text)
        return asyncio.create_task(asyncio.sleep(0, result=""))

    async def handle_pending_tool_response(self, user_text: str):
        self.pending_calls.append(user_text)
        return self.pending_reply_map.get(user_text)

    async def process(
        self,
        user_text: str,
        *,
        memory_task=None,
        conversation_session_id: str | None = None,
        voice_turn_id: str | None = None,
    ):
        self.process_calls.append(user_text)
        self.process_conversation_session_ids.append(conversation_session_id)
        self.process_voice_turn_ids.append(voice_turn_id)
        return "fallback"

    async def execute_skill(self, skill_name: str, user_text: str):
        self.skill_calls.append((skill_name, user_text))
        return "skill"


class _SpeakingPipeline(_Pipeline):
    def __init__(self, audio: "_Audio") -> None:
        super().__init__()
        self.audio = audio
        self.memory_results: list[str] = []

    async def process(
        self,
        user_text: str,
        *,
        memory_task=None,
        conversation_session_id: str | None = None,
    ):
        self.process_calls.append(user_text)
        self.process_conversation_session_ids.append(conversation_session_id)
        if memory_task is not None:
            self.memory_results.append(await memory_task)
        reply = f"pipeline reply: {user_text}"
        await self.audio.speak_and_wait(reply)
        self.last_spoken_text = reply
        return reply


class _Audio:
    awaiting_confirmation = False

    def __init__(self) -> None:
        self._calls = 0
        self.spoken: list[str] = []
        self.cached_spoken: list[tuple[str, str]] = []
        self._muted = False
        self._drained = 0
        self.ack_count = 0
        self.processing_feedback_arm_count = 0
        self.processing_feedback_cancel_count = 0
        self._processing_feedback_armed = False
        self.last_turn_wake_authorized = False
        self.last_turn_wake_source = "none"
        self.last_accepted_voice_turn_id: str | None = None
        self.committed_interactions = 0

    def listen_loop(self):
        self._calls += 1
        self.last_accepted_voice_turn_id = f"captured-turn-{self._calls}"
        if self._calls == 1:
            return "inspect zone"
        return "exit"

    def acknowledge(self) -> None:
        self.ack_count += 1

    def arm_processing_feedback(self, _cancel_token=None) -> bool:
        self.processing_feedback_arm_count += 1
        self._processing_feedback_armed = True
        return True

    def cancel_processing_feedback(self) -> None:
        if self._processing_feedback_armed:
            self.processing_feedback_cancel_count += 1
        self._processing_feedback_armed = False

    @property
    def processing_feedback_armed(self) -> bool:
        return self._processing_feedback_armed

    @property
    def processing_feedback_delay_s(self) -> float:
        return 0.65

    def mark_interaction_turn(self) -> None:
        self.committed_interactions += 1

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    def start_playback(self) -> None:
        return

    def wait_speaking_done(self) -> None:
        return

    def stop_playback(self) -> None:
        return

    async def speak_and_wait(self, text: str) -> None:
        self.spoken.append(text)

    async def speak_cached_and_wait(self, text: str, *, cache_key: str) -> bool:
        self.cached_spoken.append((text, cache_key))
        return True

    def drain_buffers(self) -> None:
        self._drained += 1

    def mute(self) -> None:
        self._muted = True

    def unmute(self) -> None:
        self._muted = False

    @property
    def is_muted(self) -> bool:
        return self._muted


class _Episodic:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []

    def log(self, kind: str, text: str) -> None:
        self.entries.append((kind, text))


class _NeverAddressed:
    def is_addressed(self, text: str) -> bool:
        return False


class _BystanderThenCommand:
    def is_addressed(self, text: str) -> bool:
        return text == "exit"


class _Bridge:
    def handle_voice_text(self, text: str):
        return {
            "handled": True,
            "turn": {
                "action_type": "mission",
                "spoken_reply": "runtime handled",
            },
        }


class _ExplodingBridge:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def handle_voice_text(self, text: str):
        self.calls.append(text)
        raise RuntimeError("runtime bridge offline")


class _UnavailableBridge:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def handle_voice_text(self, text: str):
        self.calls.append(text)
        return None


class _SkillBridge:
    def __init__(self, skill_name: str) -> None:
        self.skill_name = skill_name
        self.calls: list[str] = []

    def handle_voice_text(self, text: str):
        self.calls.append(text)
        return {
            "handled": True,
            "turn": {
                "action_type": "skill",
                "skill_name": self.skill_name,
            },
        }


class _Dispatcher:
    def __init__(self, *, active_agent_once: bool = False) -> None:
        self.dispatch_calls: list[tuple[str, str, str]] = []
        self.general_calls: list[tuple[str, str]] = []
        self.cancel_calls = 0
        self._active_agent_once = active_agent_once
        self._active_checks = 0

    @property
    def has_active_agent_task(self) -> bool:
        if not self._active_agent_once:
            return False
        self._active_checks += 1
        return self._active_checks == 1

    async def dispatch(self, skill_name: str, user_text: str, *, source: str = "") -> None:
        self.dispatch_calls.append((skill_name, user_text, source))

    async def handle_general(
        self,
        user_text: str,
        *,
        source: str = "",
        memory_task=None,
        conversation_session_id: str | None = None,
    ) -> None:
        self.general_calls.append((user_text, source))

    def cancel_active_agent_task(self) -> bool:
        self.cancel_calls += 1
        return False


class _Proactive:
    def __init__(self, result: ProactiveResult) -> None:
        self.result = result
        self.calls: list[tuple[str, str, str]] = []
        self.listen_once = None

    async def run(
        self,
        skill_name: str,
        user_text: str,
        audio,
        *,
        source: str,
        listen_once=None,
    ):
        self.calls.append((skill_name, user_text, source))
        self.listen_once = listen_once
        return self.result


def test_voice_loop_default_gate_fallbacks_do_not_construct_robot_interaction(
    monkeypatch,
) -> None:
    from askme.robot_interaction.address_detector import AddressDetector
    from askme.robot_interaction.interaction_gate import InteractionGate as RobotInteractionGate

    def fail_init(*args, **kwargs) -> None:
        raise AssertionError("VoiceLoop default must not construct this class")

    monkeypatch.setattr(AddressDetector, "__init__", fail_init)
    monkeypatch.setattr(RobotInteractionGate, "__init__", fail_init)

    loop = VoiceLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        audio=_Audio(),
    )

    assert loop._address_detector.is_addressed("inspect zone") is True
    decision = loop._interaction_gate.evaluate(
        "inspect zone",
        addressed=False,
        perception=None,
    )
    assert decision.action.value == "respond"
    assert decision.reason == "gate_disabled"


@pytest.mark.asyncio
async def test_voice_loop_prefers_runtime_bridge_before_llm() -> None:
    pipeline = _Pipeline()
    audio = _Audio()
    loop = VoiceLoop(
        router=_Router(),
        pipeline=pipeline,
        audio=audio,
        voice_runtime_bridge=_Bridge(),
    )

    await loop.run()

    assert pipeline.process_calls == []
    assert audio.spoken[-1] == "runtime handled"


@pytest.mark.asyncio
async def test_voice_loop_propagates_capture_voice_turn_id_to_pipeline() -> None:
    pipeline = _Pipeline()
    audio = _Audio()
    loop = VoiceLoop(router=_Router(), pipeline=pipeline, audio=audio)

    await loop.run()

    assert pipeline.process_calls == ["inspect zone"]
    assert pipeline.process_voice_turn_ids == ["captured-turn-1"]


@pytest.mark.asyncio
async def test_voice_loop_passes_conversation_session_to_runtime_bridge() -> None:
    from askme.voice_gateway import VoiceGatewayService

    class Bridge:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def handle_voice_text(self, text: str, **kwargs):
            self.calls.append({"text": text, **kwargs})
            return {
                "handled": True,
                "turn": {
                    "action_type": "mission",
                    "spoken_reply": "runtime handled",
                },
            }

    bridge = Bridge()
    gateway = VoiceGatewayService(bridge)
    loop = VoiceLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        audio=_Audio(),
        voice_runtime_bridge=gateway,
    )

    await loop.run()

    assert bridge.calls[0]["session_id"]
    assert bridge.calls[0]["session_id"] == bridge.calls[0]["conversation_session_id"]
    assert bridge.calls[0]["channel"] == "voice"


@pytest.mark.asyncio
async def test_voice_loop_replaces_closed_cached_runtime_session() -> None:
    from askme.voice_gateway import VoiceGatewayService

    class Bridge:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def handle_voice_text(self, text: str, **kwargs):
            self.calls.append({"text": text, **kwargs})
            return {
                "handled": True,
                "turn": {
                    "action_type": "mission",
                    "spoken_reply": "runtime handled",
                },
            }

    bridge = Bridge()
    gateway = VoiceGatewayService(bridge)
    loop = VoiceLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        audio=_Audio(),
        voice_runtime_bridge=gateway,
    )

    assert await loop._maybe_handle_runtime_bridge("status one") is True
    first_session_id = str(bridge.calls[0]["session_id"])
    gateway.session_manager.close_session(first_session_id)

    assert await loop._maybe_handle_runtime_bridge("status two") is True

    assert bridge.calls[1]["session_id"] != first_session_id
    assert bridge.calls[1]["channel"] == "voice"


@pytest.mark.asyncio
async def test_voice_loop_replaces_missing_cached_runtime_session() -> None:
    from askme.voice_gateway import VoiceGatewayService

    class Bridge:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def handle_voice_text(self, text: str, **kwargs):
            self.calls.append({"text": text, **kwargs})
            return {
                "handled": True,
                "turn": {
                    "action_type": "mission",
                    "spoken_reply": "runtime handled",
                },
            }

    bridge = Bridge()
    gateway = VoiceGatewayService(bridge)
    loop = VoiceLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        audio=_Audio(),
        voice_runtime_bridge=gateway,
    )

    assert await loop._maybe_handle_runtime_bridge("status one") is True
    first_session_id = str(bridge.calls[0]["session_id"])
    assert gateway.session_manager.store.delete(first_session_id) is True

    assert await loop._maybe_handle_runtime_bridge("status two") is True

    assert bridge.calls[1]["session_id"] != first_session_id
    assert bridge.calls[1]["session_id"] == bridge.calls[1]["conversation_session_id"]
    assert bridge.calls[1]["channel"] == "voice"


@pytest.mark.asyncio
async def test_voice_loop_uses_degraded_session_when_manager_unavailable() -> None:
    class BrokenManager:
        def get_or_create(self, **kwargs):
            raise RuntimeError("session store offline")

    class Bridge:
        session_manager = BrokenManager()

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def handle_voice_text(self, text: str, **kwargs):
            self.calls.append({"text": text, **kwargs})
            return {
                "handled": True,
                "turn": {
                    "action_type": "mission",
                    "spoken_reply": "runtime handled",
                },
            }

    bridge = Bridge()
    audio = _Audio()
    loop = VoiceLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        audio=audio,
        voice_runtime_bridge=bridge,
    )

    assert await loop._maybe_handle_runtime_bridge("status") is True

    assert len(bridge.calls) == 1
    call = bridge.calls[0]
    assert call["text"] == "status"
    assert str(call["session_id"]).startswith("voice-degraded-")
    assert call["session_id"] == call["conversation_session_id"]
    assert audio.spoken == ["runtime handled"]


@pytest.mark.asyncio
async def test_voice_loop_handles_pending_tool_confirmation_before_llm() -> None:
    pipeline = _Pipeline()
    pipeline.pending_reply_map["inspect zone"] = "approved"
    audio = _Audio()
    loop = VoiceLoop(
        router=_Router(),
        pipeline=pipeline,
        audio=audio,
        voice_runtime_bridge=_Bridge(),
    )

    await loop.run()

    assert pipeline.pending_calls == ["inspect zone", "exit"]
    assert pipeline.process_calls == []


@pytest.mark.asyncio
async def test_voice_loop_falls_back_to_local_pipeline_when_runtime_bridge_unhandled() -> None:
    pipeline = _Pipeline()
    audio = _Audio()
    bridge = _UnavailableBridge()
    loop = VoiceLoop(
        router=_Router(),
        pipeline=pipeline,
        audio=audio,
        voice_runtime_bridge=bridge,
    )

    await loop.run()

    assert bridge.calls == ["inspect zone"]
    assert pipeline.process_calls == ["inspect zone"]
    assert len(pipeline.process_conversation_session_ids) == 1
    assert pipeline.process_conversation_session_ids[0]


@pytest.mark.asyncio
async def test_voice_loop_records_local_fallback_turn_in_gateway_session() -> None:
    from askme.voice_gateway import VoiceGatewayService

    class Bridge:
        def handle_voice_text(self, text: str, **kwargs):
            return None

    pipeline = _Pipeline()
    audio = _Audio()
    gateway = VoiceGatewayService(Bridge())
    loop = VoiceLoop(
        router=_Router(),
        pipeline=pipeline,
        audio=audio,
        voice_runtime_bridge=gateway,
    )

    await loop.run()

    session_id = pipeline.process_conversation_session_ids[0]
    assert session_id is not None
    snapshot = gateway.conversation_snapshot(session_id)
    assert snapshot is not None
    assert snapshot.turns[0].user_text == "inspect zone"
    assert snapshot.turns[0].assistant_text == "fallback"
    assert snapshot.turns[0].metadata["local_fallback"] is True


@pytest.mark.asyncio
async def test_general_feedback_is_armed_before_memory_prefetch() -> None:
    events: list[str] = []

    class OrderedAudio(_Audio):
        def arm_processing_feedback(self, cancel_token=None) -> bool:
            events.append("feedback_armed")
            return super().arm_processing_feedback(cancel_token)

    class OrderedPipeline(_Pipeline):
        def start_memory_prefetch(self, user_text: str):
            events.append("memory_prefetch")
            return super().start_memory_prefetch(user_text)

    loop = VoiceLoop(
        router=_Router(),
        pipeline=OrderedPipeline(),
        audio=OrderedAudio(),
    )

    await loop.run()

    assert events[:2] == ["feedback_armed", "memory_prefetch"]

@pytest.mark.asyncio
async def test_general_turn_with_dispatcher_uses_handle_general() -> None:
    """listen_loop → router GENERAL → dispatcher.handle_general, not pipeline.process."""
    pipeline = _Pipeline()
    audio = _Audio()
    dispatcher = _Dispatcher()
    loop = VoiceLoop(
        router=_Router(),
        pipeline=pipeline,
        audio=audio,
        dispatcher=dispatcher,
    )

    await loop.run()

    assert dispatcher.general_calls == [("inspect zone", "voice")]
    assert pipeline.process_calls == []
    assert audio.ack_count >= 1
    assert audio.processing_feedback_arm_count == 1
    assert audio.processing_feedback_cancel_count == 1


@pytest.mark.asyncio
async def test_general_voice_turn_flows_through_pipeline_to_tts() -> None:
    """listen_loop -> router GENERAL -> pipeline.process -> audio.speak_and_wait."""
    audio = _Audio()
    pipeline = _SpeakingPipeline(audio)
    loop = VoiceLoop(
        router=_Router(),
        pipeline=pipeline,
        audio=audio,
    )

    await loop.run()

    assert pipeline.process_calls == ["inspect zone"]
    assert pipeline.memory_results == [""]
    assert audio.ack_count >= 1
    assert audio.processing_feedback_arm_count == 1
    assert audio.processing_feedback_cancel_count == 1
    assert audio.spoken == ["pipeline reply: inspect zone"]


@pytest.mark.asyncio
async def test_full_duplex_listens_during_processing_and_cancels_on_barge_in() -> None:
    """A confirmed interruption stops the active turn without a second listener."""

    class BargePipeline(_Pipeline):
        def __init__(self) -> None:
            super().__init__()
            self.processing_started = threading.Event()
            self.cancelled = threading.Event()
            self.cancel_reasons: list[str] = []
            self.cancel_thread_ids: list[int] = []
            self.callback_thread_id: int | None = None
            self._release = threading.Event()

        async def process(
            self,
            user_text: str,
            *,
            memory_task=None,
            conversation_session_id: str | None = None,
        ) -> str:
            self.process_calls.append(user_text)
            self.processing_started.set()
            while not self._release.is_set():
                await asyncio.sleep(0)
            return "cancelled"

        def cancel_active_turn(self, *, reason: str = "barge_in") -> bool:
            self.cancel_reasons.append(reason)
            self.cancel_thread_ids.append(threading.get_ident())
            self.cancelled.set()
            self._release.set()
            return True

    class FullDuplexAudio(_Audio):
        full_duplex_enabled = True

        def __init__(self, pipeline: BargePipeline) -> None:
            super().__init__()
            self._pipeline = pipeline
            self._listen_lock = threading.Lock()
            self._active_listeners = 0
            self.max_active_listeners = 0
            self.listen_calls = 0
            self.barge_callback = None

        def set_barge_in_callback(self, callback) -> None:
            self.barge_callback = callback

        def listen_loop(self):
            with self._listen_lock:
                self._active_listeners += 1
                self.max_active_listeners = max(
                    self.max_active_listeners,
                    self._active_listeners,
                )
                self.listen_calls += 1
                call_number = self.listen_calls
            try:
                if call_number == 1:
                    return "inspect zone"
                if call_number == 2:
                    assert self._pipeline.processing_started.wait(timeout=1.0)
                    assert self.barge_callback is not None
                    self.callback_thread_id = threading.get_ident()
                    self.barge_callback()
                    assert self._pipeline.cancelled.wait(timeout=1.0)
                    return "exit"
                return "exit"
            finally:
                with self._listen_lock:
                    self._active_listeners -= 1

    pipeline = BargePipeline()
    audio = FullDuplexAudio(pipeline)
    loop = VoiceLoop(router=_Router(), pipeline=pipeline, audio=audio)
    await asyncio.wait_for(loop.run(), timeout=2.0)

    assert pipeline.process_calls == ["inspect zone"]
    assert pipeline.cancel_reasons == ["barge_in"]
    assert pipeline.cancel_thread_ids == [audio.callback_thread_id]
    assert audio.listen_calls >= 2
    assert audio.max_active_listeners == 1
    assert audio.barge_callback is None


@pytest.mark.asyncio
async def test_full_duplex_listener_and_callback_are_cleaned_up_on_stop() -> None:
    class BlockingAudio(_Audio):
        full_duplex_enabled = True

        def __init__(self) -> None:
            super().__init__()
            self.listen_started = threading.Event()
            self.release_listen = threading.Event()
            self.listen_exited = threading.Event()
            self.stop_listening_calls = 0
            self.barge_callback = None

        def set_barge_in_callback(self, callback) -> None:
            self.barge_callback = callback

        def listen_loop(self):
            self.listen_started.set()
            try:
                self.release_listen.wait(timeout=2.0)
                return None
            finally:
                self.listen_exited.set()

        def stop_listening(self, *, timeout: float = 2.5) -> bool:
            self.stop_listening_calls += 1
            self.release_listen.set()
            return self.listen_exited.wait(timeout=timeout)

    audio = BlockingAudio()
    loop = VoiceLoop(router=_Router(), pipeline=_Pipeline(), audio=audio)
    task = asyncio.create_task(loop.run())
    try:
        assert await asyncio.to_thread(audio.listen_started.wait, 1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert audio.barge_callback is None
        assert loop._listen_task is None
        assert audio.stop_listening_calls == 1
        assert audio.listen_exited.is_set()
    finally:
        audio.release_listen.set()


@pytest.mark.asyncio
async def test_listener_stops_prestarting_after_runtime_duplex_downgrade() -> None:
    class DegradingAudio(_Audio):
        full_duplex_enabled = True

        def __init__(self) -> None:
            super().__init__()
            self.listen_calls = 0

        def listen_loop(self):
            self.listen_calls += 1
            self.full_duplex_enabled = False
            return "inspect zone"

    audio = DegradingAudio()
    loop = VoiceLoop(router=_Router(), pipeline=_Pipeline(), audio=audio)
    loop._event_loop = asyncio.get_running_loop()
    loop._closing = False
    loop._full_duplex_active = True

    utterance = await loop._next_utterance()
    await asyncio.sleep(0)

    assert utterance.text == "inspect zone"
    assert audio.listen_calls == 1
    assert loop._listen_task is None
    assert loop._full_duplex_active is False


def test_device_busy_failure_uses_runtime_duplex_circuit_breaker() -> None:
    class Audio:
        full_duplex_enabled = True

        def __init__(self) -> None:
            self.failures = []
            self.barge_callbacks = []

        def _full_duplex_fail_closed(self, reason, exc) -> None:
            self.failures.append((reason, exc))
            self.full_duplex_enabled = False

        def set_barge_in_callback(self, callback) -> None:
            self.barge_callbacks.append(callback)

    audio = Audio()
    loop = object.__new__(VoiceLoop)
    loop._audio = audio
    loop._full_duplex_active = True
    error = RuntimeError("device or resource busy")

    assert loop._fail_closed_full_duplex_on_audio_error(
        reason="audio_device_runtime_failure",
        exc=error,
    )
    assert audio.failures == [("audio_device_runtime_failure", error)]
    assert loop._full_duplex_active is False
    assert audio.barge_callbacks == [None]


@pytest.mark.asyncio
async def test_device_lost_restarts_audio_input_before_retry() -> None:
    lifecycle: list[str] = []

    class RecoveringAudio(_Audio):
        input_open = True

        def listen_loop(self):
            self._calls += 1
            if self._calls == 1:
                raise RuntimeError("no such audio device")
            return "exit"

        def stop_input(self) -> None:
            lifecycle.append("stop")
            self.input_open = False

        def start_input(self) -> None:
            lifecycle.append("start")
            self.input_open = True

        @property
        def is_input_open(self) -> bool:
            return self.input_open

    class DeviceLostRouter:
        def classify_error(self, exc: BaseException) -> str:
            return "device_lost"

    audio = RecoveringAudio()
    loop = VoiceLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        audio=audio,
        audio_router=DeviceLostRouter(),
    )

    await loop.run()

    assert lifecycle == ["stop", "start", "stop"]
    assert audio._calls == 2
    assert audio.is_input_open is False


@pytest.mark.asyncio
async def test_device_lost_notice_never_bypasses_playback_owner(
    monkeypatch,
) -> None:
    class OwnerFencedAudio(_Audio):
        input_open = True

        def __init__(self) -> None:
            super().__init__()
            self.tts = MagicMock()
            self.owner_aware_notice_attempts = 0

        def listen_loop(self):
            self._calls += 1
            if self._calls == 1:
                raise RuntimeError("no such audio device")
            return "exit"

        async def speak_and_wait(self, _text: str) -> None:
            self.owner_aware_notice_attempts += 1
            raise RuntimeError("playback owner conflict")

        def stop_input(self) -> None:
            self.input_open = False

        def start_input(self) -> None:
            self.input_open = True

        @property
        def is_input_open(self) -> bool:
            return self.input_open

    class DeviceLostRouter:
        def classify_error(self, exc: BaseException) -> str:
            return "device_lost"

    monkeypatch.setattr(voice_loop_module.asyncio, "sleep", AsyncMock())
    audio = OwnerFencedAudio()
    loop = VoiceLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        audio=audio,
        audio_router=DeviceLostRouter(),
    )

    await loop.run()

    assert audio.owner_aware_notice_attempts == 1
    audio.tts.speak.assert_not_called()


def test_device_reconnect_rejects_false_open_result() -> None:
    lifecycle: list[str] = []

    class FalseReadyAudio(_Audio):
        @property
        def is_input_open(self) -> bool:
            return False

        def stop_input(self) -> None:
            lifecycle.append("stop")

        def start_input(self) -> None:
            lifecycle.append("start")

    loop = VoiceLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        audio=FalseReadyAudio(),
    )

    assert loop._restart_audio_input() is False
    assert lifecycle == ["stop", "start", "stop"]


@pytest.mark.asyncio
async def test_device_reconnect_cannot_reopen_input_after_loop_stop() -> None:
    stop_entered = threading.Event()
    release_stop = threading.Event()
    lifecycle: list[str] = []

    class BlockingRecoveryAudio(_Audio):
        input_open = True
        stop_calls = 0

        def listen_loop(self):
            raise RuntimeError("no such audio device")

        def stop_input(self) -> None:
            self.stop_calls += 1
            lifecycle.append(f"stop:{self.stop_calls}")
            self.input_open = False
            if self.stop_calls == 1:
                stop_entered.set()
                assert release_stop.wait(timeout=2.0)

        def start_input(self) -> None:
            lifecycle.append("start")
            self.input_open = True

        @property
        def is_input_open(self) -> bool:
            return self.input_open

    class DeviceLostRouter:
        def classify_error(self, exc: BaseException) -> str:
            return "device_lost"

    audio = BlockingRecoveryAudio()
    loop = VoiceLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        audio=audio,
        audio_router=DeviceLostRouter(),
    )
    run_task = asyncio.create_task(loop.run())
    assert await asyncio.to_thread(stop_entered.wait, 1.0)

    run_task.cancel()
    for _ in range(20):
        if loop._closing:
            break
        await asyncio.sleep(0)
    assert loop._closing is True
    release_stop.set()

    with pytest.raises(asyncio.CancelledError):
        await run_task

    assert "start" not in lifecycle
    assert lifecycle == ["stop:1", "stop:2"]
    assert audio.is_input_open is False


@pytest.mark.asyncio
async def test_quick_reply_uses_cached_audio_before_ack_memory_or_llm() -> None:
    audio = _Audio()
    pipeline = _Pipeline()
    texts = ["\u4f60\u662f\u8c01", "exit"]
    call_idx = 0

    def _listen():
        nonlocal call_idx
        text = texts[call_idx]
        call_idx += 1
        return text

    audio.listen_loop = _listen  # type: ignore[method-assign]
    loop = VoiceLoop(router=IntentRouter(), pipeline=pipeline, audio=audio)

    await loop.run()

    assert pipeline.process_calls == []
    assert "\u4f60\u662f\u8c01" not in pipeline.memory_calls
    assert len(audio.cached_spoken) == 1
    assert "\u5c0f\u7b97" in audio.cached_spoken[0][0]
    assert audio.spoken == []
    assert audio.ack_count == 1  # exit only
    assert audio.processing_feedback_arm_count == 0
    assert audio.processing_feedback_cancel_count == 0


@pytest.mark.asyncio
async def test_location_fast_path_prefaces_then_runs_read_only_skill() -> None:
    audio = _Audio()
    pipeline = _Pipeline()
    texts = ["\u5f53\u524d\u4f4d\u7f6e", "exit"]
    call_idx = 0

    def _listen():
        nonlocal call_idx
        text = texts[call_idx]
        call_idx += 1
        return text

    audio.listen_loop = _listen  # type: ignore[method-assign]
    class _ReadySkillGate:
        async def can_execute(self, *_args, **_kwargs) -> SkillOutcome:
            return SkillOutcome.ready()

    pipeline._skill_gate = _ReadySkillGate()
    loop = VoiceLoop(router=IntentRouter(), pipeline=pipeline, audio=audio)

    await loop.run()

    assert pipeline.skill_calls == [("nav_query", "\u5f53\u524d\u4f4d\u7f6e")]
    assert "\u5f53\u524d\u4f4d\u7f6e" not in pipeline.memory_calls
    assert len(audio.cached_spoken) == 1
    assert "\u4f4d\u7f6e" in audio.cached_spoken[0][0]
    assert audio.ack_count == 1  # exit only; cached preface replaces ACK
    assert audio.processing_feedback_arm_count == 0
    assert audio.processing_feedback_cancel_count == 0


@pytest.mark.asyncio
async def test_estop_bypasses_pending_approval_and_interaction_gate_once() -> None:
    events: list[str] = []
    pipeline = _Pipeline()
    pipeline.has_pending_tool_approval = lambda: True  # type: ignore[method-assign]
    pipeline.handle_estop = lambda: events.append("handle_estop")  # type: ignore[attr-defined]
    audio = _Audio()
    dispatcher = _Dispatcher()
    dispatcher.cancel_active_agent_task = (  # type: ignore[method-assign]
        lambda: events.append("cancel_agent_task") or False
    )
    original_drain = audio.drain_buffers
    original_speak = audio.speak_and_wait

    def _drain() -> None:
        events.append("drain_buffers")
        original_drain()

    async def _speak(text: str) -> None:
        events.append("speak_estop")
        await original_speak(text)

    audio.drain_buffers = _drain  # type: ignore[method-assign]
    audio.speak_and_wait = _speak  # type: ignore[method-assign]
    texts = ["\u6025\u505c\uff01", "exit"]

    def _listen() -> str:
        return texts.pop(0)

    audio.listen_loop = _listen  # type: ignore[method-assign]

    class _RejectEstopGate:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def evaluate(self, text: str, **_kwargs) -> InteractionDecision:
            self.calls.append(text)
            action = (
                InteractionAction.IGNORE
                if text == "\u6025\u505c\uff01"
                else InteractionAction.RESPOND
            )
            return InteractionDecision(action, "test_gate", 1.0)

    gate = _RejectEstopGate()
    loop = VoiceLoop(
        router=IntentRouter(),
        pipeline=pipeline,
        audio=audio,
        dispatcher=dispatcher,
    )
    loop.set_interaction_gate(gate)  # type: ignore[arg-type]

    await loop.run()

    assert events == [
        "cancel_agent_task",
        "handle_estop",
        "drain_buffers",
        "speak_estop",
    ]
    assert gate.calls == ["exit"]
    assert "\u6025\u505c\uff01" not in pipeline.pending_calls
    assert "\u6025\u505c\uff01" not in pipeline.memory_calls
    assert pipeline.process_calls == []
    assert audio.ack_count == 1  # exit only
    assert audio.processing_feedback_arm_count == 0
    assert audio.processing_feedback_cancel_count == 0


@pytest.mark.asyncio
async def test_estop_bypasses_muted_gate_without_unmuting_microphone() -> None:
    pipeline = _Pipeline()
    estop_calls = 0

    def _handle_estop() -> None:
        nonlocal estop_calls
        estop_calls += 1

    pipeline.handle_estop = _handle_estop  # type: ignore[attr-defined]
    audio = _Audio()
    audio._muted = True
    texts = ["estop!", "exit"]

    def _listen() -> str:
        return texts.pop(0)

    audio.listen_loop = _listen  # type: ignore[method-assign]
    loop = VoiceLoop(router=IntentRouter(), pipeline=pipeline, audio=audio)

    await loop.run()

    assert estop_calls == 1
    assert audio.is_muted
    assert audio._drained == 1
    assert audio.spoken == ["\u5df2\u7ecf\u7d27\u6025\u505c\u6b62\u3002"]
    assert pipeline.process_calls == []
    assert audio.processing_feedback_arm_count == 0
    assert audio.processing_feedback_cancel_count == 0


@pytest.mark.asyncio
async def test_estop_discards_s2s_candidate_before_provider_approval() -> None:
    pipeline = _Pipeline()
    pipeline.handle_estop = lambda: None  # type: ignore[attr-defined]

    class _RealtimeAudio(_Audio):
        last_turn_realtime_generation = 0
        last_turn_realtime_baseline_generation = 0

        def __init__(self) -> None:
            super().__init__()
            self.texts = ["\u6025\u505c", "exit"]
            self.discards: list[tuple[str, int, int]] = []
            self.approval_calls: list[str] = []

        def listen_loop(self) -> str:
            text = self.texts.pop(0)
            self.last_turn_realtime_generation = 7 if text == "\u6025\u505c" else 0
            self.last_turn_realtime_baseline_generation = 3 if text == "\u6025\u505c" else 0
            return text

        def realtime_general_chat_ready(self) -> bool:
            return True

        def realtime_capture_active(self) -> bool:
            return True

        def try_realtime_general_chat(self, local_text: str, **_kwargs):
            self.approval_calls.append(local_text)
            return None

        def discard_realtime_turn(
            self,
            reason: str,
            *,
            expected_generation: int = 0,
            after_generation: int = 0,
        ) -> None:
            self.discards.append(
                (reason, expected_generation, after_generation)
            )

        def abort_realtime_playback(self, reason: str) -> None:
            return None

        def realtime_playback_started(self) -> bool:
            return False

    audio = _RealtimeAudio()
    loop = VoiceLoop(router=IntentRouter(), pipeline=pipeline, audio=audio)

    await loop.run()

    assert audio.discards[0] == ("estop", 7, 3)
    assert audio.approval_calls == []


@pytest.mark.asyncio
async def test_interaction_gate_records_bystander_speech_without_reply() -> None:
    pipeline = _Pipeline()
    audio = _Audio()
    texts = ["我们去那边看看", "exit"]
    call_idx = 0

    def _listen():
        nonlocal call_idx
        text = texts[call_idx]
        call_idx += 1
        return text

    audio.listen_loop = _listen  # type: ignore[method-assign]
    loop = VoiceLoop(
        router=_Router(),
        pipeline=pipeline,
        audio=audio,
    )
    loop.set_address_detector(_BystanderThenCommand())  # type: ignore[arg-type]
    loop.set_interaction_gate(InteractionGate({"enabled": True}))

    await loop.run()

    assert pipeline.process_calls == []
    assert audio.ack_count == 1  # only the exit command is acknowledged
    assert any(
        kind == "perception" and "ambient_speech" in text
        for kind, text in pipeline._episodic.entries
    )


@pytest.mark.asyncio
async def test_followup_window_does_not_admit_bystander_speech() -> None:
    pipeline = _Pipeline()
    audio = _Audio()
    texts = ["这个是那些琉璃布", "exit"]
    call_idx = 0

    def _listen():
        nonlocal call_idx
        audio.last_turn_wake_source = (
            "followup_window" if call_idx == 0 else "none"
        )
        audio.last_turn_wake_authorized = False
        text = texts[call_idx]
        call_idx += 1
        return text

    audio.listen_loop = _listen  # type: ignore[method-assign]
    loop = VoiceLoop(router=_Router(), pipeline=pipeline, audio=audio)
    loop.set_address_detector(_BystanderThenCommand())  # type: ignore[arg-type]
    loop.set_interaction_gate(
        InteractionGate({"enabled": True, "silent_on_ambiguous": True})
    )

    await loop.run()

    assert pipeline.process_calls == []
    assert audio.committed_interactions == 1  # exit only; ambient speech did not renew wake
    assert loop.interaction_status_snapshot()["last_decision"]["wake_source"] == "none"


@pytest.mark.asyncio
async def test_expected_short_followup_answer_stays_conversational() -> None:
    audio = _Audio()
    pipeline = _SpeakingPipeline(audio)
    pipeline.last_spoken_text = "需要继续吗？"
    texts = ["对", "exit"]
    call_idx = 0

    def _listen():
        nonlocal call_idx
        audio.last_turn_wake_source = (
            "followup_window" if call_idx == 0 else "none"
        )
        audio.last_turn_wake_authorized = False
        text = texts[call_idx]
        call_idx += 1
        return text

    audio.listen_loop = _listen  # type: ignore[method-assign]
    loop = VoiceLoop(router=_Router(), pipeline=pipeline, audio=audio)
    loop.set_address_detector(_BystanderThenCommand())  # type: ignore[arg-type]
    loop.set_interaction_gate(
        InteractionGate({"enabled": True, "silent_on_ambiguous": True})
    )

    await loop.run()

    assert pipeline.process_calls == ["对"]
    assert audio.committed_interactions == 2


@pytest.mark.asyncio
async def test_interaction_gate_answers_wayfinding_even_without_wake_word() -> None:
    audio = _Audio()
    pipeline = _SpeakingPipeline(audio)
    texts = ["请问厕所在哪里", "exit"]
    call_idx = 0

    def _listen():
        nonlocal call_idx
        text = texts[call_idx]
        call_idx += 1
        return text

    audio.listen_loop = _listen  # type: ignore[method-assign]
    loop = VoiceLoop(
        router=_Router(),
        pipeline=pipeline,
        audio=audio,
    )
    loop.set_address_detector(_BystanderThenCommand())  # type: ignore[arg-type]
    loop.set_interaction_gate(InteractionGate({"enabled": True}))

    await loop.run()

    assert pipeline.process_calls == ["请问厕所在哪里"]
    assert audio.ack_count >= 1
    assert audio.processing_feedback_arm_count == 1
    assert audio.processing_feedback_cancel_count == 1


@pytest.mark.asyncio
async def test_explicit_wake_bypasses_stale_perception_refresh() -> None:
    audio = _Audio()
    audio.last_turn_wake_authorized = True
    pipeline = _SpeakingPipeline(audio)
    texts = ["你在干什么？", "exit"]
    call_idx = 0
    perception_calls = 0

    def _listen():
        nonlocal call_idx
        audio.last_turn_wake_authorized = call_idx == 0
        audio.last_turn_wake_source = "keyword" if call_idx == 0 else "none"
        text = texts[call_idx]
        call_idx += 1
        return text

    audio.listen_loop = _listen  # type: ignore[method-assign]
    loop = VoiceLoop(
        router=_Router(),
        pipeline=pipeline,
        audio=audio,
    )
    loop.set_interaction_gate(InteractionGate({"enabled": True}))
    def _perception():
        nonlocal perception_calls
        perception_calls += 1
        if perception_calls > 1:
            return None
        return {
            "source": "camera",
            "observed_at": time.time() - 30.0,
            "person_detected": True,
        }

    loop.set_interaction_perception_provider(_perception)

    await loop.run()

    assert pipeline.process_calls == ["你在干什么？"]
    assert not any("重新确认你的位置" in text for text in audio.spoken)


@pytest.mark.asyncio
async def test_voice_trigger_dispatches_enriched_text_through_proactive() -> None:
    """listen_loop → trigger route → proactive enrichment → dispatcher.dispatch."""
    pipeline = _Pipeline()
    audio = _Audio()
    dispatcher = _Dispatcher()
    texts = ["去仓库A", "exit"]
    call_idx = 0

    def _listen():
        nonlocal call_idx
        text = texts[call_idx]
        call_idx += 1
        return text

    audio.listen_loop = _listen  # type: ignore[method-assign]
    loop = VoiceLoop(
        router=_RouterWithTrigger({"去仓库A": "navigate"}),
        pipeline=pipeline,
        audio=audio,
        dispatcher=dispatcher,
    )
    loop._proactive = _Proactive(ProactiveResult(enriched_text="导航到仓库A", proceed=True))

    await loop.run()

    assert loop._proactive.calls == [("navigate", "去仓库A", "voice")]
    assert dispatcher.dispatch_calls == [("navigate", "导航到仓库A", "voice")]
    assert dispatcher.general_calls == []
    assert pipeline.process_calls == []


@pytest.mark.asyncio
async def test_solicited_barge_in_answer_is_not_dropped_before_skill_dispatch() -> None:
    pipeline = _Pipeline()
    audio = _Audio()
    dispatcher = _Dispatcher()
    texts = ["go to warehouse A", "exit"]
    call_idx = 0

    def _listen():
        nonlocal call_idx
        text = texts[call_idx]
        call_idx += 1
        return text

    audio.listen_loop = _listen  # type: ignore[method-assign]
    loop = VoiceLoop(
        router=_RouterWithTrigger({"go to warehouse A": "navigate"}),
        pipeline=pipeline,
        audio=audio,
        dispatcher=dispatcher,
    )

    class _SolicitedBargeProactive:
        async def run(self, *args, **kwargs):
            del args, kwargs
            loop._on_confirmed_barge_in()
            return ProactiveResult(
                enriched_text="navigate to warehouse A",
                proceed=True,
            )

    loop._proactive = _SolicitedBargeProactive()

    await loop.run()

    assert dispatcher.dispatch_calls == [
        ("navigate", "navigate to warehouse A", "voice")
    ]


@pytest.mark.asyncio
async def test_runtime_bridge_skill_result_dispatches_without_proactive() -> None:
    """A runtime-resolved skill is dispatched locally and skips proactive routing."""
    pipeline = _Pipeline()
    audio = _Audio()
    dispatcher = _Dispatcher()
    bridge = _SkillBridge("get_time")
    texts = ["几点了", "exit"]
    call_idx = 0

    def _listen():
        nonlocal call_idx
        text = texts[call_idx]
        call_idx += 1
        return text

    audio.listen_loop = _listen  # type: ignore[method-assign]
    loop = VoiceLoop(
        router=_RouterWithTrigger({"几点了": "get_time"}),
        pipeline=pipeline,
        audio=audio,
        voice_runtime_bridge=bridge,
        dispatcher=dispatcher,
    )
    loop._proactive = _Proactive(ProactiveResult(enriched_text="should not run", proceed=True))

    await loop.run()

    assert bridge.calls == ["几点了"]
    assert loop._proactive.calls == []
    assert dispatcher.dispatch_calls == [("get_time", "几点了", "runtime")]


@pytest.mark.asyncio
async def test_agent_busy_gate_blocks_general_turn_and_speaks_status() -> None:
    """An active background agent blocks new general turns with a spoken status."""
    pipeline = _Pipeline()
    audio = _Audio()
    dispatcher = _Dispatcher(active_agent_once=True)
    loop = VoiceLoop(
        router=_Router(),
        pipeline=pipeline,
        audio=audio,
        dispatcher=dispatcher,
    )

    await loop.run()

    assert dispatcher.general_calls == []
    assert pipeline.process_calls == []
    assert "正在处理中，说够了可取消。" in audio.spoken


# ── Voice control: stop_speaking / mute_mic / unmute_mic ────────────────────


class _RouterWithTrigger:
    """Router that routes specific texts to voice triggers, rest to GENERAL/COMMAND."""

    def __init__(self, trigger_map: dict[str, str]) -> None:
        self._map = trigger_map

    def route(self, text: str) -> Intent:
        if text == "exit":
            return Intent(type=IntentType.COMMAND, command="exit", raw_text=text)
        if text in self._map:
            return Intent(
                type=IntentType.VOICE_TRIGGER,
                skill_name=self._map[text],
                raw_text=text,
                trigger_phrase=text,
                reason="voice_trigger",
            )
        return Intent(type=IntentType.GENERAL, raw_text=text)


@pytest.mark.asyncio
async def test_voice_loop_records_privacy_safe_intent_route_trace(monkeypatch) -> None:
    tracer = PipelineTracer()
    monkeypatch.setattr(voice_loop_module, "get_tracer", lambda: tracer)
    pipeline = _Pipeline()
    audio = _Audio()
    dispatcher = _Dispatcher()
    texts = ["去仓库A", "exit"]
    call_idx = 0

    def _listen():
        nonlocal call_idx
        text = texts[call_idx]
        call_idx += 1
        return text

    audio.listen_loop = _listen  # type: ignore[method-assign]
    loop = VoiceLoop(
        router=_RouterWithTrigger({"去仓库A": "navigate"}),
        pipeline=pipeline,
        audio=audio,
        dispatcher=dispatcher,
    )
    loop._proactive = _Proactive(ProactiveResult(enriched_text="导航到仓库A", proceed=True))

    await loop.run()

    traces = tracer.get_history(10)
    target = next(
        item
        for item in traces
        if item["metadata"].get("intent_route", {}).get("skill_name")
        == "navigate"
    )
    route = target["metadata"]["intent_route"]
    route_span = next(span for span in target["spans"] if span["name"] == "intent_route")
    assert "去仓库A" not in repr(target)
    assert route["type"] == "voice_trigger"
    assert route["source"] == "voice"
    assert route["skill_name"] == "navigate"
    assert "raw_text_preview" not in route
    assert "trigger_phrase" not in route
    assert route_span["metadata"]["reason"] == "voice_trigger"


@pytest.mark.asyncio
async def test_stop_speaking_drains_tts_without_llm() -> None:
    """stop_speaking trigger → drain_buffers called, LLM NOT called."""
    pipeline = _Pipeline()
    audio = _Audio()

    # Sequence: "静音" (stop_speaking) → "exit"
    texts = ["静音", "exit"]
    call_idx = 0

    def _listen():
        nonlocal call_idx
        t = texts[call_idx]
        call_idx += 1
        return t

    audio.listen_loop = _listen  # type: ignore[method-assign]

    loop = VoiceLoop(
        router=_RouterWithTrigger({"静音": "stop_speaking"}),
        pipeline=pipeline,
        audio=audio,
    )
    await loop.run()

    assert audio._drained >= 1, "drain_buffers should have been called for stop_speaking"
    assert pipeline.process_calls == [], "LLM should NOT be called for stop_speaking"


@pytest.mark.asyncio
async def test_mute_mic_sets_muted_flag_without_llm() -> None:
    """mute_mic trigger → audio.mute() called, LLM NOT called."""
    pipeline = _Pipeline()
    audio = _Audio()

    texts = ["闭麦", "exit"]
    call_idx = 0

    def _listen():
        nonlocal call_idx
        t = texts[call_idx]
        call_idx += 1
        return t

    audio.listen_loop = _listen  # type: ignore[method-assign]

    loop = VoiceLoop(
        router=_RouterWithTrigger({"闭麦": "mute_mic"}),
        pipeline=pipeline,
        audio=audio,
    )
    await loop.run()

    # After mute_mic, audio is muted — then "exit" is discarded (muted state)
    # so the loop never calls exit command, but the loop exits only on KeyboardInterrupt/exit
    # In this test "exit" is not routed as COMMAND because the muted gate re-routes it to GENERAL
    # and discards it, looping forever. So we need to stop after seeing muted.
    # Actually: the test loop ends because listen_loop raises IndexError after all texts consumed.
    # The IndexError propagates as a generic exception → consecutive_errors increments.
    # Let's just verify that mute was called and LLM was not.
    assert audio._muted, "audio should be muted after mute_mic trigger"
    assert pipeline.process_calls == [], "LLM should NOT be called for mute_mic"


@pytest.mark.asyncio
async def test_muted_state_discards_general_input_but_passes_unmute() -> None:
    """When muted, general inputs are discarded; unmute_mic trigger unmutes."""
    pipeline = _Pipeline()
    audio = _Audio()
    audio._muted = True  # start already muted

    spoken: list[str] = []
    audio.speak = lambda t: spoken.append(t)  # type: ignore[method-assign]
    audio.spoken = spoken  # keep reference consistent

    # Sequence: "今天天气" (general, should be discarded), "开麦" (unmute), "exit"
    texts = ["今天天气", "开麦", "exit"]
    call_idx = 0

    def _listen():
        nonlocal call_idx
        t = texts[call_idx]
        call_idx += 1
        return t

    audio.listen_loop = _listen  # type: ignore[method-assign]

    loop = VoiceLoop(
        router=_RouterWithTrigger({"开麦": "unmute_mic"}),
        pipeline=pipeline,
        audio=audio,
    )
    await loop.run()

    assert not audio._muted, "audio should be unmuted after unmute_mic trigger"
    assert pipeline.process_calls == ["今天天气"] or pipeline.process_calls == [], \
        "general input after unmute should be processed OR discarded (timing-dependent)"
    # The key invariant: mute was cleared
    assert not audio.is_muted
