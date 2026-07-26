from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from askme.voice.core.media_contracts import VoiceMediaFrame
from askme.voice.orchestration.audio_agent import AgentState, AudioAgent
from askme.voice.orchestration.interrupt_recovery import (
    InterruptionRecoveryCoordinator,
)


class _Coordinator:
    def __init__(self) -> None:
        self.offered: list[VoiceMediaFrame] = []
        self.discarded: list[tuple[str, int, int]] = []
        self.truncated: list[tuple[str, int, int]] = []
        self.truncate_result = True
        self.truncate_handles_failure = False
        self.quarantined = False
        self.approved: list[str] = []
        self.expected_generations: list[int] = []
        self.output = None
        self.generation = 0
        self.finish_count = 0
        self.active = True
        self.recovery_result = True
        self.recovery_started = threading.Event()
        self.recovery_release: threading.Event | None = None
        self.recovery_calls = 0
        self.close_reasons: list[str] = []
        self.offer_result = True
        self.offer_error: BaseException | None = None
        self.finish_error: BaseException | None = None
        self.finish_result: bool | None = None

    def offer_audio(self, frame: VoiceMediaFrame) -> bool:
        self.offered.append(frame)
        if self.offer_error is not None:
            raise self.offer_error
        return self.offer_result

    def approve_general_chat(
        self,
        text: str,
        *,
        expected_generation: int = 0,
    ):
        self.approved.append(text)
        self.expected_generations.append(expected_generation)
        if self.output is not None:
            self.output(
                VoiceMediaFrame(
                    pcm=np.asarray([1, -2], dtype="<i2").tobytes(),
                    sample_rate=24_000,
                ),
                False,
            )
        return SimpleNamespace(initial_text="你好", wait=lambda timeout=None: "你好呀")

    def prepare_general_chat(
        self,
        text: str,
        *,
        expected_generation: int = 0,
    ):
        self.approved.append(text)
        self.expected_generations.append(expected_generation)
        return SimpleNamespace(
            generation=expected_generation,
            initial_text="你好",
            completed=True,
            wait=lambda timeout=None: "你好呀",
        )

    def release_general_chat(self, approval) -> bool:
        if approval.generation <= 0:
            return False
        if self.output is not None:
            self.output(
                VoiceMediaFrame(
                    pcm=np.asarray([1, -2], dtype="<i2").tobytes(),
                    sample_rate=24_000,
                ),
                False,
            )
        return True

    def finish_input(self) -> bool | None:
        self.finish_count += 1
        if self.finish_error is not None:
            raise self.finish_error
        return self.finish_result

    def recover_at_turn_boundary(self) -> bool:
        self.recovery_calls += 1
        self.recovery_started.set()
        if self.recovery_release is not None:
            assert self.recovery_release.wait(timeout=1.0)
        self.active = self.recovery_result
        if self.active:
            self.quarantined = False
        return self.active

    def close(self, reason: str = "shutdown") -> None:
        self.active = False
        self.close_reasons.append(reason)

    def discard_current(
        self,
        reason: str,
        *,
        expected_generation: int = 0,
        after_generation: int = 0,
    ) -> None:
        self.discarded.append(
            (reason, expected_generation, after_generation)
        )

    def truncate_current(
        self,
        reason: str,
        *,
        audio_end_ms: int,
        expected_generation: int = 0,
    ) -> bool:
        self.truncated.append((reason, audio_end_ms, expected_generation))
        if not self.truncate_result and self.truncate_handles_failure:
            self.quarantined = True
            self.discarded.append(
                (f"{reason}_truncate_failed", expected_generation, 0)
            )
        return self.truncate_result

    def status_snapshot(self) -> dict[str, object]:
        return {
            "mode": "general_chat",
            "active": self.active,
            "generation": self.generation,
            "quarantined": self.quarantined,
        }


class _TTS:
    def __init__(self) -> None:
        self.generation = 7
        self.queued: list[tuple[np.ndarray, int, bool, int | None]] = []
        self.played_ms = 0
        self.drained = 0
        self.stopped = 0
        self.started = 0
        self.joined = 0
        self._hold_token: object | None = None

    def _get_generation(self) -> int:
        return self.generation

    def begin_streaming_pcm(self) -> int:
        return self.generation

    def queue_streaming_pcm(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        final: bool = False,
        generation: int | None = None,
    ) -> bool:
        self.queued.append((samples.copy(), sample_rate, final, generation))
        return generation == self.generation

    def streaming_pcm_played_ms(self, generation: int) -> int:
        return self.played_ms if generation == self.generation else 0

    def drain_buffers(self) -> None:
        self.drained += 1
        self.generation += 1

    def stop_immediately(self) -> None:
        self.stopped += 1

    def start_playback(self) -> None:
        self.started += 1

    def stop_playback(self) -> None:
        self.joined += 1

    def pause_playback(self, *, timeout_s: float) -> object:
        del timeout_s
        self._hold_token = object()
        return self._hold_token

    def resume_playback(self, token: object) -> bool:
        if token is not self._hold_token:
            return False
        self._hold_token = None
        return True

    def abort_playback_hold(self, token: object) -> bool:
        if token is not self._hold_token:
            return False
        self._hold_token = None
        return True


def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


def _bare_agent() -> tuple[AudioAgent, _Coordinator, _TTS, list[str]]:
    agent = object.__new__(AudioAgent)
    coordinator = _Coordinator()
    tts = _TTS()
    started: list[str] = []
    agent._init_output_ownership_state()
    agent._realtime_coordinator = coordinator
    agent._realtime_mode = "general_chat"
    agent._realtime_output_lock = threading.Lock()
    agent._realtime_output_tts_generation = None
    agent._realtime_output_provider_generation = 0
    agent._realtime_output_terminated_provider_generation = 0
    agent._realtime_output_started = False
    agent._realtime_output_voice_turn_id = None
    agent._realtime_last_physical_played_ms = 0
    agent._realtime_recovery_lock = threading.Lock()
    agent._realtime_recovery_stop = threading.Event()
    agent._realtime_recovery_thread = None
    agent._realtime_capture_armed = True
    agent._realtime_faulted_coordinator = None
    agent._realtime_recovery_attempts = 0
    agent._realtime_recovery_successes = 0
    agent._realtime_recovery_failures = 0
    agent._realtime_recovery_last_error = ""
    agent._muted = False
    agent._agent_state = AgentState.IDLE
    agent._ready_chime_generation = 0
    agent._interruption_recovery = InterruptionRecoveryCoordinator(tts)
    agent._barge_in_callback = None
    agent.stop_event = threading.Event()
    agent.tts = tts
    agent._refresh_voice_metrics = lambda: {}  # type: ignore[method-assign]
    agent._begin_post_tts_input_cooldown = lambda: None  # type: ignore[method-assign]
    agent.mark_interaction_turn = lambda: None  # type: ignore[method-assign]
    agent._schedule_ready_chime = lambda: None  # type: ignore[method-assign]
    agent.cancel_processing_feedback = lambda: None  # type: ignore[method-assign]
    original_start = tts.start_playback

    def _start() -> None:
        original_start()
        started.append("started")

    tts.start_playback = _start  # type: ignore[method-assign]
    coordinator.output = agent._queue_realtime_audio
    return agent, coordinator, tts, started


def _release_two_phase(
    agent: AudioAgent,
    text: str = "你好",
    *,
    expected_generation: int,
):
    approval = agent.prepare_realtime_general_chat(
        text,
        expected_generation=expected_generation,
    )
    assert approval is not None
    assert agent.release_realtime_general_chat(
        approval,
        expected_generation=expected_generation,
    )
    return approval


def _arm_barge_in(agent: AudioAgent) -> None:
    assert agent._begin_output_interruption(peak=900, rms=32.0)
    agent._confirm_output_interruption(peak=1_100, rms=40.0)


def test_post_aec_capture_is_offered_as_pcm16_without_a_second_microphone() -> None:
    agent, coordinator, _tts, _started = _bare_agent()
    samples = np.asarray([0, 32767, -32768], dtype=np.int16)

    assert agent._offer_realtime_capture(samples, sample_rate=16_000) is True

    assert len(coordinator.offered) == 1
    frame = coordinator.offered[0]
    assert frame.sample_rate == 16_000
    assert frame.channels == 1
    assert frame.pcm == samples.astype("<i2", copy=False).tobytes()
    assert frame.metadata["capture_stage"] == "post_aec"


@pytest.mark.parametrize("failure_mode", ["false", "exception"])
def test_realtime_offer_failure_fences_the_partial_cloud_turn(
    failure_mode: str,
) -> None:
    agent, coordinator, _tts, _started = _bare_agent()
    if failure_mode == "false":
        coordinator.offer_result = False
    else:
        coordinator.offer_error = RuntimeError("socket failed")

    assert agent._offer_realtime_capture(
        np.asarray([1, -1], dtype=np.int16),
        sample_rate=16_000,
    ) is False

    assert agent._realtime_capture_armed is False
    assert coordinator.active is False
    _wait_until(lambda: coordinator.close_reasons == ["audio_offer_failed"])


def test_finish_input_failure_keeps_the_local_asr_result_and_fences_cloud() -> None:
    agent, coordinator, _tts, _started = _bare_agent()
    coordinator.finish_error = RuntimeError("finish failed")
    coordinator.generation = 2
    agent._realtime_generation_at_listen_start = 1
    agent._turn_traces = MagicMock()
    agent.audio_queue = queue.Queue()
    agent._metrics = MagicMock()
    agent._clear_input_failure = lambda: None  # type: ignore[method-assign]
    agent._refresh_voice_metrics = lambda: {}  # type: ignore[method-assign]
    agent._asr_mgr = MagicMock()

    assert agent._accept_result("本地结果") == "本地结果"

    assert agent.audio_queue.get_nowait() == "本地结果"
    assert coordinator.finish_count == 1
    _wait_until(lambda: coordinator.close_reasons == ["finish_input_failed"])
    assert agent._realtime_capture_armed is False
    assert agent.last_turn_realtime_generation == 0


def test_finish_input_false_keeps_the_local_asr_result_and_fences_cloud() -> None:
    agent, coordinator, _tts, _started = _bare_agent()
    coordinator.finish_result = False
    coordinator.generation = 2
    agent._realtime_generation_at_listen_start = 1
    agent._turn_traces = MagicMock()
    agent.audio_queue = queue.Queue()
    agent._metrics = MagicMock()
    agent._clear_input_failure = lambda: None  # type: ignore[method-assign]
    agent._refresh_voice_metrics = lambda: {}  # type: ignore[method-assign]
    agent._asr_mgr = MagicMock()

    assert agent._accept_result("本地结果") == "本地结果"

    assert agent.audio_queue.get_nowait() == "本地结果"
    _wait_until(lambda: coordinator.close_reasons == ["finish_input_failed"])
    assert agent._realtime_capture_armed is False
    assert agent.last_turn_realtime_generation == 0


def test_realtime_recovery_arms_capture_only_on_the_next_turn_boundary() -> None:
    agent, coordinator, _tts, _started = _bare_agent()
    coordinator.active = False
    coordinator.recovery_release = threading.Event()
    agent._realtime_capture_armed = False

    assert agent._prepare_realtime_turn_boundary() is False
    assert coordinator.recovery_started.wait(timeout=1.0)
    samples = np.asarray([1, -1], dtype=np.int16)
    assert agent._offer_realtime_capture(samples, sample_rate=16_000) is False
    assert coordinator.offered == []

    coordinator.recovery_release.set()
    _wait_until(lambda: agent._realtime_recovery_thread is None)
    assert coordinator.active is True
    assert agent._realtime_capture_armed is False

    assert agent._prepare_realtime_turn_boundary() is True
    assert agent._offer_realtime_capture(samples, sample_rate=16_000) is True
    assert len(coordinator.offered) == 1


def test_failed_realtime_recovery_keeps_the_whole_turn_on_cascade() -> None:
    agent, coordinator, _tts, _started = _bare_agent()
    coordinator.active = False
    coordinator.recovery_result = False
    agent._realtime_capture_armed = False

    assert agent._prepare_realtime_turn_boundary() is False
    _wait_until(lambda: agent._realtime_recovery_thread is None)

    assert agent._realtime_capture_armed is False
    assert agent._realtime_recovery_failures == 1
    assert agent._offer_realtime_capture(
        np.asarray([1, -1], dtype=np.int16),
        sample_rate=16_000,
    ) is False


def test_shutdown_cancels_recovery_without_leaving_a_reopened_session() -> None:
    agent, coordinator, _tts, _started = _bare_agent()
    coordinator.active = False
    coordinator.recovery_release = threading.Event()
    agent._realtime_capture_armed = False

    assert agent._prepare_realtime_turn_boundary() is False
    assert coordinator.recovery_started.wait(timeout=1.0)
    agent.stop_event.set()
    coordinator.recovery_release.set()
    agent._stop_realtime_recovery("shutdown")

    assert agent._realtime_recovery_thread is None
    assert agent._realtime_capture_armed is False
    assert coordinator.active is False
    assert coordinator.close_reasons == ["recovery_cancelled"]


def test_replaced_coordinator_is_not_started_after_status_probe() -> None:
    agent, coordinator, _tts, _started = _bare_agent()
    coordinator.active = False
    replacement = _Coordinator()
    status_started = threading.Event()
    status_release = threading.Event()
    original_status = coordinator.status_snapshot

    def _blocked_status() -> dict[str, object]:
        status_started.set()
        assert status_release.wait(timeout=1.0)
        return original_status()

    coordinator.status_snapshot = _blocked_status  # type: ignore[method-assign]
    result: list[bool] = []
    worker = threading.Thread(
        target=lambda: result.append(agent._prepare_realtime_turn_boundary())
    )
    worker.start()
    assert status_started.wait(timeout=1.0)
    agent._realtime_coordinator = replacement
    status_release.set()
    worker.join(timeout=1.0)

    assert result == [False]
    assert coordinator.recovery_calls == 0


def test_successfully_recovered_obsolete_coordinator_is_closed() -> None:
    agent, coordinator, _tts, _started = _bare_agent()
    coordinator.active = False
    coordinator.recovery_release = threading.Event()
    replacement = _Coordinator()

    assert agent._prepare_realtime_turn_boundary() is False
    assert coordinator.recovery_started.wait(timeout=1.0)
    agent._realtime_coordinator = replacement
    coordinator.recovery_release.set()
    _wait_until(lambda: agent._realtime_recovery_thread is None)

    assert coordinator.active is False
    assert coordinator.close_reasons == ["recovery_obsolete"]
    assert replacement.close_reasons == []


def test_general_chat_approval_releases_streaming_pcm_on_current_tts_generation() -> None:
    agent, coordinator, tts, started = _bare_agent()

    approval = _release_two_phase(agent, expected_generation=3)

    assert approval is not None
    assert coordinator.approved == ["你好"]
    assert coordinator.expected_generations == [3]
    assert started == ["started"]
    assert len(tts.queued) == 1
    samples, sample_rate, final, generation = tts.queued[0]
    assert samples.tolist() == [1 / 32768.0, -2 / 32768.0]
    assert sample_rate == 24_000
    assert final is False
    assert generation == 7


def test_stale_realtime_discard_cannot_clear_successor_generation() -> None:
    agent, coordinator, _tts, _started = _bare_agent()
    with agent._realtime_output_lock:
        agent._realtime_output_tts_generation = 17
        agent._realtime_output_provider_generation = 9
        agent._realtime_output_started = True
        agent._realtime_output_voice_turn_id = "turn-b"

    agent.discard_realtime_turn(
        "late_turn_a_cleanup",
        expected_generation=8,
    )

    with agent._realtime_output_lock:
        assert agent._realtime_output_tts_generation == 17
        assert agent._realtime_output_provider_generation == 9
        assert agent._realtime_output_started is True
        assert agent._realtime_output_voice_turn_id == "turn-b"
    assert coordinator.discarded == [("late_turn_a_cleanup", 8, 0)]


def test_stale_realtime_abort_cannot_stop_successor_playback() -> None:
    agent, coordinator, tts, _started = _bare_agent()
    _release_two_phase(agent, expected_generation=9)
    successor_token = agent._active_output_trace_token
    assert successor_token is not None
    tts.stopped = 0
    tts.drained = 0
    tts.joined = 0

    assert (
        agent.abort_realtime_playback(
            "late_turn_a_abort",
            expected_generation=8,
        )
        is False
    )

    assert agent._active_output_trace_token == successor_token
    assert tts.stopped == 0
    assert tts.drained == 0
    assert tts.joined == 0
    with agent._realtime_output_lock:
        assert agent._realtime_output_provider_generation == 9
        assert agent._realtime_output_started is True
    assert coordinator.discarded == [("late_turn_a_abort", 8, 0)]


def test_legacy_one_step_realtime_admission_fails_closed_without_pcm() -> None:
    agent, coordinator, tts, started = _bare_agent()

    approval = agent.try_realtime_general_chat(
        "小算，带我去前台",
        expected_generation=3,
    )

    assert approval is None
    assert coordinator.approved == []
    assert coordinator.discarded == [
        ("legacy_one_step_admission_disabled", 3, 0)
    ]
    assert tts.queued == []
    assert started == []


def test_confirmed_barge_in_without_frozen_playback_owner_is_a_noop() -> None:
    agent, coordinator, tts, _started = _bare_agent()
    callbacks: list[str] = []
    agent._barge_in_callback = lambda: callbacks.append("pipeline")

    assert agent._notify_confirmed_barge_in() is False

    assert coordinator.discarded == []
    assert tts.stopped == 0
    assert tts.drained == 0
    assert callbacks == []


def test_confirmed_barge_in_truncates_at_physically_played_provider_audio() -> None:
    agent, coordinator, tts, _started = _bare_agent()
    callbacks: list[str] = []
    agent._barge_in_callback = lambda: callbacks.append("pipeline")
    _release_two_phase(agent, expected_generation=3)
    _arm_barge_in(agent)
    tts.played_ms = 240

    agent._notify_confirmed_barge_in()

    _wait_until(lambda: bool(coordinator.truncated))
    assert coordinator.truncated == [("barge_in", 240, 3)]
    assert coordinator.discarded == []
    assert callbacks == ["pipeline"]


def test_repeated_barge_in_does_not_truncate_then_delete_the_same_turn() -> None:
    agent, coordinator, tts, _started = _bare_agent()
    agent._barge_in_callback = None
    _release_two_phase(agent, expected_generation=3)
    _arm_barge_in(agent)
    tts.played_ms = 240

    agent._notify_confirmed_barge_in()
    agent._notify_confirmed_barge_in()

    _wait_until(lambda: bool(coordinator.truncated))
    assert coordinator.truncated == [("barge_in", 240, 3)]
    assert coordinator.discarded == []


def test_realtime_truncate_failure_falls_back_to_history_delete() -> None:
    agent, coordinator, tts, _started = _bare_agent()
    agent._barge_in_callback = None
    coordinator.truncate_result = False
    _release_two_phase(agent, expected_generation=4)
    _arm_barge_in(agent)
    tts.played_ms = 180

    agent._notify_confirmed_barge_in()

    _wait_until(lambda: bool(coordinator.truncated))
    assert coordinator.truncated == [("barge_in", 180, 4)]
    assert coordinator.discarded == [("barge_in", 4, 0)]


def test_coordinator_owned_truncate_fallback_is_not_deleted_twice() -> None:
    agent, coordinator, tts, _started = _bare_agent()
    agent._barge_in_callback = None
    coordinator.truncate_result = False
    coordinator.truncate_handles_failure = True
    _release_two_phase(agent, expected_generation=4)
    _arm_barge_in(agent)
    tts.played_ms = 180

    agent._notify_confirmed_barge_in()

    _wait_until(lambda: bool(coordinator.truncated))
    assert coordinator.truncated == [("barge_in", 180, 4)]
    assert coordinator.discarded == [("barge_in_truncate_failed", 4, 0)]


def test_realtime_abort_truncates_once_then_stops_physical_playback() -> None:
    agent, coordinator, tts, _started = _bare_agent()
    agent._refresh_voice_metrics = lambda: None  # type: ignore[method-assign]
    _release_two_phase(agent, expected_generation=5)
    tts.played_ms = 90

    agent.abort_realtime_playback("response_cancelled")

    _wait_until(lambda: bool(coordinator.truncated))
    assert coordinator.truncated == [("response_cancelled", 90, 5)]
    assert coordinator.discarded == []
    assert tts.drained == 1
    assert tts.stopped == 1


def test_realtime_snapshot_retains_frozen_playhead_after_barge_in() -> None:
    agent, coordinator, tts, _started = _bare_agent()
    agent._barge_in_callback = None
    _release_two_phase(agent, expected_generation=5)
    _arm_barge_in(agent)
    tts.played_ms = 135

    agent._notify_confirmed_barge_in()

    _wait_until(lambda: bool(coordinator.truncated))
    assert agent.realtime_context_snapshot()["physical_played_ms"] == 135


def test_barge_in_stops_physical_audio_before_waiting_for_cloud_truncate() -> None:
    agent, coordinator, tts, _started = _bare_agent()
    call_order: list[str] = []
    truncate_entered = threading.Event()
    truncate_release = threading.Event()
    tts.played_ms = 120
    original_stop = tts.stop_immediately

    def _stop() -> None:
        call_order.append("stop")
        original_stop()

    def _truncate(
        reason: str,
        *,
        audio_end_ms: int,
        expected_generation: int = 0,
    ) -> bool:
        del reason, audio_end_ms, expected_generation
        call_order.append("truncate")
        truncate_entered.set()
        assert truncate_release.wait(timeout=1.0)
        return True

    tts.stop_immediately = _stop  # type: ignore[method-assign]
    coordinator.truncate_current = _truncate  # type: ignore[method-assign]
    agent._barge_in_callback = lambda: call_order.append("pipeline")
    _release_two_phase(agent, expected_generation=6)
    _arm_barge_in(agent)

    started_at = time.monotonic()
    agent._notify_confirmed_barge_in()
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.1
    assert truncate_entered.wait(timeout=1.0)
    assert call_order.index("stop") < call_order.index("truncate")
    assert "pipeline" in call_order
    truncate_release.set()


def test_input_failure_quarantines_provider_turn_started_during_capture() -> None:
    agent, coordinator, _tts, _started = _bare_agent()
    coordinator.generation = 3
    agent._realtime_generation_at_listen_start = 2
    agent._input_state_lock = threading.Lock()
    agent._input_asr_timeouts = 0
    agent._input_last_failure_reason = None
    agent._input_vad_state = "listening"

    agent._mark_input_failure("asr_timeout")

    assert coordinator.discarded == [("asr_timeout", 3, 2)]


def test_provider_stack_attaches_realtime_session_to_production_audio_path(
    monkeypatch,
) -> None:
    from askme.providers.voice import build_audio_frontend

    session = MagicMock()
    attached: list[tuple[object, object]] = []

    class _AEC:
        def stats(self) -> dict[str, object]:
            return {"available": False}

    class _Audio:
        def __init__(self, *args, **kwargs) -> None:
            self._asr_mgr = object()
            self.tts = object()

        def configure_realtime_dialogue(self, provider, config) -> None:
            attached.append((provider, config))

    monkeypatch.setattr(
        "askme.voice.input.aec_processor.create_aec_processor",
        lambda **kwargs: _AEC(),
    )
    monkeypatch.setattr(
        "askme.voice.input.full_duplex_gate.decide_full_duplex",
        lambda *args, **kwargs: SimpleNamespace(requested=False),
    )
    monkeypatch.setattr(
        "askme.voice.orchestration.full_duplex_setup.configure_full_duplex",
        lambda **kwargs: SimpleNamespace(
            enabled=False,
            reason="not_requested",
            echo_control="none",
            aec_backend="unavailable",
        ),
    )
    monkeypatch.setattr("askme.voice.orchestration.audio_agent.AudioAgent", _Audio)
    monkeypatch.setattr(
        "askme.voice.output.audio_router.AudioRouter",
        lambda: object(),
    )
    monkeypatch.setattr(
        "askme.voice.realtime.factory.build_realtime_dialogue",
        lambda config: session,
    )

    stack = build_audio_frontend(
        {
            "voice": {
                "realtime": {
                    "enabled": True,
                    "mode": "general_chat",
                    "app_id": "app",
                    "access_token": "token",
                }
            }
        }
    )

    assert stack.realtime is session
    assert attached[0][0] is session
    assert attached[0][1].mode.value == "general_chat"
