"""Tests for AudioAgent: noise filtering, confirmation context, echo gate,
barge-in hold, agent state transitions, mute/unmute, volume/speed delegation."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from contextvars import Context
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from askme.voice.audio_agent import (
    _BARGE_IN_HOLD_S,
    _CONFIRMATION_WORDS,
    _MAX_SPEECH_DURATION,
    _MIN_VALID_TEXT_LEN,
    _NOISE_UTTERANCES,
    _SINGLE_CHAR_COMMANDS,
    AgentState,
    AudioAgent,
)

from askme.voice.core.turn_timeline import (
    TimelineQuery,
    VoiceTimelineStage,
    VoiceTurnTimeline,
)
from askme.voice.core.turn_trace import VoiceTurnTraceRecorder
from askme.voice.input.vad_controller import VADEvent

# AudioAgent constructor validates sherpa-onnx ASR model files exist on disk.
# Skip the construction-dependent tests when the ~100MB model is absent
# (e.g. CI without model download). Tests of pure constants below stay enabled.
_ASR_MODEL_TOKENS = Path(
    "models/asr/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt"
)
_requires_asr_model = pytest.mark.skipif(
    not _ASR_MODEL_TOKENS.exists(),
    reason=f"ASR model not present at {_ASR_MODEL_TOKENS}",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(voice_mode: bool = False, **voice_overrides) -> AudioAgent:
    """Create an AudioAgent in text mode (no real audio devices)."""
    config = {"voice": {"tts": {"backend": "edge"}, **voice_overrides}}
    metrics = MagicMock()
    metrics.update_voice_state = MagicMock()
    metrics.mark_voice_listen_started = MagicMock()
    metrics.mark_voice_input = MagicMock()
    metrics.mark_voice_error = MagicMock()
    return AudioAgent(config, voice_mode=voice_mode, metrics=metrics)


def test_audio_agent_routes_late_playback_to_the_accepted_voice_turn() -> None:
    timeline = VoiceTurnTimeline()
    recorder = VoiceTurnTraceRecorder(timeline=timeline)
    metrics = MagicMock()
    agent = AudioAgent(
        {"voice": {"tts": {"backend": "edge"}}},
        voice_mode=False,
        metrics=metrics,
        turn_traces=recorder,
    )
    try:
        output_turn = recorder.start(source="microphone", media_transport="sounddevice")
        recorder.finish("accepted")
        capture_turn = recorder.start(source="microphone", media_transport="sounddevice")
        agent.tts.start_playback = MagicMock()
        agent.tts.stop_playback = MagicMock()

        token = agent.start_playback(voice_turn_id=output_turn.voice_turn_id)
        assert token is not None
        agent._mark_output_voice_trace("barge_in_detected", peak=1200, rms=48.0)
        agent._mark_output_barge_in(peak=1400, rms=52.0)
        agent.stop_playback(token)

        output_stages = {
            event.stage
            for event in timeline.snapshot(
                TimelineQuery(voice_turn_id=output_turn.voice_turn_id, limit=100)
            ).events
        }
        capture_stages = {
            event.stage
            for event in timeline.snapshot(
                TimelineQuery(voice_turn_id=capture_turn.voice_turn_id, limit=100)
            ).events
        }
        assert {
            VoiceTimelineStage.SPEAKER_RENDER_STARTED,
            VoiceTimelineStage.SPEAKER_RENDER_STOPPED,
            VoiceTimelineStage.INTERRUPT_DETECTED,
            VoiceTimelineStage.INTERRUPT_CONFIRMED,
        } <= output_stages
        assert VoiceTimelineStage.SPEAKER_RENDER_STARTED not in capture_stages
        assert VoiceTimelineStage.SPEAKER_RENDER_STOPPED not in capture_stages
        assert VoiceTimelineStage.INTERRUPT_CONFIRMED not in capture_stages
    finally:
        agent.shutdown()


def test_audio_agent_drops_output_evidence_without_an_owning_turn() -> None:
    timeline = VoiceTurnTimeline()
    recorder = VoiceTurnTraceRecorder(timeline=timeline)
    agent = AudioAgent(
        {"voice": {"tts": {"backend": "edge"}}},
        voice_mode=False,
        metrics=MagicMock(),
        turn_traces=recorder,
    )
    try:
        capture_turn = recorder.start(source="microphone", media_transport="sounddevice")
        agent.tts.start_playback = MagicMock()
        agent.tts.stop_playback = MagicMock()

        agent.start_playback()
        agent.stop_playback()

        stages = {
            event.stage
            for event in timeline.snapshot(
                TimelineQuery(voice_turn_id=capture_turn.voice_turn_id, limit=100)
            ).events
        }
        assert stages == {VoiceTimelineStage.LISTEN_STARTED}
        assert agent._orphan_output_trace_event_count == 2
    finally:
        agent.shutdown()


def test_stale_playback_token_cannot_stop_or_close_its_successor() -> None:
    timeline = VoiceTurnTimeline()
    recorder = VoiceTurnTraceRecorder(timeline=timeline)
    agent = AudioAgent(
        {"voice": {"tts": {"backend": "edge"}}},
        voice_mode=False,
        metrics=MagicMock(),
        turn_traces=recorder,
    )
    try:
        first = recorder.start(source="microphone", media_transport="sounddevice")
        recorder.finish("accepted")
        second = recorder.start(source="microphone", media_transport="sounddevice")
        recorder.finish("accepted")
        agent.tts.start_playback = MagicMock()
        agent.tts.stop_playback = MagicMock()
        agent.tts.stop_immediately = MagicMock()
        agent.tts.drain_buffers = MagicMock()

        first_token = agent.start_playback(voice_turn_id=first.voice_turn_id)
        assert first_token is not None
        agent.stop_immediately()
        agent.tts.stop_playback.reset_mock()
        second_token = agent.start_playback(voice_turn_id=second.voice_turn_id)
        assert second_token is not None

        agent.stop_playback(first_token)

        second_stages = {
            event.stage
            for event in timeline.snapshot(
                TimelineQuery(voice_turn_id=second.voice_turn_id, limit=100)
            ).events
        }
        assert VoiceTimelineStage.SPEAKER_RENDER_STARTED in second_stages
        assert VoiceTimelineStage.SPEAKER_RENDER_STOPPED not in second_stages
        agent.tts.stop_playback.assert_not_called()

        agent.stop_playback(second_token)
        second_stages = {
            event.stage
            for event in timeline.snapshot(
                TimelineQuery(voice_turn_id=second.voice_turn_id, limit=100)
            ).events
        }
        assert VoiceTimelineStage.SPEAKER_RENDER_STOPPED in second_stages
        agent.tts.stop_playback.assert_called_once_with()
    finally:
        agent.shutdown()


def test_stale_playback_token_cannot_close_successor_interruption_hold() -> None:
    agent = AudioAgent(
        {"voice": {"tts": {"backend": "edge"}}},
        voice_mode=False,
        metrics=MagicMock(),
    )
    try:
        agent.tts.start_playback = MagicMock()
        agent.tts.stop_playback = MagicMock()
        agent.tts.stop_immediately = MagicMock()
        agent.tts.drain_buffers = MagicMock()
        first = agent.start_playback(voice_turn_id="turn-a")
        assert first is not None
        agent.stop_immediately()
        second = agent.start_playback(voice_turn_id="turn-b")
        assert second is not None
        recovery = MagicMock()
        agent._interruption_recovery = recovery
        agent._interruption_output_trace_token = second

        agent.stop_playback(first)

        recovery.close.assert_not_called()
        assert agent._active_output_trace_token == second
    finally:
        agent.shutdown()


def test_stale_recovery_callback_cannot_resume_successor_hold() -> None:
    agent = AudioAgent(
        {"voice": {"tts": {"backend": "edge"}}},
        voice_mode=False,
        metrics=MagicMock(),
    )
    try:
        agent.tts.start_playback = MagicMock()
        agent.tts.stop_playback = MagicMock()
        agent.tts.stop_immediately = MagicMock()
        agent.tts.drain_buffers = MagicMock()
        first = agent.start_playback(voice_turn_id="turn-a")
        assert first is not None
        agent.stop_immediately()
        second = agent.start_playback(voice_turn_id="turn-b")
        assert second is not None
        recovery = MagicMock()
        agent._interruption_recovery = recovery
        agent._interruption_output_trace_token = second
        stale_before = agent._stale_playback_stop_count

        recovered = agent._recover_interrupted_playback(
            "vad_dismissed",
            expected_token=first,
        )

        assert recovered is False
        recovery.recover.assert_not_called()
        assert agent._interruption_output_trace_token == second
        assert agent._active_output_trace_token == second
        assert agent._stale_playback_stop_count == stale_before + 1
    finally:
        agent.shutdown()


def test_playback_owner_conflict_cannot_enqueue_text_into_active_turn() -> None:
    agent = AudioAgent(
        {"voice": {"tts": {"backend": "edge"}}},
        voice_mode=False,
        metrics=MagicMock(),
    )
    try:
        agent.tts.start_playback = MagicMock()
        agent.tts.stop_playback = MagicMock()
        agent.tts.speak = MagicMock()
        first = agent.start_playback(voice_turn_id="turn-a")
        assert first is not None

        def _competing_turn() -> None:
            assert agent.start_playback(voice_turn_id="turn-b") is None
            with pytest.raises(RuntimeError, match="playback owner conflict"):
                agent.speak("must-not-enter-turn-a")

        Context().run(_competing_turn)

        agent.tts.speak.assert_not_called()
        assert agent._active_output_trace_token == first
        agent.stop_playback(first)
    finally:
        agent.shutdown()


def test_immediate_stop_is_joined_before_successor_can_start() -> None:
    agent = AudioAgent(
        {"voice": {"tts": {"backend": "edge"}}},
        voice_mode=False,
        metrics=MagicMock(),
    )
    events: list[str] = []
    try:
        agent.tts.start_playback = MagicMock(
            side_effect=lambda: events.append("start")
        )
        agent.tts.stop_immediately = MagicMock(
            side_effect=lambda: events.append("signal_stop")
        )
        agent.tts.drain_buffers = MagicMock(
            side_effect=lambda: events.append("drain_generation")
        )
        agent.tts.stop_playback = MagicMock(
            side_effect=lambda: events.append("join_playback")
        )
        first = agent.start_playback(voice_turn_id="turn-a")
        assert first is not None
        events.clear()

        agent.stop_immediately()
        second = agent.start_playback(voice_turn_id="turn-b")

        assert second is not None
        assert events == [
            "signal_stop",
            "drain_generation",
            "join_playback",
            "start",
        ]
    finally:
        agent.shutdown()


def test_audio_router_controls_audio_agent_persistent_microphone() -> None:
    from askme.voice.audio_router import AudioRouter

    router = AudioRouter()
    config = {"voice": {"tts": {"backend": "edge"}}}
    metrics = MagicMock()
    agent = AudioAgent(
        config,
        voice_mode=False,
        metrics=metrics,
        audio_router=router,
    )
    mic = MagicMock()
    agent._mic = mic
    agent.voice_mode = True
    agent._input_requested = True
    try:
        with router.output_session():
            mic.stop.assert_called_once_with()
            mic.start.assert_not_called()
        mic.start.assert_called_once_with()
    finally:
        agent.voice_mode = False
        agent.shutdown()


class _SingleFrameMicContext:
    def __init__(
        self,
        samples: np.ndarray,
        *,
        stop_event: threading.Event,
    ) -> None:
        self.sample_rate = 16_000
        self.pre_roll: deque[np.ndarray] = deque()
        self._samples = samples
        self._stop_event = stop_event

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read_chunk(self) -> np.ndarray:
        self._stop_event.set()
        return self._samples.copy()

    def buffer_pre_roll(self, samples: np.ndarray) -> None:
        self.pre_roll.append(samples.copy())


class _SingleFrameMic:
    def __init__(self, context: _SingleFrameMicContext) -> None:
        self._context = context

    def _flush_queue(self) -> None:
        return None

    def open(self) -> _SingleFrameMicContext:
        return self._context


class _RecordingVad:
    def __init__(self, event: VADEvent) -> None:
        self.event = event
        self.speech_active = False
        self.barge_in_buffer: list[np.ndarray] = []
        self.feed_calls: list[tuple[np.ndarray, int, bool]] = []

    def reset(self) -> None:
        self.speech_active = False

    def feed(
        self,
        samples_i16: np.ndarray,
        peak: int,
        *,
        tts_active: bool,
    ) -> VADEvent:
        self.feed_calls.append((samples_i16.copy(), peak, tts_active))
        return self.event


class _RecordingAsr:
    _cloud_active = False

    def __init__(self) -> None:
        self.start_session_calls = 0
        self.audio_frames: list[np.ndarray] = []

    def preconnect_cloud(self) -> None:
        return None

    def start_session(self) -> None:
        self.start_session_calls += 1

    def feed_audio(self, samples_f32, samples_i16, sample_rate: int) -> None:
        self.audio_frames.append(np.asarray(samples_i16).copy())

    def check_endpoint(self):
        return None

    def reset(self) -> None:
        return None


def _single_frame_agent(
    samples: np.ndarray,
    event: VADEvent,
    *,
    tts_active: bool,
) -> tuple[AudioAgent, _RecordingVad, _RecordingAsr, _SingleFrameMicContext]:
    agent = _make_agent(echo_gate_peak=800)
    context = _SingleFrameMicContext(samples, stop_event=agent.stop_event)
    vad = _RecordingVad(event)
    asr = _RecordingAsr()
    agent.asr = object()
    agent.vad = object()
    agent.kws = None
    agent._mic = _SingleFrameMic(context)
    agent._vad_ctrl = vad
    agent._asr_mgr = asr
    agent._refresh_voice_metrics = MagicMock()
    agent._apply_pending_runtime_updates = MagicMock()
    agent.tts.is_active = MagicMock(return_value=tts_active)
    return agent, vad, asr, context


def test_capture_processor_output_reaches_vad_and_disables_legacy_echo_gate() -> None:
    raw = np.full(160, 0.01, dtype=np.float32)
    agent, vad, _asr, _context = _single_frame_agent(
        raw,
        VADEvent.SILENCE,
        tts_active=True,
    )
    calls: list[tuple[np.ndarray, int, bool]] = []

    def clean_capture(samples, sample_rate_hz: int, tts_active: bool):
        calls.append((samples.copy(), sample_rate_hz, tts_active))
        return np.full_like(samples, 0.25)

    agent.set_capture_processor(clean_capture)
    try:
        assert agent.listen_loop() is None
    finally:
        agent.shutdown()

    assert len(calls) == 1
    assert calls[0][1:] == (16_000, True)
    assert len(vad.feed_calls) == 1
    samples_i16, peak, tts_active = vad.feed_calls[0]
    assert peak == 8191
    assert tts_active is True
    assert np.all(samples_i16 == 8191)


def test_capture_processor_failure_keeps_original_echo_gate() -> None:
    raw = np.full(160, 0.01, dtype=np.float32)
    agent, vad, _asr, context = _single_frame_agent(
        raw,
        VADEvent.SILENCE,
        tts_active=True,
    )

    def fail_capture(samples, sample_rate_hz: int, tts_active: bool):
        raise RuntimeError("aec unavailable")

    failures: list[BaseException] = []
    agent.set_capture_processor(fail_capture, on_failure=failures.append)
    try:
        assert agent.listen_loop() is None
    finally:
        agent.shutdown()

    assert vad.feed_calls == []
    assert len(context.pre_roll) == 1
    assert len(failures) == 1
    assert str(failures[0]) == "aec unavailable"


def test_capture_processor_failure_suppresses_current_tts_echo_frame() -> None:
    raw = np.full(160, 0.25, dtype=np.float32)
    agent, vad, _asr, context = _single_frame_agent(
        raw,
        VADEvent.SPEECH_START,
        tts_active=True,
    )
    agent._audio_proc._echo_gate_peak = 0

    def fail_capture(samples, sample_rate_hz: int, tts_active: bool):
        raise RuntimeError("aec failed mid-playback")

    def restore_half_duplex(_exc: BaseException) -> None:
        agent._audio_proc._echo_gate_peak = 800

    agent.set_capture_processor(fail_capture, on_failure=restore_half_duplex)
    try:
        assert agent.listen_loop() is None
    finally:
        agent.shutdown()

    assert vad.feed_calls == []
    assert len(context.pre_roll) == 1


def test_confirmed_barge_in_waits_for_asr_before_cancelling_playback() -> None:
    raw = np.full(160, 0.25, dtype=np.float32)
    agent, vad, asr, _context = _single_frame_agent(
        raw,
        VADEvent.BARGE_IN_CONFIRMED,
        tts_active=True,
    )
    vad.barge_in_buffer.append(raw.copy())
    events: list[str] = []
    token = object()
    agent.tts.pause_playback = MagicMock(
        side_effect=lambda **_kwargs: events.append("pause") or token
    )
    agent.tts.resume_playback = MagicMock(
        side_effect=lambda actual: events.append("resume") or actual is token
    )
    agent.tts.abort_playback_hold = MagicMock(
        side_effect=lambda actual: events.append("abort") or actual is token
    )
    agent.tts.start_playback = MagicMock()
    agent.tts.drain_buffers = MagicMock(side_effect=lambda: events.append("drain"))
    agent.tts.stop_immediately = MagicMock(side_effect=lambda: events.append("stop"))
    agent.set_barge_in_callback(lambda: events.append("callback"))
    playback_token = agent.start_playback()
    assert playback_token is not None

    try:
        assert agent.listen_loop() is None
        assert events == ["pause", "resume"]
        assert asr.start_session_calls == 1
    finally:
        agent.shutdown()


def test_wake_word_wait_records_microphone_frames() -> None:
    agent = object.__new__(AudioAgent)
    agent.stop_event = threading.Event()
    agent._record_input_observation = MagicMock()
    agent._refresh_voice_metrics = MagicMock()
    agent.kws_stream = MagicMock()
    agent.kws = MagicMock()
    agent.kws.spotter.is_ready.return_value = False
    agent.kws.spotter.get_result.return_value = ""

    samples = np.array([0.25, -0.5], dtype=np.float32)
    mic = MagicMock()
    mic.sample_rate = 16000

    def read_chunk() -> np.ndarray:
        agent.stop_event.set()
        return samples

    mic.read_chunk.side_effect = read_chunk

    assert agent._wait_for_wake_word_mic(mic) is False
    agent._record_input_observation.assert_called_once_with(
        peak=16384,
        rms=12952.69,
        vad_state="wake_word",
        gate_state="open",
    )


def test_kws_runtime_error_marks_stream_unavailable_for_fail_closed_mode() -> None:
    agent = object.__new__(AudioAgent)
    agent.voice_mode = True
    agent._require_wake_word = True
    agent.stop_event = threading.Event()
    agent._record_input_observation = MagicMock()
    agent._refresh_voice_metrics = MagicMock()
    agent.kws_stream = MagicMock()
    agent.kws = MagicMock()
    agent.kws.available = True
    agent.kws_stream.accept_waveform.side_effect = RuntimeError("kws failed")
    mic = MagicMock()
    mic.sample_rate = 16_000
    mic.read_chunk.return_value = np.zeros(160, dtype=np.float32)

    assert agent._wait_for_wake_word_mic(mic) is False
    assert agent.kws_stream is None
    assert agent.kws_unavailable_safety_only is True
    agent._refresh_voice_metrics.assert_called_once()


def test_asr_result_does_not_renew_followup_window_before_admission() -> None:
    agent = object.__new__(AudioAgent)
    agent._turn_traces = MagicMock()
    agent.audio_queue = MagicMock()
    agent._metrics = MagicMock()
    agent._clear_input_failure = MagicMock()
    agent._refresh_voice_metrics = MagicMock()
    agent._asr_mgr = MagicMock()
    agent._last_interaction_time = 123.0
    agent._agent_state = AgentState.LISTENING

    assert agent._accept_result("旁人聊天", asr_source="cloud") == "旁人聊天"
    assert agent._last_interaction_time == 123.0

    agent.mark_interaction_turn()
    assert agent._last_interaction_time > 123.0


def test_accepted_asr_result_logs_only_non_content_metadata(caplog) -> None:
    agent = object.__new__(AudioAgent)
    agent._turn_traces = MagicMock()
    agent.audio_queue = MagicMock()
    agent._metrics = MagicMock()
    agent._clear_input_failure = MagicMock()
    agent._refresh_voice_metrics = MagicMock()
    agent._asr_mgr = MagicMock()
    agent._last_interaction_time = 0.0
    agent._agent_state = AgentState.LISTENING
    private_phrase = "PRIVATE-ASR-不要写入日志-7391"

    with caplog.at_level("INFO"):
        assert agent._accept_result(private_phrase, asr_source="cloud") == private_phrase

    assert private_phrase not in caplog.text
    assert f"chars={len(private_phrase)}" in caplog.text


def test_kws_unavailable_product_mode_filters_ordinary_transcript() -> None:
    agent = _make_agent(
        product_readiness={"require_wake_word": True},
    )
    agent.voice_mode = True
    agent.kws = None
    agent.last_turn_wake_source = "kws_unavailable_safety_only"
    agent._turn_traces = MagicMock()
    agent._discard_realtime_capture_if_started = MagicMock()
    try:
        result = agent._accept_result(
            "老王说帮我查一下天气",
            asr_source="cloud",
        )

        assert result is None
        assert agent.audio_queue.empty()
        agent._metrics.mark_voice_input.assert_not_called()
        agent._discard_realtime_capture_if_started.assert_called_once_with(
            "kws_unavailable_safety_only_filtered"
        )
    finally:
        agent.voice_mode = False
        agent.shutdown()


@pytest.mark.parametrize(
    "safety_text",
    [
        "停",
        "停下来！",
        "别动",
        "不要动",
        "停止说话",
        "关闭麦克风",
        "开麦",
    ],
)
def test_kws_unavailable_product_mode_admits_only_local_safety(
    safety_text: str,
) -> None:
    agent = _make_agent(
        product_readiness={"require_wake_word": True},
    )
    agent.voice_mode = True
    agent.kws = None
    agent.last_turn_wake_source = "kws_unavailable_safety_only"
    try:
        assert agent._accept_result(safety_text, asr_source="cloud") == safety_text
        assert agent.audio_queue.get_nowait() == safety_text
        agent._metrics.mark_voice_input.assert_called_once_with(safety_text)
    finally:
        agent.voice_mode = False
        agent.shutdown()


def test_kws_unavailable_product_mode_is_visible_in_health() -> None:
    agent = _make_agent(
        product_readiness={"require_wake_word": True},
    )
    agent.voice_mode = True
    agent.kws = None
    try:
        snapshot = agent.status_snapshot()

        assert snapshot["kws_required"] is True
        assert snapshot["kws_unavailable_safety_only"] is True
        assert snapshot["input_policy"] == "kws_unavailable_safety_only"
    finally:
        agent.voice_mode = False
        agent.shutdown()


def test_missing_kws_stream_fails_closed_even_when_engine_is_loaded() -> None:
    agent = _make_agent(
        product_readiness={"require_wake_word": True},
    )
    kws = MagicMock()
    kws.available = True
    agent.voice_mode = True
    agent.kws = kws
    agent.kws_stream = None
    try:
        snapshot = agent.status_snapshot()

        assert snapshot["kws_available"] is True
        assert snapshot["wake_word_enabled"] is False
        assert snapshot["kws_unavailable_safety_only"] is True
        assert agent._accept_result("旁人聊天", asr_source="cloud") is None
        assert agent.audio_queue.empty()
    finally:
        agent.voice_mode = False
        agent.shutdown()


def test_kws_unavailable_product_mode_never_arms_realtime_llm() -> None:
    agent = _make_agent(
        product_readiness={"require_wake_word": True},
    )
    coordinator = MagicMock()
    coordinator.status_snapshot.return_value = {
        "active": True,
        "quarantined": False,
    }
    agent.voice_mode = True
    agent.kws = None
    agent._realtime_coordinator = coordinator
    try:
        assert agent._prepare_realtime_turn_boundary() is False
        assert agent._realtime_turn_capture_is_armed() is False
        coordinator.status_snapshot.assert_not_called()
    finally:
        agent._realtime_coordinator = None
        agent.voice_mode = False
        agent.shutdown()


def test_kws_unavailable_product_mode_never_offers_audio_to_realtime_llm() -> None:
    agent = _make_agent(
        product_readiness={"require_wake_word": True},
    )
    coordinator = MagicMock()
    coordinator.offer_audio.return_value = True
    agent.voice_mode = True
    agent.kws = None
    agent._realtime_coordinator = coordinator
    with agent._realtime_recovery_lock:
        agent._realtime_capture_armed = True
    try:
        assert (
            agent._offer_realtime_capture(
                np.zeros(160, dtype=np.int16),
                sample_rate=16_000,
            )
            is False
        )
        coordinator.offer_audio.assert_not_called()
    finally:
        agent._realtime_coordinator = None
        agent.voice_mode = False
        agent.shutdown()


def test_followup_window_never_claims_real_wake_authorization_without_kws() -> None:
    samples = np.zeros(160, dtype=np.float32)
    agent, _vad, _asr, _context = _single_frame_agent(
        samples,
        VADEvent.SILENCE,
        tts_active=False,
    )
    agent._require_wake_word = False
    agent._last_interaction_time = time.monotonic()
    try:
        assert agent.listen_loop() is None

        assert agent.last_turn_wake_source == "followup_window"
        assert agent.last_turn_wake_authorized is False
    finally:
        agent.shutdown()


def test_followup_window_is_measured_from_last_completed_interaction() -> None:
    agent = object.__new__(AudioAgent)
    agent._wake_timeout = 30.0
    agent._last_interaction_time = time.monotonic()

    assert agent._followup_window_active() is True

    agent._last_interaction_time -= 31.0
    assert agent._followup_window_active() is False


# ---------------------------------------------------------------------------
# Noise filtering constants
# ---------------------------------------------------------------------------


class TestNoiseFiltering:
    """Verify noise utterance definitions are correctly configured."""

    def test_common_noise_words_in_set(self):
        for word in ["嗯", "哦", "啊", "嗯嗯", "哦哦"]:
            assert word in _NOISE_UTTERANCES

    def test_filler_words_in_noise(self):
        for word in ["那个", "这个", "就是", "然后"]:
            assert word in _NOISE_UTTERANCES

    def test_single_particles_in_noise(self):
        for word in ["的", "了", "吧", "嘛"]:
            assert word in _NOISE_UTTERANCES

    def test_confirmation_words_defined(self):
        for word in ["好", "对", "是", "不", "确认", "取消", "ok", "yes", "no"]:
            assert word in _CONFIRMATION_WORDS

    def test_single_char_commands_defined(self):
        for cmd in ["停", "走", "站", "开", "关"]:
            assert cmd in _SINGLE_CHAR_COMMANDS


class TestNoiseFilterLogic:
    """Test the noise filter logic as implemented in listen_loop."""

    def _is_noise(self, text: str, awaiting_confirmation: bool = False) -> bool:
        """Replicate the noise filter logic from listen_loop."""
        is_confirmation_word = (
            awaiting_confirmation and text in _CONFIRMATION_WORDS
        )
        return (
            not is_confirmation_word
            and (
                text in _NOISE_UTTERANCES
                or text in _CONFIRMATION_WORDS
                or (len(text) == 1 and text not in _SINGLE_CHAR_COMMANDS)
                or (len(text) < _MIN_VALID_TEXT_LEN and text not in _SINGLE_CHAR_COMMANDS)
            )
        )

    def test_noise_utterances_filtered(self):
        for word in ["嗯", "哦哦", "那个", "就是"]:
            assert self._is_noise(word), f"'{word}' should be noise"

    def test_valid_commands_pass(self):
        for cmd in ["导航到仓库", "紧急停止", "帮我搜索天气"]:
            assert not self._is_noise(cmd), f"'{cmd}' should NOT be noise"

    def test_single_char_commands_pass(self):
        for cmd in ["停", "走", "开"]:
            assert not self._is_noise(cmd), f"'{cmd}' should NOT be noise"

    def test_single_char_non_commands_filtered(self):
        for char in ["嘿", "哟", "唉"]:
            assert self._is_noise(char), f"'{char}' should be noise"

    def test_confirmation_words_filtered_when_not_awaiting(self):
        """Confirmation words ARE noise when not awaiting confirmation."""
        for word in ["好", "对", "是的", "确认"]:
            assert self._is_noise(word, awaiting_confirmation=False)

    def test_confirmation_words_pass_when_awaiting(self):
        """Confirmation words pass through when awaiting confirmation."""
        for word in ["好", "对", "是的", "确认", "取消", "ok", "yes"]:
            assert not self._is_noise(word, awaiting_confirmation=True)

    def test_noise_still_filtered_when_awaiting(self):
        """Regular noise is still filtered even when awaiting confirmation."""
        for word in ["嗯", "哦", "那个"]:
            assert self._is_noise(word, awaiting_confirmation=True)


# ---------------------------------------------------------------------------
# Agent state transitions
# ---------------------------------------------------------------------------


@_requires_asr_model
class TestAgentState:
    def test_initial_state_is_idle(self):
        agent = _make_agent()
        try:
            assert agent.state == AgentState.IDLE
        finally:
            agent.shutdown()

    def test_mute_sets_muted_state(self):
        agent = _make_agent()
        try:
            agent.mute()
            assert agent.state == AgentState.MUTED
            assert agent.is_muted is True
        finally:
            agent.shutdown()

    def test_unmute_returns_to_idle(self):
        agent = _make_agent()
        try:
            agent.mute()
            agent.unmute()
            assert agent.state == AgentState.IDLE
            assert agent.is_muted is False
        finally:
            agent.shutdown()

    def test_start_playback_sets_speaking(self):
        agent = _make_agent()
        try:
            agent.start_playback()
            assert agent.state == AgentState.SPEAKING
        finally:
            agent.shutdown()

    def test_stop_playback_returns_to_idle(self):
        agent = _make_agent()
        try:
            agent.start_playback()
            agent.stop_playback()
            assert agent.state == AgentState.IDLE
        finally:
            agent.shutdown()

    def test_agent_state_enum_values(self):
        assert AgentState.IDLE.value == "idle"
        assert AgentState.LISTENING.value == "listening"
        assert AgentState.PROCESSING.value == "processing"
        assert AgentState.SPEAKING.value == "speaking"
        assert AgentState.MUTED.value == "muted"


# ---------------------------------------------------------------------------
# Volume / speed delegation
# ---------------------------------------------------------------------------


@_requires_asr_model
class TestVolumeSpeed:
    def test_set_volume_delegates_to_tts(self):
        agent = _make_agent()
        try:
            result = agent.set_volume(0.7)
            assert abs(result - 0.7) < 0.01
            assert abs(agent.tts.volume - 0.7) < 0.01
        finally:
            agent.shutdown()

    def test_adjust_volume(self):
        agent = _make_agent()
        try:
            agent.set_volume(0.5)
            result = agent.adjust_volume(0.2)
            assert abs(result - 0.7) < 0.01
        finally:
            agent.shutdown()

    def test_set_speed_delegates_to_tts(self):
        agent = _make_agent()
        try:
            result = agent.set_speed(1.5)
            assert result == 1.5
            assert agent.tts.speed == 1.5
        finally:
            agent.shutdown()

    def test_adjust_speed(self):
        agent = _make_agent()
        try:
            agent.set_speed(1.0)
            result = agent.adjust_speed(0.3)
            assert abs(result - 1.3) < 0.01
        finally:
            agent.shutdown()


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


@_requires_asr_model
class TestConvenienceWrappers:
    def test_speak_delegates(self):
        agent = _make_agent()
        try:
            agent.tts.speak = MagicMock()
            agent.speak("test")
            agent.tts.speak.assert_called_once_with("test")
        finally:
            agent.shutdown()

    def test_drain_buffers_delegates(self):
        agent = _make_agent()
        try:
            agent.tts.tts_buffer.append(np.zeros(100, dtype=np.float32))
            agent.drain_buffers()
            assert not agent.tts._has_buffered_audio()
        finally:
            agent.shutdown()

    def test_is_busy_reflects_tts_state(self):
        agent = _make_agent()
        try:
            assert not agent.is_busy  # nothing queued
            agent.speak("hello world test")
            # text is queued — should be busy until worker picks it up
            # (worker may pick it up fast, but queue should be non-empty briefly)
        finally:
            agent.shutdown()


# ---------------------------------------------------------------------------
# Status snapshot / metrics
# ---------------------------------------------------------------------------


@_requires_asr_model
class TestStatusSnapshot:
    def test_text_mode_snapshot(self):
        agent = _make_agent(voice_mode=False)
        try:
            snap = agent.status_snapshot()
            assert snap["mode"] == "text"
            assert snap["enabled"] is False
            assert snap["input_ready"] is False
            assert snap["output_ready"] is True
            assert snap["woken_up"] is True
        finally:
            agent.shutdown()

    def test_muted_reflected_in_snapshot(self):
        agent = _make_agent()
        try:
            agent.mute()
            snap = agent.status_snapshot()
            assert snap["muted"] is True
            assert snap["agent_state"] == "muted"
        finally:
            agent.shutdown()

    def test_tts_backend_in_snapshot(self):
        agent = _make_agent()
        try:
            snap = agent.status_snapshot()
            assert snap["tts_backend"] == "edge"
        finally:
            agent.shutdown()

    def test_full_duplex_decision_is_visible_in_media_status(self):
        agent = _make_agent(voice_mode=False)
        try:
            agent.full_duplex_enabled = False
            agent.full_duplex_status = {
                "enabled": False,
                "reason": "aec_backend_unavailable",
                "echo_control": "none",
                "aec_backend": "unavailable",
            }

            snap = agent.status_snapshot()

            assert snap["media"]["full_duplex"] == agent.full_duplex_status
        finally:
            agent.shutdown()

    def test_half_duplex_speaking_status_requires_waiting_for_playback(self):
        agent = _make_agent(voice_mode=False)
        try:
            agent.voice_mode = True
            agent.asr = object()
            agent.vad = object()
            agent.full_duplex_enabled = False
            agent._agent_state = AgentState.SPEAKING

            interaction = agent.status_snapshot()["interaction"]

            assert interaction["state"] == "speaking"
            assert interaction["can_talk"] is False
            assert interaction["hint"] == "wait_for_playback"
        finally:
            agent.shutdown()

    def test_full_duplex_speaking_status_allows_barge_in(self):
        agent = _make_agent(voice_mode=False)
        try:
            agent.voice_mode = True
            agent.asr = object()
            agent.vad = object()
            agent.full_duplex_enabled = True
            agent._agent_state = AgentState.SPEAKING

            interaction = agent.status_snapshot()["interaction"]

            assert interaction["state"] == "speaking"
            assert interaction["can_talk"] is True
            assert interaction["hint"] == "barge_in_allowed"
        finally:
            agent.shutdown()


# ---------------------------------------------------------------------------
# awaiting_confirmation flag
# ---------------------------------------------------------------------------


@_requires_asr_model
class TestAwaitingConfirmation:
    def test_default_false(self):
        agent = _make_agent()
        try:
            assert agent.awaiting_confirmation is False
        finally:
            agent.shutdown()

    def test_can_be_set(self):
        agent = _make_agent()
        try:
            agent.awaiting_confirmation = True
            assert agent.awaiting_confirmation is True
        finally:
            agent.shutdown()


# ---------------------------------------------------------------------------
# Echo gate configuration
# ---------------------------------------------------------------------------


@_requires_asr_model
class TestEchoGateConfig:
    def test_default_echo_gate_peak(self):
        agent = _make_agent()
        try:
            assert agent._echo_gate_peak == 800
        finally:
            agent.shutdown()

    def test_custom_echo_gate_peak(self):
        agent = _make_agent(echo_gate_peak=500)
        try:
            assert agent._echo_gate_peak == 500
        finally:
            agent.shutdown()

    def test_disabled_echo_gate(self):
        agent = _make_agent(echo_gate_peak=0)
        try:
            assert agent._echo_gate_peak == 0
        finally:
            agent.shutdown()


# ---------------------------------------------------------------------------
# Noise gate configuration
# ---------------------------------------------------------------------------


@_requires_asr_model
class TestNoiseGateConfig:
    def test_default_noise_gate_disabled(self):
        agent = _make_agent()
        try:
            assert agent._noise_gate_peak == 0
        finally:
            agent.shutdown()

    def test_custom_noise_gate(self):
        agent = _make_agent(noise_gate_peak=400)
        try:
            assert agent._noise_gate_peak == 400
        finally:
            agent.shutdown()


# ---------------------------------------------------------------------------
# ASR timeout configuration
# ---------------------------------------------------------------------------


@_requires_asr_model
class TestASRTimeoutConfig:
    def test_default_asr_timeout(self):
        agent = _make_agent()
        try:
            assert agent._asr_timeout == 10.0
        finally:
            agent.shutdown()

    def test_custom_asr_timeout(self):
        agent = _make_agent(asr={"asr_timeout": 15.0})
        try:
            assert agent._asr_timeout == 15.0
        finally:
            agent.shutdown()


# ---------------------------------------------------------------------------
# Input device configuration
# ---------------------------------------------------------------------------


@_requires_asr_model
class TestInputDeviceConfig:
    def test_default_input_device(self):
        agent = _make_agent()
        try:
            assert agent._input_device is None
        finally:
            agent.shutdown()

    def test_int_input_device(self):
        agent = _make_agent(input_device=2)
        try:
            assert agent._input_device == 2
        finally:
            agent.shutdown()

    def test_string_input_device(self):
        agent = _make_agent(input_device="hw:1,0")
        try:
            assert agent._input_device == "hw:1,0"
        finally:
            agent.shutdown()

    def test_numeric_string_input_device(self):
        agent = _make_agent(input_device="3")
        try:
            assert agent._input_device == 3
        finally:
            agent.shutdown()


# ---------------------------------------------------------------------------
# Chime synthesis
# ---------------------------------------------------------------------------


@_requires_asr_model
class TestChimeSynthesis:
    def test_acknowledge_chime_is_valid_audio(self):
        agent = _make_agent()
        try:
            audio = agent._chime_acknowledge()
            assert isinstance(audio, np.ndarray)
            assert audio.dtype == np.float32
            assert len(audio) > 0
        finally:
            agent.shutdown()

    def test_wake_chime_is_valid_audio(self):
        agent = _make_agent()
        try:
            audio = agent._chime_wake()
            assert isinstance(audio, np.ndarray)
            assert len(audio) > 0
        finally:
            agent.shutdown()

    def test_error_chime_is_valid_audio(self):
        agent = _make_agent()
        try:
            audio = agent._chime_error()
            assert isinstance(audio, np.ndarray)
            assert len(audio) > 0
        finally:
            agent.shutdown()

    def test_acknowledge_shorter_than_wake(self):
        """Acknowledge chime should be shorter (2 notes) than wake (3 notes)."""
        agent = _make_agent()
        try:
            ack = agent._chime_acknowledge()
            wake = agent._chime_wake()
            assert len(ack) < len(wake)
        finally:
            agent.shutdown()

    def test_chime_prefers_tts_feedback_path(self, monkeypatch):
        """Sunrise chimes should use TTS feedback output before legacy aplay."""
        import threading

        agent = _make_agent()
        called = threading.Event()

        def fake_feedback(audio, sample_rate):
            called.set()
            assert sample_rate == agent._SR
            assert len(audio) > 0
            return True

        monkeypatch.setattr(agent.tts, "play_feedback_audio", fake_feedback)
        try:
            agent._play_chime("acknowledge")
            assert called.wait(timeout=1.0)
        finally:
            agent.shutdown()

    def test_acknowledge_does_not_disable_the_slow_turn_thinking_chime(
        self, monkeypatch
    ):
        """ACK and long-tail feedback have independent rate-limit clocks."""
        import threading

        agent = _make_agent()
        played: list[int] = []
        called = threading.Event()

        def fake_feedback(audio, sample_rate):
            played.append(len(audio))
            called.set()
            return True

        monkeypatch.setattr(agent.tts, "play_feedback_audio", fake_feedback)
        try:
            agent._play_chime("acknowledge")
            assert called.wait(timeout=1.0)

            called.clear()
            agent._play_chime("thinking")
            assert called.wait(timeout=1.0)

            assert len(played) == 2
        finally:
            agent.shutdown()

    def test_shutdown_stops_active_fallback_feedback(self, monkeypatch):
        import askme.voice.orchestration.audio_agent as audio_agent_module

        agent = _make_agent()
        events: list[str] = []
        process = MagicMock()
        process.poll.return_value = None
        process.terminate.side_effect = lambda: events.append("terminate")
        monkeypatch.setattr(
            agent.tts,
            "cancel_feedback_audio",
            lambda: events.append("provider_cancel"),
        )
        monkeypatch.setattr(
            audio_agent_module.sd,
            "stop",
            lambda: (_ for _ in ()).throw(
                AssertionError("global sounddevice stop must not be used")
            ),
        )
        sounddevice_cancelled = threading.Event()
        agent._feedback_active = True
        agent._feedback_process = process
        agent._feedback_sounddevice_active = True
        agent._feedback_sounddevice_cancel_event = sounddevice_cancelled

        agent.shutdown()

        assert agent._feedback_active is False
        assert sounddevice_cancelled.is_set() is True
        assert events == [
            "provider_cancel",
            "terminate",
        ]


def test_aplay_chime_publishes_reference_after_first_device_write(monkeypatch):
    import subprocess

    agent = _make_agent()
    order: list[str] = []
    delivered = threading.Event()

    class _Stdin:
        def write(self, payload):
            order.append("write")
            return len(payload)

        def flush(self):
            order.append("flush")

    class _Proc:
        stdin = _Stdin()
        returncode = 0

        def communicate(self, input=None, timeout=None):
            del input, timeout
            order.append("communicate")
            return b"", b""

    monkeypatch.setattr(agent.tts, "play_feedback_audio", lambda *_args: False)
    monkeypatch.setattr(agent.tts, "_aplay_bin", "aplay")
    monkeypatch.setattr(
        agent.tts,
        "publish_feedback_render_reference",
        lambda *_args, **_kwargs: (order.append("reference"), delivered.set()),
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: _Proc())
    try:
        agent._play_chime("acknowledge")
        assert delivered.wait(timeout=1.0)
        deadline = time.monotonic() + 1.0
        while "communicate" not in order and time.monotonic() < deadline:
            time.sleep(0.01)
        assert order[:4] == ["write", "flush", "reference", "communicate"]
    finally:
        agent.shutdown()


def test_sounddevice_chime_publishes_reference_after_stream_start(monkeypatch):
    import askme.voice.audio_agent as audio_agent_mod

    agent = _make_agent()
    order: list[str] = []
    delivered = threading.Event()

    monkeypatch.setattr(agent.tts, "play_feedback_audio", lambda *_args: False)
    monkeypatch.setattr(agent.tts, "_aplay_bin", None)
    monkeypatch.setattr(
        agent.tts,
        "publish_feedback_render_reference",
        lambda *_args, **_kwargs: (order.append("reference"), delivered.set()),
    )
    class _OutputStream:
        def __init__(self, *, callback, finished_callback, **_kwargs):
            self.callback = callback
            self.finished_callback = finished_callback

        def start(self):
            order.append("play")
            outdata = np.zeros((agent._SR, 1), dtype=np.float32)
            try:
                self.callback(outdata, len(outdata), None, None)
            except audio_agent_mod.sd.CallbackStop:
                self.finished_callback()

        def abort(self, ignore_errors=True):
            del ignore_errors

        def close(self, ignore_errors=True):
            del ignore_errors
            order.append("wait")

    monkeypatch.setattr(audio_agent_mod.sd, "OutputStream", _OutputStream)
    try:
        agent._play_chime("acknowledge")
        assert delivered.wait(timeout=1.0)
        deadline = time.monotonic() + 1.0
        while "wait" not in order and time.monotonic() < deadline:
            time.sleep(0.01)
        assert order == ["play", "reference", "wait"]
    finally:
        agent.shutdown()


# ---------------------------------------------------------------------------
# Barge-in hold constants
# ---------------------------------------------------------------------------


class TestBargeInConstants:
    def test_barge_in_hold_is_150ms(self):
        assert _BARGE_IN_HOLD_S == 0.15

    def test_max_speech_duration_is_30s(self):
        assert _MAX_SPEECH_DURATION == 30.0

    def test_min_valid_text_len_is_2(self):
        assert _MIN_VALID_TEXT_LEN == 2


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_stop_listening_joins_capture_before_a_clean_restart() -> None:
    class BlockingMic:
        sample_rate = 16_000
        _chunk_samples = 160
        _usb_audio_proc = None

        def __init__(self) -> None:
            self.pre_roll: deque[np.ndarray] = deque()
            self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
            self._lock = threading.Lock()
            self._started = threading.Condition(self._lock)
            self._read_count = 0
            self._active_readers = 0
            self.max_active_readers = 0

        def _flush_queue(self) -> None:
            while True:
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    return

        def open(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read_chunk(self) -> np.ndarray:
            with self._started:
                self._read_count += 1
                self._active_readers += 1
                self.max_active_readers = max(
                    self.max_active_readers,
                    self._active_readers,
                )
                self._started.notify_all()
            try:
                return self._audio_queue.get(timeout=2.0)
            finally:
                with self._lock:
                    self._active_readers -= 1

        def wait_for_read(self, count: int) -> bool:
            deadline = time.monotonic() + 1.0
            with self._started:
                while self._read_count < count:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not self._started.wait(remaining):
                        return False
                return True

    agent = _make_agent(voice_mode=False)
    mic = BlockingMic()
    agent.asr = object()
    agent.vad = object()
    agent.kws = None
    agent._mic = mic
    agent._vad_ctrl = _RecordingVad(VADEvent.SILENCE)
    agent._asr_mgr = _RecordingAsr()
    agent._refresh_voice_metrics = MagicMock()
    agent._apply_pending_runtime_updates = MagicMock()
    results: list[str | None] = []
    threads: list[threading.Thread] = []

    try:
        for listen_number in (1, 2):
            thread = threading.Thread(
                target=lambda: results.append(agent.listen_loop()),
                daemon=True,
            )
            threads.append(thread)
            thread.start()
            assert mic.wait_for_read(listen_number)

            assert agent.stop_listening(timeout=1.0) is True
            thread.join(timeout=0.2)
            assert not thread.is_alive()

        assert results == [None, None]
        assert mic.max_active_readers == 1
    finally:
        agent.stop_event.set()
        for _ in threads:
            try:
                mic._audio_queue.put_nowait(np.zeros(160, dtype=np.float32))
            except queue.Full:
                pass
        for thread in threads:
            thread.join(timeout=1.0)
        agent.shutdown()


def test_stop_listening_aborts_blocked_cloud_finish_before_restart() -> None:
    class FrameMic:
        sample_rate = 16_000
        _chunk_samples = 160
        _usb_audio_proc = None

        def __init__(self) -> None:
            self.pre_roll: deque[np.ndarray] = deque()
            self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        def _flush_queue(self) -> None:
            return None

        def open(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read_chunk(self) -> np.ndarray:
            return np.full(160, 0.2, dtype=np.float32)

    class BlockingFinishAsr(_RecordingAsr):
        def __init__(self) -> None:
            super().__init__()
            self._condition = threading.Condition()
            self._releases: list[threading.Event] = []
            self.finish_calls = 0
            self.active_finishes = 0
            self.max_active_finishes = 0

        def finish_and_get_result(self, awaiting_confirmation=False):
            del awaiting_confirmation
            release = threading.Event()
            with self._condition:
                self._releases.append(release)
                self.finish_calls += 1
                self.active_finishes += 1
                self.max_active_finishes = max(
                    self.max_active_finishes,
                    self.active_finishes,
                )
                self._condition.notify_all()
            release.wait(timeout=5.0)
            with self._condition:
                self.active_finishes -= 1
            return None

        def abort_session(self) -> None:
            with self._condition:
                if self._releases:
                    self._releases[-1].set()

        def wait_for_finish(self, count: int) -> bool:
            deadline = time.monotonic() + 1.0
            with self._condition:
                while self.finish_calls < count:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not self._condition.wait(remaining):
                        return False
                return True

    agent = _make_agent(voice_mode=False)
    asr = BlockingFinishAsr()
    agent.asr = object()
    agent.vad = object()
    agent.kws = None
    agent._mic = FrameMic()
    agent._vad_ctrl = _RecordingVad(VADEvent.SPEECH_END)
    agent._asr_mgr = asr
    agent._refresh_voice_metrics = MagicMock()
    agent._apply_pending_runtime_updates = MagicMock()
    agent.tts.is_active = MagicMock(return_value=False)
    results: list[str | None] = []
    threads: list[threading.Thread] = []

    try:
        for finish_number in (1, 2):
            thread = threading.Thread(
                target=lambda: results.append(agent.listen_loop()),
                daemon=True,
            )
            threads.append(thread)
            thread.start()
            assert asr.wait_for_finish(finish_number)

            assert agent.stop_listening(timeout=0.5) is True
            thread.join(timeout=0.2)
            assert not thread.is_alive()

        assert results == [None, None]
        assert asr.max_active_finishes == 1
    finally:
        agent.stop_event.set()
        for release in asr._releases:
            release.set()
        for thread in threads:
            thread.join(timeout=1.0)
        agent.shutdown()


@_requires_asr_model
class TestLifecycle:
    def test_shutdown_sets_stop_event(self):
        agent = _make_agent()
        agent.shutdown()
        assert agent.stop_event.is_set()

    def test_text_mode_always_woken_up(self):
        """In text mode (no KWS), woken_up defaults to True."""
        agent = _make_agent(voice_mode=False)
        try:
            assert agent.woken_up is True
        finally:
            agent.shutdown()

    def test_speak_error_triggers_metrics(self):
        agent = _make_agent()
        try:
            agent.speak_error()
            agent._metrics.mark_voice_error.assert_called_once()
        finally:
            agent.shutdown()
