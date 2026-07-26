"""Integration tests for AudioAgent.listen_loop with modular components mocked.

Drives sequences of VAD events through listen_loop and verifies outcomes
without any real audio devices, ASR models, or TTS engines.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from askme.voice.asr_manager import ASRResult
from askme.voice.audio_agent import AudioAgent
from askme.voice.vad_controller import VADEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16000
_CHUNK = np.zeros(160, dtype=np.float32)
_CHUNK_I16 = np.zeros(160, dtype=np.int16)


def _make_agent() -> AudioAgent:
    """Create an AudioAgent with all heavy engines mocked out.

    Patches ASREngine, VADEngine, KWSEngine, PunctuationRestorer,
    TTSEngine, MicInput, AudioProcessor, VADController, ASRManager
    so no real models or devices are loaded.
    """
    patches = {
        "kws_engine": patch("askme.voice.audio_agent.KWSEngine"),
        "tts_engine": patch("askme.voice.audio_agent.TTSEngine"),
        "mic_cls": patch("askme.voice.audio_agent.MicInput"),
        "audio_proc_cls": patch("askme.voice.audio_agent.AudioProcessor"),
        "vad_ctrl_cls": patch("askme.voice.audio_agent.VADController"),
        "asr_mgr_cls": patch("askme.voice.audio_agent.ASRManager"),
        "sd": patch("askme.voice.audio_agent.sd"),
    }

    started = {k: p.start() for k, p in patches.items()}

    # These tests exercise the generic listen/VAD/ASR interface. Keep KWS
    # unavailable and explicitly opt this fixture out of the product
    # fail-closed wake-word policy; wake-word behaviour is covered separately.
    mock_kws_inst = started["kws_engine"].return_value
    mock_kws_inst.available = False

    # TTS mock
    mock_tts = started["tts_engine"].return_value
    mock_tts._is_playing = False
    mock_tts.tts_text_queue = MagicMock()
    mock_tts.tts_text_queue.empty.return_value = True
    mock_tts.is_active.return_value = False
    mock_tts.backend = "mock"
    mock_tts.volume = 1.0
    mock_tts.speed = 1.0

    # Metrics mock
    mock_metrics = MagicMock()

    agent = AudioAgent(
        {"voice": {"product_readiness": {"require_wake_word": False}}},
        voice_mode=True,
        metrics=mock_metrics,
    )

    # Replace the modular components with fresh mocks we control
    agent._mic = MagicMock()
    agent._mic.sample_rate = _SAMPLE_RATE
    agent._audio_proc = MagicMock()
    agent._audio_proc.is_noise_gated.return_value = False
    agent._vad_ctrl = MagicMock()
    agent._vad_ctrl.speech_active = False
    agent._vad_ctrl.barge_in_buffer = []
    agent._asr_mgr = MagicMock()

    # Store for test access
    agent._test_patches = patches  # type: ignore[attr-defined]
    agent._test_metrics = mock_metrics  # type: ignore[attr-defined]

    return agent


def _teardown(agent: AudioAgent) -> None:
    """Stop all patches from _make_agent."""
    for p in agent._test_patches.values():  # type: ignore[attr-defined]
        p.stop()


@contextmanager
def _agent_ctx() -> Generator[AudioAgent, None, None]:
    """Context manager that creates and tears down a test agent."""
    agent = _make_agent()
    try:
        yield agent
    finally:
        _teardown(agent)


def _setup_mic_open(agent: AudioAgent) -> MagicMock:
    """Configure mic.open() to yield itself as a context manager."""
    mic = agent._mic
    mic_ctx = mic  # the mic IS the context in our mock
    mic.open.return_value.__enter__ = MagicMock(return_value=mic_ctx)
    mic.open.return_value.__exit__ = MagicMock(return_value=False)
    mic.flush_pre_roll.return_value = []
    mic.pre_roll = MagicMock()
    return mic


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestListenLoopIntegration:
    """Integration tests that drive AudioAgent.listen_loop through event sequences."""

    def test_happy_path_speech_returns_text(self):
        """SILENCE -> SPEECH_START -> SPEECH_CONTINUE -> SPEECH_END -> returns text."""
        with _agent_ctx() as agent:
            mic = _setup_mic_open(agent)
            proc = agent._audio_proc
            vad = agent._vad_ctrl
            asr = agent._asr_mgr

            # mic.read_chunk returns a chunk each call
            mic.read_chunk.return_value = _CHUNK

            # AudioProcessor: no echo gating
            proc.process.return_value = (_CHUNK, _CHUNK_I16, 2000, False)

            # VAD event sequence: SILENCE, SPEECH_START, SPEECH_CONTINUE, SPEECH_END
            call_count = 0
            events = [
                VADEvent.SILENCE,
                VADEvent.SPEECH_START,
                VADEvent.SPEECH_CONTINUE,
                VADEvent.SPEECH_END,
            ]

            def vad_feed(*args, **kwargs):
                nonlocal call_count
                idx = min(call_count, len(events) - 1)
                evt = events[idx]
                call_count += 1
                # Update speech_active state to match
                if evt == VADEvent.SPEECH_START:
                    vad.speech_active = True
                elif evt in (VADEvent.SPEECH_END, VADEvent.MAX_DURATION_EXCEEDED):
                    vad.speech_active = False
                return evt

            vad.feed.side_effect = vad_feed

            # ASR: check_endpoint returns None, finish returns valid text
            asr.check_endpoint.return_value = None
            asr.finish_and_get_result.return_value = ASRResult(
                text="你好世界", source="local", is_noise=False,
            )

            result = agent.listen_loop()

            assert result == "你好世界"
            voice_turn = agent.status_snapshot()["voice_turn"]
            latest = voice_turn["latest"]
            stages = {stage["name"]: stage for stage in latest["stages"]}
            assert latest["status"] == "accepted"
            assert latest["media_transport"] == "local_sounddevice"
            assert stages["first_audio_frame"]
            assert stages["vad_start"]["metadata"]["peak"] == 2000
            assert stages["vad_end"]["metadata"]["peak"] == 2000
            assert stages["asr_final"]["metadata"]["asr_source"] == "local"
            asr.start_session.assert_called_once()
            asr.finish_and_get_result.assert_called_once()
            agent._test_metrics.mark_voice_listen_started.assert_called_once()

    def test_timeout_returns_none(self):
        """No speech within _asr_timeout -> returns None."""
        with _agent_ctx() as agent:
            mic = _setup_mic_open(agent)
            proc = agent._audio_proc
            vad = agent._vad_ctrl
            asr = agent._asr_mgr

            agent._asr_timeout = 0.05  # 50ms timeout for fast test

            mic.read_chunk.return_value = _CHUNK
            proc.process.return_value = (_CHUNK, _CHUNK_I16, 50, False)
            vad.feed.return_value = VADEvent.SILENCE

            # check_endpoint always None (no speech)
            asr.check_endpoint.return_value = None

            result = agent.listen_loop()

            assert result is None
            voice_turn = agent.status_snapshot()["voice_turn"]
            assert voice_turn["latest"]["status"] == "timeout"
            asr.reset.assert_called()

    def test_wake_prompt_does_not_spawn_raw_aplay_beep(self, monkeypatch):
        """The shared chime path must handle Windows without a raw aplay call."""
        import subprocess
        import threading

        with _agent_ctx() as agent:
            mic = _setup_mic_open(agent)
            proc = agent._audio_proc
            vad = agent._vad_ctrl
            asr = agent._asr_mgr
            raw_aplay_called = threading.Event()

            agent.kws.available = True
            agent.kws_stream = object()
            agent._wait_for_wake_word_mic = MagicMock(return_value=True)
            agent._play_chime = MagicMock()
            agent._asr_timeout = 0.03

            mic.read_chunk.return_value = _CHUNK
            proc.process.return_value = (_CHUNK, _CHUNK_I16, 50, False)
            vad.feed.return_value = VADEvent.SILENCE
            asr.check_endpoint.return_value = None

            def fake_run(*_args, **_kwargs):
                raw_aplay_called.set()
                return MagicMock(returncode=0)

            monkeypatch.setattr(subprocess, "run", fake_run)

            assert agent.listen_loop() is None
            assert not raw_aplay_called.wait(timeout=0.1)
            agent._play_chime.assert_called_once_with("wake")

    def test_barge_in_cancels_only_after_asr_accepts_the_interruption(self):
        """Detection holds playback; accepted ASR then cancels and drains it."""
        with _agent_ctx() as agent:
            mic = _setup_mic_open(agent)
            proc = agent._audio_proc
            vad = agent._vad_ctrl
            asr = agent._asr_mgr

            mic.read_chunk.return_value = _CHUNK
            mic.flush_pre_roll.return_value = []
            proc.process.return_value = (_CHUNK, _CHUNK_I16, 3000, False)

            # Simulate TTS active
            agent.tts.is_active.return_value = True
            hold_token = object()
            lifecycle: list[str] = []
            agent.tts.pause_playback.side_effect = (
                lambda **_kwargs: lifecycle.append("pause") or hold_token
            )
            agent.tts.abort_playback_hold.side_effect = (
                lambda token: lifecycle.append("abort") or token is hold_token
            )
            agent.tts.stop_immediately.side_effect = lambda: lifecycle.append("stop")
            agent.tts.drain_buffers.side_effect = lambda: lifecycle.append("drain")
            agent.set_barge_in_callback(lambda: lifecycle.append("callback"))
            output_turn = agent._turn_traces.start(
                source="test_output",
                media_transport="sounddevice",
            )
            agent._turn_traces.finish("accepted")
            playback_token = agent.start_playback(
                voice_turn_id=output_turn.voice_turn_id,
            )
            assert playback_token is not None

            barge_buf_chunk = np.ones(160, dtype=np.float32)
            vad.barge_in_buffer = [barge_buf_chunk]

            call_count = 0

            def vad_feed(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return VADEvent.BARGE_IN_START
                if call_count == 2:
                    vad.speech_active = True
                    return VADEvent.BARGE_IN_CONFIRMED
                if call_count == 3:
                    vad.speech_active = False
                    return VADEvent.SPEECH_END
                return VADEvent.SILENCE

            vad.feed.side_effect = vad_feed

            asr.check_endpoint.return_value = None
            asr.finish_and_get_result.return_value = ASRResult(
                text="停下来", source="local", is_noise=False,
            )

            result = agent.listen_loop()

            assert result == "停下来"
            voice_turn = agent.status_snapshot()["voice_turn"]
            latest = voice_turn["latest"]
            stages = {stage["name"]: stage for stage in latest["stages"]}
            # Output facts stay on the frozen speaker owner and never attach
            # to the concurrently captured user turn.
            assert voice_turn["counters"]["barge_in_count"] == 1
            assert voice_turn["counters"]["orphan_output_event_count"] == 0
            assert "barge_in_confirmed" not in stages
            assert lifecycle == ["pause", "callback", "abort", "stop", "drain"]
            agent.tts.stop_immediately.assert_called_once()
            agent.tts.drain_buffers.assert_called_once()
            # Buffer was fed to ASR
            assert asr.feed_audio.call_count >= 1

    def test_false_barge_in_resumes_playback_without_pipeline_cancellation(self):
        """Noise ASR after a confirmed candidate resumes the exact held output."""
        with _agent_ctx() as agent:
            mic = _setup_mic_open(agent)
            proc = agent._audio_proc
            vad = agent._vad_ctrl
            asr = agent._asr_mgr

            mic.read_chunk.return_value = _CHUNK
            mic.flush_pre_roll.return_value = []
            proc.process.return_value = (_CHUNK, _CHUNK_I16, 3000, False)
            agent.tts.is_active.return_value = True
            hold_token = object()
            lifecycle: list[str] = []
            agent.tts.pause_playback.side_effect = (
                lambda **_kwargs: lifecycle.append("pause") or hold_token
            )
            agent.tts.resume_playback.side_effect = (
                lambda token: lifecycle.append("resume") or token is hold_token
            )
            agent.tts.abort_playback_hold.side_effect = (
                lambda _token: lifecycle.append("abort") or True
            )
            agent.tts.stop_immediately.side_effect = lambda: lifecycle.append("stop")
            agent.tts.drain_buffers.side_effect = lambda: lifecycle.append("drain")
            agent.set_barge_in_callback(lambda: lifecycle.append("callback"))
            output_turn = agent._turn_traces.start(
                source="test_output",
                media_transport="sounddevice",
            )
            agent._turn_traces.finish("accepted")
            playback_token = agent.start_playback(
                voice_turn_id=output_turn.voice_turn_id,
            )
            assert playback_token is not None
            vad.barge_in_buffer = [np.ones(160, dtype=np.float32)]

            events = [
                VADEvent.BARGE_IN_START,
                VADEvent.BARGE_IN_CONFIRMED,
                VADEvent.SPEECH_END,
                VADEvent.SPEECH_START,
                VADEvent.SPEECH_END,
            ]
            call_count = 0

            def vad_feed(*_args, **_kwargs):
                nonlocal call_count
                event = events[min(call_count, len(events) - 1)]
                call_count += 1
                vad.speech_active = event in {
                    VADEvent.BARGE_IN_CONFIRMED,
                    VADEvent.SPEECH_START,
                }
                return event

            vad.feed.side_effect = vad_feed
            asr.check_endpoint.return_value = None
            asr.finish_and_get_result.side_effect = [
                ASRResult(text="嗯", source="local", is_noise=True),
                ASRResult(text="继续巡检", source="local", is_noise=False),
            ]

            assert agent.listen_loop() == "继续巡检"
            assert lifecycle == ["pause", "resume"]
            agent.tts.stop_immediately.assert_not_called()
            agent.tts.drain_buffers.assert_not_called()

    def test_noise_result_resets_and_continues(self):
        """SPEECH_END with noise result -> asr.reset + vad.reset, loop continues."""
        with _agent_ctx() as agent:
            mic = _setup_mic_open(agent)
            proc = agent._audio_proc
            vad = agent._vad_ctrl
            asr = agent._asr_mgr

            mic.read_chunk.return_value = _CHUNK
            proc.process.return_value = (_CHUNK, _CHUNK_I16, 2000, False)

            call_count = 0

            def vad_feed(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    vad.speech_active = True
                    return VADEvent.SPEECH_START
                if call_count == 2:
                    vad.speech_active = False
                    return VADEvent.SPEECH_END
                # Second round: speech -> valid result
                if call_count == 3:
                    vad.speech_active = True
                    return VADEvent.SPEECH_START
                if call_count == 4:
                    vad.speech_active = False
                    return VADEvent.SPEECH_END
                return VADEvent.SILENCE

            vad.feed.side_effect = vad_feed

            asr.check_endpoint.return_value = None

            finish_call = 0

            def finish_side_effect(*args, **kwargs):
                nonlocal finish_call
                finish_call += 1
                if finish_call == 1:
                    # First: noise
                    return ASRResult(text="嗯", source="local", is_noise=True)
                # Second: valid
                return ASRResult(text="导航到仓库", source="local", is_noise=False)

            asr.finish_and_get_result.side_effect = finish_side_effect

            result = agent.listen_loop()

            assert result == "导航到仓库"
            # After noise, reset was called
            assert asr.reset.call_count >= 1
            assert vad.reset.call_count >= 1

    def test_max_duration_forced_noise_continues(self):
        """MAX_DURATION_EXCEEDED + force_endpoint returns None (noise) -> loop continues."""
        with _agent_ctx() as agent:
            mic = _setup_mic_open(agent)
            proc = agent._audio_proc
            vad = agent._vad_ctrl
            asr = agent._asr_mgr

            agent._asr_timeout = 2.0  # generous timeout

            mic.read_chunk.return_value = _CHUNK
            proc.process.return_value = (_CHUNK, _CHUNK_I16, 2000, False)

            call_count = 0

            def vad_feed(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    vad.speech_active = True
                    return VADEvent.SPEECH_START
                if call_count == 2:
                    vad.speech_active = False
                    return VADEvent.MAX_DURATION_EXCEEDED
                # After forced endpoint fails, next round produces valid speech
                if call_count == 3:
                    vad.speech_active = True
                    return VADEvent.SPEECH_START
                if call_count == 4:
                    vad.speech_active = False
                    return VADEvent.SPEECH_END
                return VADEvent.SILENCE

            vad.feed.side_effect = vad_feed

            asr.check_endpoint.return_value = None

            # force_endpoint returns None (too short / noise)
            asr.force_endpoint.return_value = None

            asr.finish_and_get_result.return_value = ASRResult(
                text="开始巡检", source="local", is_noise=False,
            )

            result = agent.listen_loop()

            assert result == "开始巡检"
            asr.force_endpoint.assert_called_once()

    def test_exception_marks_metrics(self):
        """mic.open() raises -> mark_voice_error called, exception re-raised."""
        with _agent_ctx() as agent:
            mic = agent._mic
            # Make mic.open() raise an exception
            mic.open.return_value.__enter__ = MagicMock(
                side_effect=RuntimeError("device not found")
            )
            mic.open.return_value.__exit__ = MagicMock(return_value=False)

            with pytest.raises(RuntimeError, match="device not found"):
                agent.listen_loop()

            agent._test_metrics.mark_voice_error.assert_called_once_with(
                "RuntimeError"
            )
            assert (
                agent.status_snapshot()["input"]["last_failure_reason"]
                == "asr_error"
            )

    def test_echo_gated_frame_skipped(self):
        """When AudioProcessor returns echo_gated=True, frame is skipped (no VAD feed)."""
        with _agent_ctx() as agent:
            mic = _setup_mic_open(agent)
            proc = agent._audio_proc
            vad = agent._vad_ctrl
            asr = agent._asr_mgr

            agent._asr_timeout = 0.1

            mic.read_chunk.return_value = _CHUNK

            call_count = 0

            def proc_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 3:
                    # First 3 frames: echo gated
                    return (_CHUNK, _CHUNK_I16, 200, True)
                # After that, return non-gated silence (will timeout)
                return (_CHUNK, _CHUNK_I16, 50, False)

            proc.process.side_effect = proc_side_effect
            vad.feed.return_value = VADEvent.SILENCE
            asr.check_endpoint.return_value = None

            result = agent.listen_loop()

            assert result is None
            # VAD should not have been called during the echo-gated frames
            # It should only be called for the non-gated frames
            for call_args in vad.feed.call_args_list:
                # All vad.feed calls happened after echo gating stopped
                pass
            # The echo-gated frames should have triggered buffer_pre_roll
            assert mic.buffer_pre_roll.call_count >= 1

    def test_stop_playback_starts_input_cooldown_and_resets_state(self):
        """Normal playback stop resets listen state and starts the post-TTS cooldown."""
        with _agent_ctx() as agent:
            agent._post_tts_input_cooldown_s = 1.0
            agent.tts.start_playback = MagicMock()
            token = agent.start_playback()
            assert token is not None

            before = time.monotonic()
            agent.stop_playback(token)

            assert agent._input_cooldown_until >= before + 0.9
            agent._vad_ctrl.reset.assert_called_once()
            agent._asr_mgr.reset.assert_called_once()

    def test_post_tts_cooldown_discards_frames_without_preroll(self):
        """Cooldown frames are thrown away instead of becoming ASR pre-roll."""
        with _agent_ctx() as agent:
            mic = _setup_mic_open(agent)
            proc = agent._audio_proc
            vad = agent._vad_ctrl
            asr = agent._asr_mgr

            agent._asr_timeout = 0.01
            agent._input_cooldown_until = time.monotonic() + 1.0
            mic.read_chunk.return_value = _CHUNK
            asr.check_endpoint.return_value = None

            result = agent.listen_loop()

            assert result is None
            proc.process.assert_not_called()
            vad.feed.assert_not_called()
            mic.buffer_pre_roll.assert_not_called()
            assert mic.pre_roll.clear.call_count >= 1

    def test_check_endpoint_returns_valid_text(self):
        """Local ASR endpoint detected mid-speech returns text immediately."""
        with _agent_ctx() as agent:
            mic = _setup_mic_open(agent)
            proc = agent._audio_proc
            vad = agent._vad_ctrl
            asr = agent._asr_mgr

            mic.read_chunk.return_value = _CHUNK
            proc.process.return_value = (_CHUNK, _CHUNK_I16, 2000, False)

            call_count = 0

            def vad_feed(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    vad.speech_active = True
                    return VADEvent.SPEECH_START
                return VADEvent.SPEECH_CONTINUE

            vad.feed.side_effect = vad_feed

            ep_call = 0

            def check_ep_side_effect():
                nonlocal ep_call
                ep_call += 1
                if ep_call == 2:
                    return ASRResult(text="你好", source="local")
                return None

            asr.check_endpoint.side_effect = check_ep_side_effect
            # is_noise returns False for valid text
            asr.is_noise.return_value = False

            result = agent.listen_loop()

            assert result == "你好"
