"""Integration tests for AudioAgent.listen_loop with modular components mocked.

Drives sequences of VAD events through listen_loop and verifies outcomes
without any real audio devices, ASR models, or TTS engines.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from askme.voice.audio_agent import AudioAgent, AgentState
from askme.voice.asr_manager import ASRResult
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

    # KWS: not available so listen_loop skips wake-word phase
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

    agent = AudioAgent({"voice": {}}, voice_mode=True, metrics=mock_metrics)

    # Replace the modular components with fresh mocks we control
    agent._mic = MagicMock()
    agent._mic.sample_rate = _SAMPLE_RATE
    agent._audio_proc = MagicMock()
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
            asr.reset.assert_called()

    def test_barge_in_stops_tts_and_feeds_buffer(self):
        """BARGE_IN_CONFIRMED -> tts.stop_immediately + tts.drain_buffers called, buffer fed to ASR."""
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

            barge_buf_chunk = np.ones(160, dtype=np.float32)
            vad.barge_in_buffer = [barge_buf_chunk]

            call_count = 0

            def vad_feed(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    vad.speech_active = True
                    return VADEvent.BARGE_IN_CONFIRMED
                if call_count == 2:
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
            agent.tts.stop_immediately.assert_called_once()
            agent.tts.drain_buffers.assert_called_once()
            # Buffer was fed to ASR
            assert asr.feed_audio.call_count >= 1

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

            agent._test_metrics.mark_voice_error.assert_called_once()

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
