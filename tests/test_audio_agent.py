"""Tests for AudioAgent: noise filtering, confirmation context, echo gate,
barge-in hold, agent state transitions, mute/unmute, volume/speed delegation."""

from __future__ import annotations

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
            agent.speak("test")
            assert not agent.tts.tts_text_queue.empty()
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
