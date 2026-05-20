from __future__ import annotations

import queue
from pathlib import Path

import pytest
from askme.robot.ota_bridge import OTABridgeMetrics
from askme.voice.audio_agent import AudioAgent

_ASR_MODEL_TOKENS = Path(
    "models/asr/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt"
)
pytestmark = pytest.mark.skipif(
    not _ASR_MODEL_TOKENS.exists(),
    reason=f"ASR model not present at {_ASR_MODEL_TOKENS}",
)


class _DummyTTS:
    def __init__(self, config, *, audio_router=None):
        self.backend = config.get("backend", "dummy")
        self._is_playing = False
        self.tts_text_queue = queue.Queue()

    def speak(self, text: str) -> None:
        self.tts_text_queue.put(text)

    def start_playback(self) -> None:
        self._is_playing = True

    def stop_playback(self) -> None:
        self._is_playing = False

    def wait_done(self) -> None:
        self._is_playing = False

    def drain_buffers(self) -> None:
        while not self.tts_text_queue.empty():
            self.tts_text_queue.get_nowait()
        self._is_playing = False

    def status_snapshot(self) -> dict:
        return {
            "backend": self.backend,
            "is_playing": self._is_playing,
            "queue_size": self.tts_text_queue.qsize(),
        }

    def shutdown(self) -> None:
        self._is_playing = False


def test_audio_agent_updates_ota_metrics_snapshot(monkeypatch) -> None:
    monkeypatch.setattr("askme.voice.audio_agent.TTSEngine", _DummyTTS)

    metrics = OTABridgeMetrics()
    agent = AudioAgent(
        {"voice": {"tts": {"backend": "edge"}}},
        voice_mode=False,
        metrics=metrics,
    )

    snapshot = agent.status_snapshot()
    voice_metrics = metrics.snapshot()["voice_pipeline"]

    assert snapshot["mode"] == "text"
    assert snapshot["interaction"]["state"] == "text_mode"
    assert snapshot["interaction"]["can_talk"] is False
    assert snapshot["pipeline_ok"] is True
    assert voice_metrics["mode"] == "text"
    assert voice_metrics["output_ready"] is True
    assert voice_metrics["tts_backend"] == "edge"


def test_audio_agent_status_snapshot_includes_input_diagnostics(monkeypatch) -> None:
    monkeypatch.setattr("askme.voice.audio_agent.TTSEngine", _DummyTTS)

    metrics = OTABridgeMetrics()
    agent = AudioAgent(
        {
            "voice": {
                "tts": {"backend": "edge"},
                "input_device": "hw:1,0",
                "mic_native_rate": 48000,
                "mic_channels": 2,
                "mic_channel_select": 1,
                "noise_gate_peak": 500,
                "echo_gate_peak": 30000,
            }
        },
        voice_mode=False,
        metrics=metrics,
    )

    agent._record_input_observation(
        peak=123,
        rms=12.5,
        vad_state="silent",
        gate_state="noise",
    )
    snapshot = agent.status_snapshot()
    input_status = snapshot["input"]
    media_status = snapshot["media"]
    voice_turn_status = snapshot["voice_turn"]
    interaction_status = snapshot["interaction"]

    assert snapshot["run_id"]
    assert interaction_status["state"] == "text_mode"
    assert interaction_status["hint"] == "use_text_input"
    assert media_status["media_transport"] == "local_sounddevice"
    assert media_status["session_id"] == snapshot["run_id"]
    assert media_status["participant_count"] == 0
    assert voice_turn_status["counters"]["barge_in_count"] == 0
    assert voice_turn_status["current"] is None
    assert input_status["run_id"] == snapshot["run_id"]
    assert input_status["device"] == "hw:1,0"
    assert input_status["native_rate"] == 48000
    assert input_status["channels"] == 2
    assert input_status["channel_select"] == 1
    assert input_status["noise_gate_peak"] == 500
    assert input_status["echo_gate_peak"] == 30000
    assert input_status["last_peak"] == 123
    assert input_status["last_rms"] == 12.5
    assert input_status["vad_state"] == "silent"
    assert input_status["gate_state"] == "noise"
    assert input_status["gate_recommendation"] == "observed_peak_below_noise_gate:123<500"


def test_audio_agent_status_snapshot_reports_silent_microphone(monkeypatch) -> None:
    monkeypatch.setattr("askme.voice.audio_agent.TTSEngine", _DummyTTS)

    agent = AudioAgent(
        {"voice": {"tts": {"backend": "edge"}, "noise_gate_peak": 500}},
        voice_mode=False,
        metrics=OTABridgeMetrics(),
    )

    agent._record_input_observation(
        peak=0,
        rms=0.0,
        vad_state="silent",
        gate_state="noise",
    )

    input_status = agent.status_snapshot()["input"]

    assert input_status["peak_max_10s"] == 0
    assert input_status["gate_recommendation"] == (
        "microphone_captured_silence:check_input_device_permission_or_physical_mute"
    )


def test_audio_agent_marks_voice_error_for_ota_metrics(monkeypatch) -> None:
    monkeypatch.setattr("askme.voice.audio_agent.TTSEngine", _DummyTTS)
    monkeypatch.setattr("askme.voice.audio_agent.sd.play", lambda *args, **kwargs: None)

    metrics = OTABridgeMetrics()
    agent = AudioAgent(
        {"voice": {"tts": {"backend": "edge"}}},
        voice_mode=False,
        metrics=metrics,
    )

    agent.speak_error()

    voice_metrics = metrics.snapshot()["voice_pipeline"]
    assert voice_metrics["last_error"] == "voice interaction error"
    assert voice_metrics["tts_busy"] is True
