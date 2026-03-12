from __future__ import annotations

import queue

from askme.ota_bridge import OTABridgeMetrics
from askme.voice.audio_agent import AudioAgent


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
    assert snapshot["pipeline_ok"] is True
    assert voice_metrics["mode"] == "text"
    assert voice_metrics["output_ready"] is True
    assert voice_metrics["tts_backend"] == "edge"


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
