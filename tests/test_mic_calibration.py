"""Focused tests for microphone gate calibration and OTA input reports."""

from __future__ import annotations

import json
import queue

import numpy as np
from askme.robot.ota_bridge import OTABridgeMetrics
from askme.robot.runtime_health import merge_voice_pipeline_status
from askme.voice.audio_agent import AudioAgent

from askme.voice import mic_calibration


class _DummyASRManager:
    def __init__(self, _config):
        self._asr = None
        self._stream = None
        self._punct = None


class _DummyVADController:
    def __init__(self, _config):
        self._vad = None


class _DummyTTS:
    def __init__(self, config, *, audio_router=None):
        self.backend = config.get("backend", "dummy")
        self._is_playing = False
        self.tts_text_queue = queue.Queue()

    def is_active(self) -> bool:
        return self._is_playing

    def shutdown(self) -> None:
        self._is_playing = False


def test_low_peak_below_noise_gate_reports_recommendation(monkeypatch) -> None:
    _patch_audio_agent_dependencies(monkeypatch)
    agent = _make_text_agent(noise_gate_peak=500)

    agent._record_input_observation(
        peak=123,
        rms=12.5,
        vad_state="silent",
        gate_state="noise",
    )

    input_status = agent.status_snapshot()["input"]
    assert input_status["gate_recommendation"] == "observed_peak_below_noise_gate:123<500"


def test_calibrated_gate_report_has_json_serializable_input_statistics(monkeypatch) -> None:
    _patch_audio_agent_dependencies(monkeypatch)
    agent = _make_text_agent(noise_gate_peak="auto")
    rng = np.random.RandomState(7)

    for _ in range(20):
        calibrated_gate = agent._audio_proc.auto_calibrate_gate(
            (rng.randn(1600) * 50).astype(np.int16)
        )

    assert calibrated_gate is not None
    agent._record_input_observation(
        peak=max(calibrated_gate - 1, 1),
        rms=9.5,
        vad_state="silent",
        gate_state="noise",
    )
    input_status = agent.status_snapshot()["input"]

    json.loads(json.dumps(input_status))
    assert {
        "run_id",
        "device",
        "transport",
        "sample_rate",
        "native_rate",
        "channels",
        "channel_select",
        "chunk_ms",
        "chunk_samples",
        "mic_open",
        "noise_gate_peak",
        "echo_gate_peak",
        "last_peak",
        "peak_max_10s",
        "last_rms",
        "last_observed_age_s",
        "vad_state",
        "gate_state",
        "tts_active",
        "cooldown_remaining_s",
        "asr_timeouts",
        "last_failure_reason",
        "gate_recommendation",
    } <= set(input_status)
    assert input_status["noise_gate_peak"] == calibrated_gate
    assert input_status["gate_recommendation"] == (
        f"observed_peak_below_noise_gate:{calibrated_gate - 1}<{calibrated_gate}"
    )


def test_voice_pipeline_input_field_is_preserved_through_metrics_merge(monkeypatch) -> None:
    _patch_audio_agent_dependencies(monkeypatch)
    metrics = OTABridgeMetrics()
    agent = _make_text_agent(noise_gate_peak=500, metrics=metrics)
    agent._record_input_observation(
        peak=321,
        rms=18.0,
        vad_state="silent",
        gate_state="noise",
    )

    live_status = agent.status_snapshot()
    metrics_status = metrics.snapshot()["voice_pipeline"]
    merged = merge_voice_pipeline_status(live_status, metrics_status)

    assert merged["input"]["run_id"] == live_status["run_id"]
    assert merged["input"]["last_peak"] == 321
    assert merged["input"]["gate_state"] == "noise"
    assert merged["input"]["gate_recommendation"] == "observed_peak_below_noise_gate:321<500"


def test_runtime_mic_calibration_degrades_when_peak_is_below_gate(monkeypatch) -> None:
    monkeypatch.setattr(
        mic_calibration,
        "_fetch_health",
        lambda _server, *, timeout_s: {
            "voice_pipeline_status": {
                "input": {
                    "run_id": "run-test",
                    "mic_open": True,
                    "transport": "sounddevice",
                    "noise_gate_peak": 80,
                    "last_peak": 5,
                    "peak_max_10s": 5,
                    "peak_p95_10s": 5,
                    "last_rms": 1.0,
                    "rms_p95_10s": 1.0,
                    "gate_state": "noise",
                    "vad_state": "silent",
                    "asr_timeouts": 3,
                    "last_failure_reason": "asr_timeout",
                }
            }
        },
    )

    payload = mic_calibration.collect_runtime_mic_calibration(
        server="http://runtime.local:18765/",
        duration_s=0,
        min_signal_peak=100,
    )

    assert payload["status"] == "degraded"
    assert payload["server"] == "http://runtime.local:18765"
    assert payload["summary"]["observed_peak_max"] == 5
    assert payload["summary"]["recommended_noise_gate_peak"] == 3
    assert payload["summary"]["recommendation"].startswith("input signal is very low")
    assert "observed_peak_below_noise_gate:5<80" in payload["warnings"]


def test_runtime_mic_calibration_reports_ok_when_signal_exceeds_gate(monkeypatch) -> None:
    monkeypatch.setattr(
        mic_calibration,
        "_fetch_health",
        lambda _server, *, timeout_s: {
            "voice_pipeline_status": {
                "input": {
                    "run_id": "run-test",
                    "mic_open": True,
                    "transport": "sounddevice",
                    "noise_gate_peak": 80,
                    "last_peak": 400,
                    "peak_max_10s": 400,
                    "peak_p95_10s": 360,
                    "last_rms": 40.0,
                    "rms_p95_10s": 35.0,
                    "gate_state": "open",
                    "vad_state": "speech",
                    "asr_timeouts": 0,
                }
            }
        },
    )

    payload = mic_calibration.collect_runtime_mic_calibration(
        duration_s=0,
        min_signal_peak=100,
    )

    assert payload["status"] == "ok"
    assert payload["summary"]["observed_peak_max"] == 400
    assert payload["summary"]["recommended_noise_gate_peak"] == 80
    assert payload["warnings"] == []
    json.loads(json.dumps(payload))


def test_runtime_mic_calibration_reports_unreachable_health(monkeypatch) -> None:
    def fail_fetch(_server: str, *, timeout_s: float) -> dict[str, object]:
        raise RuntimeError("runtime offline")

    monkeypatch.setattr(mic_calibration, "_fetch_health", fail_fetch)

    payload = mic_calibration.collect_runtime_mic_calibration(
        duration_s=0,
        min_signal_peak=100,
    )

    assert payload["status"] == "degraded"
    assert payload["sample_count"] == 0
    assert "runtime offline" in payload["errors"]
    assert "mic_not_open" in payload["errors"]
    assert "no_health_samples" in payload["errors"]
    assert payload["summary"]["observed_peak_max"] == 0


def test_runtime_mic_calibration_reports_missing_input_status(monkeypatch) -> None:
    monkeypatch.setattr(
        mic_calibration,
        "_fetch_health",
        lambda _server, *, timeout_s: {"voice_pipeline_status": {}},
    )

    payload = mic_calibration.collect_runtime_mic_calibration(
        duration_s=0,
        min_signal_peak=100,
    )

    assert payload["status"] == "degraded"
    assert payload["sample_count"] == 1
    assert "mic_not_open" in payload["errors"]
    assert payload["summary"]["mic_open"] is False
    assert payload["summary"]["observed_peak_max"] == 0
    assert payload["summary"]["recommendation"].startswith("microphone stream is not open")


def test_runtime_mic_calibration_aggregates_multiple_samples_with_transient_failure(
    monkeypatch,
) -> None:
    now = [1000.0]
    responses: list[dict[str, object] | Exception] = [
        _health_with_input(last_peak=120, peak_max=120),
        RuntimeError("temporary health timeout"),
        _health_with_input(last_peak=260, peak_max=260),
        _health_with_input(last_peak=180, peak_max=400),
        _health_with_input(last_peak=320, peak_max=400),
    ]

    def fake_time() -> float:
        return now[0]

    def fake_sleep(seconds: float) -> None:
        now[0] += max(0.0, seconds)

    def fetch_next(_server: str, *, timeout_s: float) -> dict[str, object]:
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(mic_calibration.time, "time", fake_time)
    monkeypatch.setattr(mic_calibration.time, "sleep", fake_sleep)
    monkeypatch.setattr(mic_calibration, "_fetch_health", fetch_next)

    payload = mic_calibration.collect_runtime_mic_calibration(
        duration_s=1.0,
        interval_s=0.25,
        min_signal_peak=100,
    )

    assert payload["status"] == "degraded"
    assert payload["sample_count"] == 4
    assert payload["summary"]["observed_peak_max"] == 400
    assert payload["summary"]["observed_peak_p95"] == 400.0
    assert payload["summary"]["recommendation"] == "input signal is above the configured gate"
    assert payload["errors"] == ["temporary health timeout"]
    assert payload["warnings"] == []


def _health_with_input(*, last_peak: int, peak_max: int) -> dict[str, object]:
    return {
        "voice_pipeline_status": {
            "input": {
                "run_id": "run-test",
                "mic_open": True,
                "transport": "sounddevice",
                "noise_gate_peak": 80,
                "last_peak": last_peak,
                "peak_max_10s": peak_max,
                "peak_p95_10s": peak_max,
                "last_rms": 40.0,
                "rms_p95_10s": 35.0,
                "gate_state": "open",
                "vad_state": "speech",
                "asr_timeouts": 0,
            }
        }
    }


def _patch_audio_agent_dependencies(monkeypatch) -> None:
    monkeypatch.setattr("askme.voice.audio_agent.ASRManager", _DummyASRManager)
    monkeypatch.setattr("askme.voice.audio_agent.VADController", _DummyVADController)
    monkeypatch.setattr("askme.voice.audio_agent.TTSEngine", _DummyTTS)


def _make_text_agent(
    *,
    noise_gate_peak: int | str,
    metrics: OTABridgeMetrics | None = None,
) -> AudioAgent:
    return AudioAgent(
        {
            "voice": {
                "tts": {"backend": "edge"},
                "input_device": "hw:1,0",
                "mic_native_rate": 48000,
                "mic_channels": 2,
                "mic_channel_select": 1,
                "noise_gate_peak": noise_gate_peak,
                "echo_gate_peak": 30000,
            }
        },
        voice_mode=False,
        metrics=metrics or OTABridgeMetrics(),
    )
