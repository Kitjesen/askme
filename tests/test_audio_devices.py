from __future__ import annotations

import numpy as np

from askme.voice import audio_devices


class _Default:
    device = [12, 11]


class _FakeSoundDevice:
    default = _Default()
    last_playrec: dict = {}

    @staticmethod
    def query_devices(device=None):
        devices = [
            {
                "name": "Mic",
                "hostapi": 0,
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
            {
                "name": "Speaker",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            },
        ]
        if device is None:
            return devices
        return devices[int(device)]

    @staticmethod
    def query_hostapis():
        return [
            {
                "name": "Windows WASAPI",
                "devices": [0, 1],
                "default_input_device": 0,
                "default_output_device": 1,
            }
        ]

    @classmethod
    def playrec(cls, out, **kwargs):
        cls.last_playrec = kwargs
        channels = int(kwargs.get("channels") or 1)
        source = np.asarray(out, dtype=np.float32)
        mono = source[:, 0] if source.ndim == 2 else source.reshape(-1)
        captured = np.zeros((len(mono), channels), dtype=np.float32)
        target_channel = min(1, channels - 1)
        captured[:, target_channel] = mono * 0.8
        return captured

    @staticmethod
    def play(_samples, **_kwargs):
        return None

    @staticmethod
    def wait():
        return None


def test_query_audio_devices_recommends_wasapi_defaults(monkeypatch):
    monkeypatch.setattr(audio_devices, "sd", _FakeSoundDevice)

    payload = audio_devices.query_audio_devices()

    assert payload["status"] == "ok"
    assert payload["recommendation"]["input_device"] == 0
    assert payload["recommendation"]["output_device"] == 1
    assert payload["recommendation"]["output_transport"] == "sounddevice"


def test_audio_loopback_uses_recommended_devices(monkeypatch, tmp_path):
    monkeypatch.setattr(audio_devices, "sd", _FakeSoundDevice)
    wav_out = tmp_path / "loop.wav"

    payload = audio_devices.run_audio_loopback(
        record_seconds=0.4,
        tone_seconds=0.2,
        output_gain=0.4,
        wav_out=wav_out,
    )

    assert payload["status"] == "ok"
    assert payload["input_device"] == 0
    assert payload["output_device"] == 1
    assert payload["signal_ok"] is True
    assert payload["selected_input_channel"] == 0
    assert wav_out.exists()
    assert _FakeSoundDevice.last_playrec["device"] == (0, 1)


class _MultiChannelSoundDevice(_FakeSoundDevice):
    @staticmethod
    def query_devices(device=None):
        devices = [
            {
                "name": "WASAPI Mic",
                "hostapi": 0,
                "max_input_channels": 4,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
            {
                "name": "WASAPI Speaker",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            },
            {
                "name": "MME Mic",
                "hostapi": 1,
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 44100.0,
            },
        ]
        if device is None:
            return devices
        return devices[int(device)]

    @staticmethod
    def query_hostapis():
        return [
            {
                "name": "Windows WASAPI",
                "devices": [0, 1],
                "default_input_device": 0,
                "default_output_device": 1,
            },
            {
                "name": "MME",
                "devices": [2],
                "default_input_device": 2,
                "default_output_device": -1,
            },
        ]


def test_audio_route_scan_reports_best_channel(monkeypatch):
    monkeypatch.setattr(audio_devices, "sd", _MultiChannelSoundDevice)

    payload = audio_devices.run_audio_route_scan(
        input_devices=[0],
        output_devices=[1],
        sample_rates=[48000],
        record_seconds=0.4,
        tone_seconds=0.2,
    )

    assert payload["status"] == "ok"
    assert payload["best_route"]["input_device"] == 0
    assert payload["best_route"]["output_device"] == 1
    assert payload["best_route"]["selected_input_channel"] == 1
    assert payload["best_route"]["channel_metrics"][1]["signal_ok"] is True
    assert payload["verified_config_hint"] == {
        "voice.input_device": 0,
        "voice.input_transport": "sounddevice",
        "voice.mic_channels": 4,
        "voice.mic_channel_select": 1,
        "voice.tts.output_device": 1,
        "voice.tts.output_transport": "sounddevice",
    }


class _SilentSoundDevice(_FakeSoundDevice):
    @classmethod
    def playrec(cls, out, **kwargs):
        cls.last_playrec = kwargs
        channels = int(kwargs.get("channels") or 1)
        return np.zeros((len(out), channels), dtype=np.float32)


def test_audio_route_scan_reports_silent_microphone_diagnosis(monkeypatch):
    monkeypatch.setattr(audio_devices, "sd", _SilentSoundDevice)

    payload = audio_devices.run_audio_route_scan(
        input_devices=[0],
        output_devices=[1],
        sample_rates=[48000],
        record_seconds=0.4,
        tone_seconds=0.2,
    )

    assert payload["status"] == "degraded"
    assert payload["failure_reason"] == "no_audio_route_captured_test_signal"
    assert payload["diagnostic_hint"].startswith(
        "audio_playback_works_but_microphone_captures_silence"
    )
    assert payload["verified_config_hint"] == {}
    assert payload["best_route"]["failure_reason"] == "microphone_captured_silence"
