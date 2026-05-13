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
        return np.asarray(out, dtype=np.float32) * 0.8

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
    assert wav_out.exists()
    assert _FakeSoundDevice.last_playrec["device"] == (0, 1)
