from __future__ import annotations

from types import SimpleNamespace

from askme.runtime.deployment_preflight import (
    probe_audio_devices,
    run_edge_robot_preflight,
)


class _FakeSoundDevice:
    def __init__(self, devices, default=(0, 1)):
        self._devices = devices
        self.default = SimpleNamespace(device=default)
        self.checked_input = None
        self.checked_output = None

    def query_devices(self, device=None, kind=None):
        if device is None and kind is None:
            return self._devices
        if device is None:
            device = self.default.device[0 if kind == "input" else 1]
        return self._devices[int(device)]

    def check_input_settings(self, **kwargs):
        self.checked_input = kwargs

    def check_output_settings(self, **kwargs):
        self.checked_output = kwargs


class _NonSequenceDevicePair:
    def __getitem__(self, index):
        return (0, 1)[index]


def test_audio_preflight_accepts_sounddevice_default_pair_protocol() -> None:
    sounddevice = _FakeSoundDevice(
        [
            {
                "name": "robot-mic",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48_000,
            },
            {
                "name": "robot-speaker",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48_000,
            },
        ],
        default=_NonSequenceDevicePair(),
    )

    payload = probe_audio_devices({"voice": {"tts": {}}}, sounddevice_module=sounddevice)

    assert payload["ok"] is True
    assert payload["input"]["index"] == 0
    assert payload["output"]["index"] == 1


def test_audio_preflight_verifies_configured_input_and_output() -> None:
    sounddevice = _FakeSoundDevice(
        [
            {
                "name": "robot-mic",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48_000,
            },
            {
                "name": "robot-speaker",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48_000,
            },
        ]
    )

    payload = probe_audio_devices(
        {
            "voice": {
                "input_device": 0,
                "tts": {"output_device": 1},
            }
        },
        sounddevice_module=sounddevice,
    )

    assert payload["ok"] is True
    assert payload["input"]["index"] == 0
    assert payload["output"]["index"] == 1
    assert sounddevice.checked_input["device"] == 0
    assert sounddevice.checked_output["device"] == 1


def test_audio_preflight_fails_when_no_output_device_is_available() -> None:
    sounddevice = _FakeSoundDevice(
        [
            {
                "name": "input-only",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48_000,
            }
        ],
        default=(0, -1),
    )

    payload = probe_audio_devices(
        {"voice": {"input_device": 0, "tts": {"output_device": None}}},
        sounddevice_module=sounddevice,
    )

    assert payload["ok"] is False
    assert any("output device" in error for error in payload["errors"])


def test_edge_preflight_combines_model_and_audio_failures_without_config_echo() -> None:
    secret = "must-not-appear"
    config = {"voice": {"tts": {"minimax_api_key": secret}}}

    payload = run_edge_robot_preflight(
        config,
        voice_health_runner=lambda *_args, **_kwargs: {
            "status": "degraded",
            "errors": ["ASR missing encoder: /app/models/asr/encoder.onnx"],
            "warnings": [],
            "models_ok": False,
        },
        audio_probe=lambda _cfg: {
            "ok": False,
            "errors": ["audio output device is not available"],
            "input": {},
            "output": {},
        },
    )

    assert payload["status"] == "blocked"
    assert payload["ready"] is False
    assert payload["errors"] == [
        "ASR missing encoder: /app/models/asr/encoder.onnx",
        "audio output device is not available",
    ]
    assert secret not in repr(payload)


def test_edge_preflight_passes_only_when_voice_and_audio_checks_pass() -> None:
    payload = run_edge_robot_preflight(
        {"voice": {}},
        voice_health_runner=lambda *_args, **_kwargs: {
            "status": "ok",
            "errors": [],
            "warnings": ["runtime voice bridge is disabled"],
            "models_ok": True,
        },
        audio_probe=lambda _cfg: {
            "ok": True,
            "errors": [],
            "input": {"index": 0},
            "output": {"index": 1},
        },
    )

    assert payload["status"] == "ok"
    assert payload["ready"] is True
