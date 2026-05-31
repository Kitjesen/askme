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


class _DuplexCheckedSoundDevice(_FakeSoundDevice):
    class Stream:
        def __init__(self, *_, device=None, **__):
            if tuple(device or ()) == (0, 1):
                raise RuntimeError("invalid duplex route")

        def start(self):
            return None

        def stop(self):
            return None

        def close(self):
            return None

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
                "name": "WDM-KS Mic",
                "hostapi": 1,
                "max_input_channels": 4,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
            {
                "name": "WDM-KS Speaker",
                "hostapi": 1,
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
            },
            {
                "name": "Windows WDM-KS",
                "devices": [2, 3],
                "default_input_device": 2,
                "default_output_device": 3,
            },
        ]


def test_query_audio_devices_recommends_first_full_duplex_route(monkeypatch):
    monkeypatch.setattr(audio_devices, "sd", _DuplexCheckedSoundDevice)

    payload = audio_devices.query_audio_devices()

    assert payload["status"] == "ok"
    assert payload["recommendation"]["input_device"] == 2
    assert payload["recommendation"]["output_device"] == 3


class _AlternateDuplexRouteSoundDevice(_DuplexCheckedSoundDevice):
    class Stream:
        def __init__(self, *_, device=None, **__):
            if tuple(device or ()) != (2, 4):
                raise RuntimeError("invalid duplex route")

        def start(self):
            return None

        def stop(self):
            return None

        def close(self):
            return None

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
                "name": "WDM-KS Mic",
                "hostapi": 1,
                "max_input_channels": 4,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
            {
                "name": "WDM-KS Speakers 1",
                "hostapi": 1,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            },
            {
                "name": "WDM-KS Speakers 2",
                "hostapi": 1,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            },
        ]
        if device is None:
            return devices
        return devices[int(device)]


def test_query_audio_devices_scans_same_hostapi_fallback_routes(monkeypatch):
    monkeypatch.setattr(audio_devices, "sd", _AlternateDuplexRouteSoundDevice)

    payload = audio_devices.query_audio_devices()

    assert payload["status"] == "ok"
    assert payload["recommendation"]["input_device"] == 2
    assert payload["recommendation"]["output_device"] == 4


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


class _InvalidFloatSoundDevice(_FakeSoundDevice):
    @classmethod
    def playrec(cls, out, **kwargs):
        cls.last_playrec = kwargs
        channels = int(kwargs.get("channels") or 1)
        captured = np.zeros((len(out), channels), dtype=np.float32)
        captured[:, 0] = np.inf
        return captured


def test_audio_route_scan_handles_invalid_float_samples(monkeypatch):
    monkeypatch.setattr(audio_devices, "sd", _InvalidFloatSoundDevice)

    payload = audio_devices.run_audio_route_scan(
        input_devices=[0],
        output_devices=[1],
        sample_rates=[48000],
        record_seconds=0.4,
        tone_seconds=0.2,
    )

    assert payload["status"] == "degraded"
    assert payload["best_route"]["failure_reason"] == "microphone_captured_silence"


# ── print_windows_beep_loopback_summary ──────────────────────────────────

class TestPrintWindowsBeepLoopbackSummary:
    def test_prints_status_ok(self, capsys):
        audio_devices.print_windows_beep_loopback_summary({
            "status": "ok",
            "input_device": 0,
            "playback_ok": True,
            "recording_ok": True,
            "playback_error": "",
            "record_error": "",
            "failure_reason": "",
            "peak": 500,
            "rms": 200,
            "tone_detected": True,
            "tone_correlation": 0.85,
        })
        out = capsys.readouterr().out
        assert "ok" in out

    def test_prints_degraded_with_errors(self, capsys):
        audio_devices.print_windows_beep_loopback_summary({
            "status": "degraded",
            "input_device": 1,
            "playback_ok": False,
            "recording_ok": True,
            "playback_error": "beep failed",
            "record_error": "",
            "failure_reason": "beep_playback_failed",
            "peak": 10,
            "rms": 5,
            "tone_detected": False,
            "tone_correlation": 0.01,
        })
        out = capsys.readouterr().out
        assert "degraded" in out
        assert "beep failed" in out
        assert "beep_playback_failed" in out

    def test_prints_record_error(self, capsys):
        audio_devices.print_windows_beep_loopback_summary({
            "status": "degraded",
            "input_device": 0,
            "playback_ok": True,
            "recording_ok": False,
            "playback_error": "",
            "record_error": "stream timeout",
            "failure_reason": "record_stream_failed",
            "peak": 0,
            "rms": 0,
            "tone_detected": False,
            "tone_correlation": 0.0,
        })
        out = capsys.readouterr().out
        assert "record-error" in out
        assert "stream timeout" in out


# ── Helper: audio device discovery ───────────────────────────────────────

class TestInputDevices:
    def test_returns_only_input_devices(self):
        devices = [
            {"index": 0, "is_input": True},
            {"index": 1, "is_input": False},
            {"index": 2, "is_input": True},
        ]
        assert audio_devices._input_devices(devices) == [0, 2]

    def test_empty_devices(self):
        assert audio_devices._input_devices([]) == []


class TestOutputDevices:
    def test_returns_only_output_devices(self):
        devices = [
            {"index": 0, "is_output": False},
            {"index": 1, "is_output": True},
            {"index": 2, "is_output": True},
        ]
        assert audio_devices._output_devices(devices) == [1, 2]

    def test_empty_devices(self):
        assert audio_devices._output_devices([]) == []


class TestHostapiIndexByDevice:
    def test_maps_device_index_to_hostapi(self):
        devices = [
            {"index": 0, "hostapi": 1},
            {"index": 5, "hostapi": 2},
        ]
        mapping = audio_devices._hostapi_index_by_device(devices)
        assert mapping == {0: 1, 5: 2}

    def test_returns_zero_for_missing_hostapi(self):
        devices = [{"index": 0}]
        mapping = audio_devices._hostapi_index_by_device(devices)
        assert mapping[0] == 0

    def test_empty_devices(self):
        assert audio_devices._hostapi_index_by_device([]) == {}


class TestNormaliseDeviceList:
    def test_converts_strings_to_ints(self):
        assert audio_devices._normalise_device_list(["1", "2"]) == [1, 2]

    def test_deduplicates(self):
        assert audio_devices._normalise_device_list([1, 2, 1, 3]) == [1, 2, 3]

    def test_handles_none(self):
        assert audio_devices._normalise_device_list(None) == []

    def test_filters_non_int_coercible(self):
        result = audio_devices._normalise_device_list(["mic_name", "1"])
        assert "mic_name" not in result


class TestCandidateRoutes:
    def test_finds_same_hostapi_routes(self):
        devices = [
            {"index": 0, "hostapi": 1, "is_input": True, "is_output": False},
            {"index": 1, "hostapi": 1, "is_input": False, "is_output": True},
            {"index": 2, "hostapi": 2, "is_input": False, "is_output": True},
        ]
        routes = audio_devices._candidate_routes(
            devices,
            input_devices=[0],
            output_devices=None,
            include_all_pairs=False,
            max_routes=10,
        )
        assert (0, 1) in routes
        assert (0, 2) not in routes  # different hostapi

    def test_include_all_pairs(self):
        devices = [
            {"index": 0, "hostapi": 1, "is_input": True, "is_output": False},
            {"index": 1, "hostapi": 2, "is_input": False, "is_output": True},
        ]
        routes = audio_devices._candidate_routes(
            devices,
            input_devices=[0],
            output_devices=None,
            include_all_pairs=True,
            max_routes=10,
        )
        assert (0, 1) in routes

    def test_respects_max_routes(self):
        devices = [
            {"index": 0, "hostapi": 1, "is_input": True, "is_output": False},
            {"index": 1, "hostapi": 1, "is_input": False, "is_output": True},
            {"index": 2, "hostapi": 1, "is_input": False, "is_output": True},
        ]
        routes = audio_devices._candidate_routes(
            devices,
            input_devices=[0],
            output_devices=None,
            include_all_pairs=False,
            max_routes=1,
        )
        assert len(routes) <= 1


class TestRouteSampleRates:
    def test_includes_device_and_standard_rates(self):
        rates = audio_devices._route_sample_rates(0, 1)
        assert 48000 in rates
        assert 44100 in rates
        assert len(rates) >= 2

    def test_no_duplicates(self):
        rates = audio_devices._route_sample_rates(0, 1)
        assert len(rates) == len(set(rates))


# ── print_audio_route_scan_summary ─────────────────────────────────────

class TestPrintAudioRouteScanSummary:
    """capsys tests for print_audio_route_scan_summary."""

    def test_empty_payload(self, capsys):
        audio_devices.print_audio_route_scan_summary({})
        out = capsys.readouterr().out
        assert "Audio route scan: unknown" in out
        # No failure_reason, diagnostic_hint, best_route, verified_config_hint, routes
        assert "failure-reason" not in out
        assert "diagnostic-hint" not in out
        assert "best:" not in out
        assert "verified-config" not in out
        assert "route:" not in out

    def test_best_route_present(self, capsys):
        audio_devices.print_audio_route_scan_summary({
            "status": "ok",
            "best_route": {
                "input_device": 0,
                "output_device": 1,
                "sample_rate": 48000,
                "selected_input_channel": 0,
                "peak": 500,
                "tone_correlation": 0.85,
                "status": "ok",
            },
        })
        out = capsys.readouterr().out
        assert "Audio route scan: ok" in out
        assert "best:" in out
        assert "in=0" in out
        assert "out=1" in out
        assert "sr=48000" in out
        assert "ch=0" in out
        assert "peak=500" in out
        assert "corr=0.85" in out
        assert "status=ok" in out

    def test_failure_reason_and_diagnostic_hint(self, capsys):
        audio_devices.print_audio_route_scan_summary({
            "status": "degraded",
            "failure_reason": "no_audio_route_captured_test_signal",
            "diagnostic_hint": (
                "audio_playback_works_but_microphone_captures_silence:"
                "check_windows_input_permission_device_mute_and_selected_array_channel"
            ),
        })
        out = capsys.readouterr().out
        assert "Audio route scan: degraded" in out
        assert "failure-reason: no_audio_route_captured_test_signal" in out
        assert "diagnostic-hint: audio_playback_works_but_microphone_captures_silence" in out

    def test_verified_config_hint(self, capsys):
        audio_devices.print_audio_route_scan_summary({
            "status": "ok",
            "best_route": {
                "input_device": 0,
                "output_device": 1,
                "sample_rate": 48000,
                "selected_input_channel": 1,
                "peak": 640,
                "tone_correlation": 0.92,
                "status": "ok",
            },
            "verified_config_hint": {
                "voice.input_device": 0,
                "voice.input_transport": "sounddevice",
                "voice.mic_channels": 4,
                "voice.mic_channel_select": 1,
                "voice.tts.output_device": 1,
                "voice.tts.output_transport": "sounddevice",
            },
        })
        out = capsys.readouterr().out
        assert "verified-config:" in out
        assert '"voice.input_device": 0' in out
        assert '"voice.tts.output_device": 1' in out

    def test_mixed_route_states(self, capsys):
        audio_devices.print_audio_route_scan_summary({
            "status": "degraded",
            "failure_reason": "no_audio_route_captured_test_signal",
            "diagnostic_hint": "microphone_captured_silence,input_output_hostapi_mismatch",
            "best_route": {
                "input_device": 0,
                "output_device": 1,
                "sample_rate": 48000,
                "selected_input_channel": 0,
                "peak": 10,
                "tone_correlation": 0.0,
                "status": "degraded",
                "failure_reason": "microphone_captured_silence",
            },
            "routes": [
                {
                    "input_device": 0,
                    "output_device": 1,
                    "sample_rate": 48000,
                    "selected_input_channel": 0,
                    "peak": 10,
                    "tone_correlation": 0.0,
                    "failure_reason": "microphone_captured_silence",
                },
                {
                    "input_device": 2,
                    "output_device": 3,
                    "sample_rate": 44100,
                    "selected_input_channel": 0,
                    "peak": 5,
                    "tone_correlation": 0.0,
                    "failure_reason": "input_output_hostapi_mismatch",
                },
            ],
        })
        out = capsys.readouterr().out
        assert "Audio route scan: degraded" in out
        assert "failure-reason: no_audio_route_captured_test_signal" in out
        assert "diagnostic-hint: microphone_captured_silence" in out
        # Two route lines
        assert "route:" in out
        assert "microphone_captured_silence" in out
        assert "input_output_hostapi_mismatch" in out
        # Best route also displayed
        assert "best:" in out
