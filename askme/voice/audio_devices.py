"""Local audio device diagnostics and loopback smoke tests."""

from __future__ import annotations

import json
import platform
import wave
from pathlib import Path
from typing import Any

import numpy as np

try:
    import sounddevice as sd
except ModuleNotFoundError:  # pragma: no cover - exercised on hosts without PortAudio
    sd = None  # type: ignore[assignment]


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)


def _clean_device(device: Any, index: int) -> dict[str, Any]:
    row = {str(k): _json_safe(v) for k, v in dict(device).items()}
    row["index"] = index
    row["is_input"] = int(row.get("max_input_channels", 0) or 0) > 0
    row["is_output"] = int(row.get("max_output_channels", 0) or 0) > 0
    return row


def _clean_hostapi(hostapi: Any) -> dict[str, Any]:
    return {str(k): _json_safe(v) for k, v in dict(hostapi).items()}


def _coerce_device(value: int | str | None) -> int | str | None:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return str(value)


def _default_device_pair() -> tuple[int | None, int | None]:
    if sd is None:
        return None, None
    default = _json_safe(sd.default.device)
    if isinstance(default, list) and len(default) >= 2:
        return default[0], default[1]
    if isinstance(default, tuple) and len(default) >= 2:
        return default[0], default[1]
    return None, None


def _preferred_hostapi_default(hostapis: list[dict[str, Any]], *, kind: str) -> int | None:
    preferred_names = ("windows wasapi", "windows directsound", "mme")
    field = "default_input_device" if kind == "input" else "default_output_device"
    for expected_name in preferred_names:
        for api in hostapis:
            if str(api.get("name", "")).strip().lower() == expected_name:
                value = api.get(field)
                return int(value) if isinstance(value, int) and value >= 0 else None
    return None


def query_audio_devices() -> dict[str, Any]:
    """Return PortAudio devices plus a Windows-friendly recommendation."""
    if sd is None:
        return {
            "status": "error",
            "platform": platform.platform(),
            "error": "sounddevice is not installed",
            "devices": [],
            "hostapis": [],
            "recommendation": {},
        }

    try:
        devices = [_clean_device(device, idx) for idx, device in enumerate(sd.query_devices())]
        hostapis = [_clean_hostapi(api) for api in sd.query_hostapis()]
        default_input, default_output = _default_device_pair()
        recommended_input = _preferred_hostapi_default(hostapis, kind="input")
        if recommended_input is None:
            recommended_input = default_input
        recommended_output = _preferred_hostapi_default(hostapis, kind="output")
        if recommended_output is None:
            recommended_output = default_output
        recommendation = {
            "input_device": recommended_input,
            "output_device": recommended_output,
            "input_transport": "sounddevice",
            "output_transport": "sounddevice",
            "config_hint": {
                "voice.input_device": recommended_input,
                "voice.input_transport": "sounddevice",
                "voice.tts.output_device": recommended_output,
                "voice.tts.output_transport": "sounddevice",
            },
        }
        return {
            "status": "ok",
            "platform": platform.platform(),
            "default_device": _json_safe(sd.default.device),
            "hostapis": hostapis,
            "devices": devices,
            "recommendation": recommendation,
        }
    except Exception as exc:  # pragma: no cover - hardware/runtime path
        return {
            "status": "error",
            "platform": platform.platform(),
            "error": str(exc),
            "devices": [],
            "hostapis": [],
            "recommendation": {},
        }


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono float32 samples as PCM16 WAV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = np.clip(samples.reshape(-1), -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2")
    with wave.open(str(target), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def _device_default_samplerate(device: int | str | None, *, fallback: int = 48000) -> int:
    if sd is None or device is None:
        return fallback
    try:
        info = sd.query_devices(device)
        return int(float(info.get("default_samplerate") or fallback))
    except Exception:
        return fallback


def _device_max_output_channels(device: int | str | None, *, fallback: int = 2) -> int:
    if sd is None or device is None:
        return fallback
    try:
        info = sd.query_devices(device)
        return max(1, int(info.get("max_output_channels") or fallback))
    except Exception:
        return fallback


def _device_max_input_channels(device: int | str | None, *, fallback: int = 1) -> int:
    if sd is None or device is None:
        return fallback
    try:
        info = sd.query_devices(device)
        return max(1, int(info.get("max_input_channels") or fallback))
    except Exception:
        return fallback


def run_audio_loopback(
    *,
    input_device: int | str | None = None,
    output_device: int | str | None = None,
    sample_rate: int | None = None,
    record_seconds: float = 2.0,
    tone_seconds: float = 0.8,
    frequency_hz: float = 880.0,
    output_gain: float = 0.25,
    min_capture_peak: int = 300,
    wav_out: str | Path | None = None,
    play_recording: bool = False,
) -> dict[str, Any]:
    """Play a tone through the selected output while recording the mic."""
    if sd is None:
        return {"status": "error", "error": "sounddevice is not installed"}

    devices_payload = query_audio_devices()
    recommendation = devices_payload.get("recommendation", {})
    input_dev = _coerce_device(input_device)
    output_dev = _coerce_device(output_device)
    if input_dev is None:
        input_dev = recommendation.get("input_device")
    if output_dev is None:
        output_dev = recommendation.get("output_device")

    resolved_sample_rate = int(
        sample_rate or _device_default_samplerate(output_dev, fallback=48000)
    )
    total_seconds = max(record_seconds, tone_seconds + 0.6)
    total_frames = max(1, int(total_seconds * resolved_sample_rate))
    tone_frames = max(1, int(tone_seconds * resolved_sample_rate))
    start_frame = max(0, int(0.25 * resolved_sample_rate))
    end_frame = min(total_frames, start_frame + tone_frames)
    input_channels = _device_max_input_channels(input_dev, fallback=1)
    output_channels = min(2, _device_max_output_channels(output_dev, fallback=2))

    out = np.zeros((total_frames, output_channels), dtype=np.float32)
    t = np.arange(end_frame - start_frame, dtype=np.float32) / float(resolved_sample_rate)
    tone = np.sin(2.0 * np.pi * frequency_hz * t).astype(np.float32)
    ramp = min(len(tone) // 8, int(0.02 * resolved_sample_rate))
    if ramp > 1:
        envelope = np.ones_like(tone)
        edge = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
        envelope[:ramp] = edge
        envelope[-ramp:] = edge[::-1]
        tone *= envelope
    out[start_frame:end_frame, :] = tone.reshape(-1, 1) * float(output_gain)

    playback_ok = False
    playback_error = ""
    try:
        captured = sd.playrec(
            out,
            samplerate=resolved_sample_rate,
            channels=input_channels,
            dtype="float32",
            device=(input_dev, output_dev),
            blocking=True,
        )
        playback_ok = True
    except Exception as exc:  # pragma: no cover - hardware/runtime path
        captured = np.empty((0, 1), dtype=np.float32)
        playback_error = str(exc)

    if captured.ndim == 2 and captured.shape[1] > 1:
        channel_energy = np.sqrt(np.mean(captured * captured, axis=0))
        channel_index = int(np.argmax(channel_energy))
        mono = captured[:, channel_index].astype(np.float32, copy=False)
    else:
        channel_index = 0
        mono = captured.reshape(-1).astype(np.float32, copy=False)
    peak = int(float(np.max(np.abs(mono))) * 32768) if len(mono) else 0
    rms = int(float(np.sqrt(np.mean(mono * mono))) * 32768) if len(mono) else 0

    segment = mono[start_frame:end_frame] if len(mono) >= end_frame else mono
    correlation = 0.0
    if len(segment) > 0:
        ref_t = np.arange(len(segment), dtype=np.float32) / float(resolved_sample_rate)
        reference = np.sin(2.0 * np.pi * frequency_hz * ref_t).astype(np.float32)
        denom = float(np.linalg.norm(segment) * np.linalg.norm(reference))
        if denom > 0:
            correlation = abs(float(np.dot(segment, reference)) / denom)

    signal_ok = bool(playback_ok and peak >= min_capture_peak)
    tone_detected = bool(signal_ok and correlation >= 0.12)
    status = "ok" if playback_ok and signal_ok else "degraded"

    wav_path = str(wav_out) if wav_out else ""
    if wav_path:
        write_wav(wav_path, mono, resolved_sample_rate)

    replay_ok = None
    replay_error = ""
    if play_recording and len(mono):
        try:
            sd.play(mono, samplerate=resolved_sample_rate, device=output_dev)
            sd.wait()
            replay_ok = True
        except Exception as exc:  # pragma: no cover - hardware/runtime path
            replay_ok = False
            replay_error = str(exc)

    return {
        "status": status,
        "playback_ok": playback_ok,
        "playback_error": playback_error,
        "signal_ok": signal_ok,
        "tone_detected": tone_detected,
        "tone_correlation": round(correlation, 3),
        "peak": peak,
        "rms": rms,
        "min_capture_peak": min_capture_peak,
        "sample_rate": resolved_sample_rate,
        "input_device": input_dev,
        "output_device": output_dev,
        "input_channels": input_channels,
        "selected_input_channel": channel_index,
        "output_channels": output_channels,
        "record_seconds": total_seconds,
        "tone_seconds": tone_seconds,
        "frequency_hz": frequency_hz,
        "wav_out": wav_path,
        "replay_ok": replay_ok,
        "replay_error": replay_error,
    }


def print_audio_devices_summary(payload: dict[str, Any]) -> None:
    print(f"Audio devices: {payload.get('status', 'unknown')}")  # noqa: T201
    if payload.get("error"):
        print(f"  error: {payload['error']}")  # noqa: T201
        return
    print(f"  platform: {payload.get('platform')}")  # noqa: T201
    print(f"  default: {payload.get('default_device')}")  # noqa: T201
    print("  input devices:")  # noqa: T201
    for device in payload.get("devices", []):
        if device.get("is_input"):
            print(f"    [{device['index']}] {device['name']} ch={device['max_input_channels']}")  # noqa: T201
    print("  output devices:")  # noqa: T201
    for device in payload.get("devices", []):
        if device.get("is_output"):
            print(f"    [{device['index']}] {device['name']} ch={device['max_output_channels']}")  # noqa: T201
    print(f"  recommendation: {json.dumps(payload.get('recommendation', {}), ensure_ascii=False)}")  # noqa: T201


def print_audio_loopback_summary(payload: dict[str, Any]) -> None:
    print(f"Audio loopback: {payload.get('status', 'unknown')}")  # noqa: T201
    print(f"  input/output: {payload.get('input_device')} -> {payload.get('output_device')}")  # noqa: T201
    print(f"  playback-ok: {payload.get('playback_ok')}")  # noqa: T201
    if payload.get("playback_error"):
        print(f"  playback-error: {payload['playback_error']}")  # noqa: T201
    print(  # noqa: T201
        "  capture: "
        f"peak={payload.get('peak')} rms={payload.get('rms')} "
        f"signal_ok={payload.get('signal_ok')} tone_detected={payload.get('tone_detected')} "
        f"corr={payload.get('tone_correlation')}"
    )
    if payload.get("wav_out"):
        print(f"  wav: {payload['wav_out']}")  # noqa: T201
