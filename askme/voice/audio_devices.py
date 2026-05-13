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


def _safe_float_audio(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(values, -1.0, 1.0).astype(np.float32, copy=False)


def _channel_metrics(
    captured: np.ndarray,
    *,
    start_frame: int,
    end_frame: int,
    sample_rate: int,
    frequency_hz: float,
    min_capture_peak: int,
) -> list[dict[str, Any]]:
    """Return per-channel capture evidence for loopback diagnosis."""
    if captured.size == 0:
        return []
    if captured.ndim == 1:
        channels = captured.reshape(-1, 1)
    else:
        channels = captured
    rows: list[dict[str, Any]] = []
    for channel_index in range(channels.shape[1]):
        samples = _safe_float_audio(channels[:, channel_index])
        peak = int(float(np.max(np.abs(samples))) * 32768) if len(samples) else 0
        rms = int(float(np.sqrt(np.mean(samples * samples))) * 32768) if len(samples) else 0
        segment = samples[start_frame:end_frame] if len(samples) >= end_frame else samples
        correlation = 0.0
        if len(segment) > 0:
            ref_t = np.arange(len(segment), dtype=np.float32) / float(sample_rate)
            reference = np.sin(2.0 * np.pi * frequency_hz * ref_t).astype(np.float32)
            denom = float(np.linalg.norm(segment) * np.linalg.norm(reference))
            if denom > 0:
                correlation = abs(float(np.dot(segment, reference)) / denom)
        rows.append(
            {
                "channel": channel_index,
                "peak": peak,
                "rms": rms,
                "tone_correlation": round(correlation, 3),
                "signal_ok": bool(peak >= min_capture_peak),
                "tone_detected": bool(peak >= min_capture_peak and correlation >= 0.12),
            }
        )
    return rows


def _best_channel(metrics: list[dict[str, Any]]) -> int:
    if not metrics:
        return 0
    best = max(
        metrics,
        key=lambda row: (
            bool(row.get("tone_detected")),
            float(row.get("tone_correlation") or 0.0),
            int(row.get("peak") or 0),
            int(row.get("rms") or 0),
        ),
    )
    return int(best.get("channel") or 0)


def _loopback_failure_reason(
    *,
    playback_ok: bool,
    playback_error: str,
    peak: int,
    min_capture_peak: int,
    tone_detected: bool,
) -> str:
    if not playback_ok:
        if "Illegal combination of I/O devices" in playback_error:
            return "input_output_hostapi_mismatch"
        if "Invalid sample rate" in playback_error:
            return "invalid_sample_rate_for_device"
        return "playback_or_record_stream_failed"
    if peak <= 1:
        return "microphone_captured_silence"
    if peak < min_capture_peak:
        return "microphone_signal_below_threshold"
    if not tone_detected:
        return "captured_signal_not_matching_test_tone"
    return ""


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

    channels = _channel_metrics(
        captured,
        start_frame=start_frame,
        end_frame=end_frame,
        sample_rate=resolved_sample_rate,
        frequency_hz=frequency_hz,
        min_capture_peak=min_capture_peak,
    )
    channel_index = _best_channel(channels)
    if captured.ndim == 2 and captured.shape[1] > channel_index:
        mono = _safe_float_audio(captured[:, channel_index])
    else:
        mono = _safe_float_audio(captured.reshape(-1))
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
    status = "ok" if playback_ok and signal_ok and tone_detected else "degraded"
    failure_reason = _loopback_failure_reason(
        playback_ok=playback_ok,
        playback_error=playback_error,
        peak=peak,
        min_capture_peak=min_capture_peak,
        tone_detected=tone_detected,
    )

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
        "failure_reason": failure_reason,
        "peak": peak,
        "rms": rms,
        "min_capture_peak": min_capture_peak,
        "sample_rate": resolved_sample_rate,
        "input_device": input_dev,
        "output_device": output_dev,
        "input_channels": input_channels,
        "selected_input_channel": channel_index,
        "channel_metrics": channels,
        "output_channels": output_channels,
        "record_seconds": total_seconds,
        "tone_seconds": tone_seconds,
        "frequency_hz": frequency_hz,
        "wav_out": wav_path,
        "replay_ok": replay_ok,
        "replay_error": replay_error,
    }


def _input_devices(devices: list[dict[str, Any]]) -> list[int]:
    return [int(device["index"]) for device in devices if device.get("is_input")]


def _output_devices(devices: list[dict[str, Any]]) -> list[int]:
    return [int(device["index"]) for device in devices if device.get("is_output")]


def _hostapi_index_by_device(devices: list[dict[str, Any]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for device in devices:
        try:
            mapping[int(device["index"])] = int(device.get("hostapi") or 0)
        except Exception:
            continue
    return mapping


def _normalise_device_list(values: list[int | str] | tuple[int | str, ...] | None) -> list[int]:
    result: list[int] = []
    for value in values or []:
        coerced = _coerce_device(value)
        if isinstance(coerced, int) and coerced not in result:
            result.append(coerced)
    return result


def _candidate_routes(
    devices: list[dict[str, Any]],
    *,
    input_devices: list[int] | None,
    output_devices: list[int] | None,
    include_all_pairs: bool,
    max_routes: int,
) -> list[tuple[int, int]]:
    inputs = input_devices or _input_devices(devices)
    outputs = output_devices or _output_devices(devices)
    hostapi_by_device = _hostapi_index_by_device(devices)
    routes: list[tuple[int, int]] = []
    for input_device in inputs:
        for output_device in outputs:
            same_hostapi = (
                hostapi_by_device.get(input_device) is not None
                and hostapi_by_device.get(input_device) == hostapi_by_device.get(output_device)
            )
            if not include_all_pairs and not same_hostapi:
                continue
            route = (input_device, output_device)
            if route not in routes:
                routes.append(route)
            if len(routes) >= max_routes:
                return routes
    return routes


def run_audio_route_scan(
    *,
    input_devices: list[int | str] | tuple[int | str, ...] | None = None,
    output_devices: list[int | str] | tuple[int | str, ...] | None = None,
    sample_rates: list[int] | tuple[int, ...] | None = None,
    record_seconds: float = 1.0,
    tone_seconds: float = 0.35,
    frequency_hz: float = 880.0,
    output_gain: float = 0.35,
    min_capture_peak: int = 300,
    include_all_pairs: bool = False,
    max_routes: int = 24,
) -> dict[str, Any]:
    """Scan input/output routes and rank them by captured signal evidence."""
    if sd is None:
        return {"status": "error", "error": "sounddevice is not installed", "routes": []}

    devices_payload = query_audio_devices()
    if devices_payload.get("status") != "ok":
        return {
            "status": "error",
            "error": devices_payload.get("error", "audio device query failed"),
            "routes": [],
        }

    devices = list(devices_payload.get("devices", []))
    inputs = _normalise_device_list(input_devices)
    outputs = _normalise_device_list(output_devices)
    routes = _candidate_routes(
        devices,
        input_devices=inputs or None,
        output_devices=outputs or None,
        include_all_pairs=include_all_pairs,
        max_routes=max(1, int(max_routes)),
    )
    rates = list(sample_rates or [48000, 44100])
    results: list[dict[str, Any]] = []
    for input_device, output_device in routes:
        for rate in rates:
            result = run_audio_loopback(
                input_device=input_device,
                output_device=output_device,
                sample_rate=int(rate),
                record_seconds=record_seconds,
                tone_seconds=tone_seconds,
                frequency_hz=frequency_hz,
                output_gain=output_gain,
                min_capture_peak=min_capture_peak,
                wav_out=None,
                play_recording=False,
            )
            results.append(result)

    results.sort(
        key=lambda row: (
            row.get("status") == "ok",
            bool(row.get("tone_detected")),
            int(row.get("peak") or 0),
            float(row.get("tone_correlation") or 0.0),
        ),
        reverse=True,
    )
    best = results[0] if results else {}
    any_ok = any(row.get("status") == "ok" for row in results)
    verified_config_hint: dict[str, Any] = {}
    if any_ok and best:
        verified_config_hint = {
            "voice.input_device": best.get("input_device"),
            "voice.input_transport": "sounddevice",
            "voice.mic_channels": best.get("input_channels"),
            "voice.mic_channel_select": best.get("selected_input_channel"),
            "voice.tts.output_device": best.get("output_device"),
            "voice.tts.output_transport": "sounddevice",
        }
    diagnostic_hint = ""
    if not any_ok:
        failure_reasons = {
            str(row.get("failure_reason"))
            for row in results
            if row.get("failure_reason")
        }
        if "microphone_captured_silence" in failure_reasons:
            diagnostic_hint = (
                "audio_playback_works_but_microphone_captures_silence:"
                "check_windows_input_permission_device_mute_and_selected_array_channel"
            )
        elif "input_output_hostapi_mismatch" in failure_reasons:
            diagnostic_hint = "try_same_hostapi_wasapi_or_mme_input_output_pair"
        elif failure_reasons:
            diagnostic_hint = ",".join(sorted(failure_reasons))
    return {
        "status": "ok" if any_ok else "degraded",
        "failure_reason": "" if any_ok else "no_audio_route_captured_test_signal",
        "diagnostic_hint": diagnostic_hint,
        "platform": devices_payload.get("platform"),
        "recommendation": devices_payload.get("recommendation", {}),
        "verified_config_hint": verified_config_hint,
        "route_count": len(results),
        "best_route": best,
        "routes": results,
        "min_capture_peak": min_capture_peak,
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
    if payload.get("failure_reason"):
        print(f"  failure-reason: {payload['failure_reason']}")  # noqa: T201
    print(  # noqa: T201
        "  capture: "
        f"peak={payload.get('peak')} rms={payload.get('rms')} "
        f"signal_ok={payload.get('signal_ok')} tone_detected={payload.get('tone_detected')} "
        f"corr={payload.get('tone_correlation')}"
    )
    if payload.get("wav_out"):
        print(f"  wav: {payload['wav_out']}")  # noqa: T201


def print_audio_route_scan_summary(payload: dict[str, Any]) -> None:
    print(f"Audio route scan: {payload.get('status', 'unknown')}")  # noqa: T201
    if payload.get("failure_reason"):
        print(f"  failure-reason: {payload['failure_reason']}")  # noqa: T201
    if payload.get("diagnostic_hint"):
        print(f"  diagnostic-hint: {payload['diagnostic_hint']}")  # noqa: T201
    best = payload.get("best_route") or {}
    if best:
        print(  # noqa: T201
            "  best: "
            f"in={best.get('input_device')} out={best.get('output_device')} "
            f"sr={best.get('sample_rate')} ch={best.get('selected_input_channel')} "
            f"peak={best.get('peak')} corr={best.get('tone_correlation')} "
            f"status={best.get('status')}"
        )
    if payload.get("verified_config_hint"):
        print(  # noqa: T201
            "  verified-config: "
            f"{json.dumps(payload['verified_config_hint'], ensure_ascii=False)}"
        )
    for row in payload.get("routes", [])[:8]:
        print(  # noqa: T201
            "  route: "
            f"in={row.get('input_device')} out={row.get('output_device')} "
            f"sr={row.get('sample_rate')} ch={row.get('selected_input_channel')} "
            f"peak={row.get('peak')} corr={row.get('tone_correlation')} "
            f"reason={row.get('failure_reason')}"
        )
