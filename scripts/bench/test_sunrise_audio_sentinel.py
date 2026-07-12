#!/usr/bin/env python3
"""Sunrise audio sentinel: product speaker playback -> room mic -> ASR.

The script intentionally tests the hardware-critical path only.  It prebuffers
TTS by default so MiniMax/network latency cannot consume the recording window,
then plays through TTSEngine's configured product output path while MicInput
records the room loopback.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import unicodedata
import wave
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from askme.voice.asr import ASREngine
from askme.voice.cloud_asr import CloudASR
from askme.voice.mic_input import MicInput
from askme.voice.tts import TTSEngine

from askme.config import get_config

_DIGIT_ALIASES = {
    "0": "\u96f6",
    "1": "\u4e00",
    "\u5e7a": "\u4e00",
    "\u58f9": "\u4e00",
    "2": "\u4e8c",
    "\u4e24": "\u4e8c",
    "\u8d30": "\u4e8c",
    "3": "\u4e09",
    "\u53c1": "\u4e09",
    "4": "\u56db",
    "\u8086": "\u56db",
    "5": "\u4e94",
    "\u4f0d": "\u4e94",
    "6": "\u516d",
    "\u9646": "\u516d",
    "7": "\u4e03",
    "\u67d2": "\u4e03",
    "8": "\u516b",
    "\u634c": "\u516b",
    "9": "\u4e5d",
    "\u7396": "\u4e5d",
}


def normalize_transcript(text: str) -> str:
    """Normalize ASR text for first-token sentinel comparisons."""
    normalized: list[str] = []
    for char in text.strip().lower():
        if char in _DIGIT_ALIASES:
            normalized.append(_DIGIT_ALIASES[char])
            continue
        if char.isspace():
            continue
        category = unicodedata.category(char)
        if category.startswith("P") or category.startswith("S"):
            continue
        normalized.append(char)
    return "".join(normalized)


def transcript_has_prefix(transcript: str, expected_prefix: str) -> bool:
    expected = normalize_transcript(expected_prefix)
    actual = normalize_transcript(transcript)
    return bool(expected) and actual.startswith(expected)


def peak_rms(samples: np.ndarray) -> tuple[int, int]:
    if len(samples) == 0:
        return 0, 0
    abs_samples = np.abs(samples)
    peak = int(float(np.max(abs_samples)) * 32768)
    rms = int(float(np.sqrt(np.mean(samples * samples))) * 32768)
    return peak, rms


def detect_onset_ms(
    samples: np.ndarray,
    sample_rate: int,
    playback_start_offset_s: float,
    *,
    min_peak: int = 300,
    baseline_multiplier: float = 4.0,
    window_ms: int = 50,
) -> tuple[float | None, int]:
    """Return first energy onset after playback start plus the peak threshold."""
    if len(samples) == 0 or sample_rate <= 0:
        return None, min_peak

    window = max(1, int(sample_rate * window_ms / 1000))
    playback_start = max(0, int(playback_start_offset_s * sample_rate))
    baseline_end = max(0, playback_start - window)
    baseline = samples[:baseline_end]
    baseline_peak, _baseline_rms = peak_rms(baseline)
    threshold = max(min_peak, int(baseline_peak * baseline_multiplier))

    hop = max(1, window // 2)
    for start in range(playback_start, max(playback_start, len(samples) - window + 1), hop):
        window_samples = samples[start : start + window]
        peak, _rms = peak_rms(window_samples)
        if peak >= threshold:
            above = np.where(np.abs(window_samples) * 32768 >= threshold)[0]
            crossing = int(above[0]) if len(above) else 0
            onset_s = (start + crossing) / sample_rate - playback_start_offset_s
            return max(0.0, onset_s * 1000.0), threshold
    return None, threshold


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    passes = sum(
        1
        for trial in trials
        if trial.get("prefix_ok") and trial.get("signal_ok", True)
    )
    required = max(1, (len(trials) // 2) + 1)
    return {
        "trials": len(trials),
        "passes": passes,
        "required_passes": required,
        "passed": passes >= required,
    }


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (samples * 32768).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def transcribe_local(samples: np.ndarray, sample_rate: int, cfg: dict[str, Any]) -> tuple[str, float]:
    started = time.perf_counter()
    engine = ASREngine(cfg.get("voice", {}).get("asr", {}))
    if engine.recognizer is None:
        return "", 0.0
    stream = engine.recognizer.create_stream()
    step = max(1, int(sample_rate * 0.1))
    for pos in range(0, len(samples), step):
        stream.accept_waveform(sample_rate, samples[pos : pos + step])
        while engine.recognizer.is_ready(stream):
            engine.recognizer.decode_stream(stream)
    return engine.recognizer.get_result(stream).strip(), (time.perf_counter() - started) * 1000.0


def transcribe_cloud(
    samples: np.ndarray,
    sample_rate: int,
    cfg: dict[str, Any],
    *,
    timeout: float,
    realtime: bool = True,
) -> tuple[str, float, str]:
    """Transcribe recorded room-loop audio with the configured CloudASR backend."""
    started = time.perf_counter()
    voice_cfg = cfg.get("voice", {}) if isinstance(cfg, dict) else {}
    cloud_cfg = voice_cfg.get("cloud_asr", {}) if isinstance(voice_cfg, dict) else {}
    if not isinstance(cloud_cfg, dict):
        cloud_cfg = {}

    cloud = CloudASR(cloud_cfg)
    if not cloud.available:
        return "", 0.0, "Cloud ASR is not enabled or api_key is missing"

    cloud_rate = int(cloud_cfg.get("sample_rate", 16000) or 16000)
    audio = samples.astype(np.float32, copy=False)
    if sample_rate != cloud_rate:
        audio = MicInput._resample_mono(audio, sample_rate, cloud_rate)
    pcm16 = MicInput.to_int16(audio)

    try:
        if not cloud.start_session():
            return "", (time.perf_counter() - started) * 1000.0, "Cloud ASR session did not start"
        step = max(1, int(cloud_rate * 0.1))
        for pos in range(0, len(pcm16), step):
            chunk = pcm16[pos : pos + step]
            cloud.feed(chunk.tobytes())
            if realtime:
                time.sleep(len(chunk) / float(cloud_rate or 1))
        text = cloud.finish_session(timeout=timeout).strip()
        return text, (time.perf_counter() - started) * 1000.0, ""
    except Exception as exc:  # pragma: no cover - network/runtime path
        try:
            cloud.cancel_session()
        except Exception:
            pass
        return "", (time.perf_counter() - started) * 1000.0, str(exc)


def slice_asr_samples(
    samples: np.ndarray,
    sample_rate: int,
    start_s: float,
    *,
    margin_s: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Return the ASR slice and its start offset in the original recording."""
    start_index = max(0, int((start_s - margin_s) * sample_rate))
    return samples[start_index:], start_index / float(sample_rate or 1)


def estimate_speech_start_s(first_play_s: float, tts: TTSEngine) -> float:
    """Estimate when real speech starts after USB direct lead-in/cushion audio."""
    leadin = float(getattr(tts, "_usb_direct_speech_leadin_seconds", 0.0) or 0.0)
    cushion = float(getattr(tts, "_usb_direct_speech_onset_cushion_seconds", 0.0) or 0.0)
    gap = float(getattr(tts, "_usb_direct_speech_onset_gap_seconds", 0.0) or 0.0)
    return first_play_s + leadin + cushion + gap


def _wait_queue_join(q: Any, timeout: float) -> bool:
    done = threading.Event()

    def wait_join() -> None:
        q.join()
        done.set()

    thread = threading.Thread(target=wait_join, daemon=True)
    thread.start()
    return done.wait(timeout=timeout)


def run_trial(args: argparse.Namespace, cfg: dict[str, Any], trial_index: int) -> dict[str, Any]:
    mic = MicInput.from_config(cfg)
    voice_cfg = cfg.get("voice", {})
    try:
        usb_direct_expected = bool(mic._should_use_usb_direct())
    except Exception:
        usb_direct_expected = False
    tts_cfg = dict(cfg.get("voice", {}).get("tts", {}))
    if args.speech_leadin_seconds is not None:
        tts_cfg["usb_direct_speech_leadin_seconds"] = args.speech_leadin_seconds
    tts = TTSEngine(tts_cfg)
    play_events: list[dict[str, Any]] = []
    chunks: list[np.ndarray] = []
    errors: list[str] = []
    record_ready = threading.Event()
    record_started_at = [0.0]
    capture_transport = ["unknown"]

    def wrap_play(kind: str, func: Any) -> Any:
        def wrapped(chunk: np.ndarray) -> bool:
            start = time.perf_counter()
            event = {
                "kind": kind,
                "start_s": start - record_started_at[0],
                "samples": int(len(chunk)),
                "done_s": None,
                "ok": None,
            }
            try:
                ok = bool(func(chunk))
                event["ok"] = ok
                return ok
            finally:
                event["done_s"] = time.perf_counter() - record_started_at[0]
                play_events.append(event)

        return wrapped

    tts._play_chunk_usb_direct_speech = wrap_play(  # type: ignore[method-assign]
        "speech",
        tts._play_chunk_usb_direct_speech,
    )
    tts._play_chunk_usb_direct_with_preroll = wrap_play(  # type: ignore[method-assign]
        "feedback",
        tts._play_chunk_usb_direct_with_preroll,
    )
    tts._play_chunk_usb_direct_locked = wrap_play(  # type: ignore[method-assign]
        "usb_locked",
        tts._play_chunk_usb_direct_locked,
    )

    if not args.live_tts:
        tts.speak(args.text)
        if not _wait_queue_join(tts.tts_text_queue, timeout=args.tts_timeout):
            tts.shutdown()
            raise TimeoutError(f"TTS prebuffer timed out after {args.tts_timeout:.1f}s")

    def record() -> None:
        try:
            with mic.open():
                record_started_at[0] = time.perf_counter()
                capture_transport[0] = (
                    "usb_direct" if getattr(mic, "_usb_audio_proc", None) is not None else "sounddevice"
                )
                record_ready.set()
                chunks_needed = max(1, int(args.record_seconds * 1000 / mic._chunk_ms))
                for _ in range(chunks_needed):
                    chunks.append(mic.read_chunk())
        except Exception as exc:  # pragma: no cover - hardware path
            errors.append(str(exc))
            record_ready.set()

    recorder = threading.Thread(target=record)
    recorder.start()
    if not record_ready.wait(timeout=5.0):
        tts.shutdown()
        raise TimeoutError("microphone did not become ready")
    if errors:
        tts.shutdown()
        raise RuntimeError(errors[0])

    time.sleep(max(0.0, args.settle_ms / 1000.0))
    playback_requested_s = time.perf_counter() - record_started_at[0]
    if args.live_tts:
        tts.speak(args.text)
    tts.start_playback()
    tts.wait_done(timeout=args.playback_timeout)
    tts.stop_playback()
    recorder.join(timeout=args.record_seconds + 5)
    tts.shutdown()

    audio = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
    sample_rate = int(mic.sample_rate)
    wav_path = Path(args.wav_out)
    if args.trials > 1:
        wav_path = wav_path.with_name(f"{wav_path.stem}_trial{trial_index}{wav_path.suffix}")
    write_wav(wav_path, audio, sample_rate)

    play_events.sort(key=lambda event: float(event["start_s"]))
    first_play_s = min((float(event["start_s"]) for event in play_events), default=playback_requested_s)
    speech_events = [event for event in play_events if event.get("kind") == "speech"]
    first_speech_play_s = min(
        (float(event["start_s"]) for event in speech_events),
        default=first_play_s,
    )
    asr_audio = audio
    asr_audio_start_s = 0.0
    if not args.no_asr_trim and speech_events:
        asr_audio, asr_audio_start_s = slice_asr_samples(
            audio,
            sample_rate,
            estimate_speech_start_s(first_speech_play_s, tts),
        )

    backend = "cloud" if args.cloud_asr else args.asr_backend
    local_transcript = ""
    cloud_transcript = ""
    local_asr_ms = 0.0
    cloud_asr_ms = 0.0
    cloud_error = ""
    if backend in ("local", "both"):
        local_transcript, local_asr_ms = transcribe_local(asr_audio, sample_rate, cfg)
    if backend in ("cloud", "both"):
        cloud_transcript, cloud_asr_ms, cloud_error = transcribe_cloud(
            asr_audio,
            sample_rate,
            cfg,
            timeout=args.cloud_finish_timeout,
            realtime=not args.cloud_feed_fast,
        )
        if cloud_error:
            errors.append(cloud_error)
    asr_source = "cloud" if backend in ("cloud", "both") else "local"
    transcript = cloud_transcript if asr_source == "cloud" else local_transcript

    onset_ms, onset_threshold = detect_onset_ms(
        audio,
        sample_rate,
        first_play_s,
        min_peak=args.min_peak,
    )
    peak, rms = peak_rms(audio)
    signal_ok = (
        peak >= args.min_peak
        and onset_ms is not None
        and (args.max_onset_ms <= 0 or onset_ms <= args.max_onset_ms)
    )

    return {
        "trial": trial_index,
        "text": args.text,
        "expect_prefix": args.expect_prefix,
        "normalized_transcript": normalize_transcript(transcript),
        "transcript": transcript,
        "asr_source": asr_source,
        "asr_backend": backend,
        "local_transcript": local_transcript,
        "cloud_transcript": cloud_transcript,
        "prefix_ok": transcript_has_prefix(transcript, args.expect_prefix),
        "signal_ok": signal_ok,
        "peak": peak,
        "rms": rms,
        "min_peak": args.min_peak,
        "max_onset_ms": args.max_onset_ms,
        "speech_leadin_seconds": args.speech_leadin_seconds,
        "asr_audio_start_s": round(asr_audio_start_s, 3),
        "asr_audio_seconds": round(len(asr_audio) / float(sample_rate or 1), 3),
        "sample_rate": sample_rate,
        "input_transport_config": str(voice_cfg.get("input_transport", "auto")),
        "input_device_config": voice_cfg.get("input_device", None),
        "input_transport_resolved": capture_transport[0],
        "usb_direct_expected": usb_direct_expected,
        "record_seconds": args.record_seconds,
        "playback_requested_s": round(playback_requested_s, 3),
        "onset_ms": None if onset_ms is None else round(onset_ms, 1),
        "onset_threshold_peak": onset_threshold,
        "asr_ms": round(cloud_asr_ms if asr_source == "cloud" else local_asr_ms, 1),
        "local_asr_ms": round(local_asr_ms, 1),
        "cloud_asr_ms": round(cloud_asr_ms, 1),
        "play_events": play_events,
        "wav": str(wav_path),
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="\u4e00\u4e8c\u4e09\u56db\u4e94\uff0c\u97f3\u9891\u54e8\u5175\u6d4b\u8bd5\u3002")
    parser.add_argument("--expect-prefix", default="\u4e00")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--record-seconds", type=float, default=10.0)
    parser.add_argument("--settle-ms", type=float, default=500.0)
    parser.add_argument("--min-peak", type=int, default=300)
    parser.add_argument("--max-onset-ms", type=float, default=3000.0)
    parser.add_argument("--tts-timeout", type=float, default=45.0)
    parser.add_argument("--playback-timeout", type=float, default=45.0)
    parser.add_argument(
        "--asr-backend",
        choices=("local", "cloud", "both"),
        default="local",
        help="ASR backend used for the pass/fail transcript gate",
    )
    parser.add_argument(
        "--cloud-asr",
        action="store_true",
        help="compatibility alias for --asr-backend cloud",
    )
    parser.add_argument("--cloud-finish-timeout", type=float, default=8.0)
    parser.add_argument(
        "--cloud-feed-fast",
        action="store_true",
        help="feed CloudASR as fast as possible instead of pacing the stream in realtime",
    )
    parser.add_argument(
        "--no-asr-trim",
        action="store_true",
        help="transcribe the full recording instead of trimming USB wake audio",
    )
    parser.add_argument(
        "--speech-leadin-seconds",
        type=float,
        default=None,
        help="override voice.tts.usb_direct_speech_leadin_seconds for calibration",
    )
    parser.add_argument("--live-tts", action="store_true", help="include TTS generation latency in the recording window")
    parser.add_argument("--wav-out", default="/tmp/sunrise_audio_sentinel.wav")
    parser.add_argument("--json-out", default="/tmp/sunrise_audio_sentinel.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cloud_asr:
        args.asr_backend = "cloud"
    cfg = get_config()
    trial_results = []
    for trial_index in range(1, max(1, args.trials) + 1):
        print(f"=== Sunrise audio sentinel trial {trial_index}/{args.trials} ===", flush=True)
        result = run_trial(args, cfg, trial_index)
        trial_results.append(result)
        print(
            "  peak={peak} rms={rms} onset_ms={onset} asr={source} transcript={text!r} prefix_ok={prefix} signal_ok={signal}".format(
                peak=result["peak"],
                rms=result["rms"],
                onset=result["onset_ms"],
                source=result["asr_source"],
                text=result["transcript"],
                prefix=result["prefix_ok"],
                signal=result["signal_ok"],
            ),
            flush=True,
        )

    summary = summarize_trials(trial_results)
    payload = {"summary": summary, "trials": trial_results}
    json_path = Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved JSON: {json_path}", flush=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
