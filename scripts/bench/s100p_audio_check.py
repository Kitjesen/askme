#!/usr/bin/env python3
"""S100P audio check.

The normal product path uses ALSA/PortAudio through sounddevice.  When ALSA has
no cards, this script can still probe the Lenovo MCP01 USB Audio endpoints via
libusb so hardware health is not confused with kernel module health.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover - depends on target host packages
    sd = None  # type: ignore[assignment]
    SOUNDDEVICE_IMPORT_ERROR = exc
else:
    SOUNDDEVICE_IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parents[2]
USB_PROBE_SOURCE = Path(__file__).with_name("mcp01_usb_audio_libusb.c")
USB_PROBE_BINARY = Path(tempfile.gettempdir()) / "mcp01_usb_audio_libusb"
USB_CAPTURE_RE = re.compile(r"\bcapture_done\b.*?\brms=([0-9]+(?:\.[0-9]+)?)\s+max_abs=(\d+)")


def run(cmd: list[str], *, timeout: int = 15) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def print_result(label: str, result: subprocess.CompletedProcess[str]) -> None:
    status = "OK" if result.returncode == 0 else f"FAIL rc={result.returncode}"
    print(f"  {label}: {status}", flush=True)
    output = (result.stdout + result.stderr).strip()
    if output:
        for line in output.splitlines()[-8:]:
            print(f"    {line}", flush=True)


def parse_usb_capture_metrics(output: str) -> tuple[float, int] | None:
    match = USB_CAPTURE_RE.search(output)
    if match is None:
        return None
    return float(match.group(1)), int(match.group(2))


def usb_capture_signal_ok(
    result: subprocess.CompletedProcess[str],
    *,
    min_peak: int,
    min_rms: float,
) -> tuple[bool, tuple[float, int] | None]:
    metrics = parse_usb_capture_metrics(result.stdout + "\n" + result.stderr)
    if result.returncode != 0 or metrics is None:
        return False, metrics
    rms, peak = metrics
    return rms >= min_rms and peak >= min_peak, metrics


def mcp01_present() -> bool:
    if shutil.which("lsusb") is None:
        return False
    result = run(["lsusb"], timeout=5)
    return "17ef:a03b" in result.stdout.lower()


def list_audio_devices() -> tuple[bool, int]:
    print("=== Audio Devices (ALSA/PortAudio) ===", flush=True)
    if sd is None:
        print(f"  sounddevice import error: {SOUNDDEVICE_IMPORT_ERROR}", flush=True)
        return False, 0

    try:
        devs = sd.query_devices()
    except Exception as exc:
        print(f"  query_devices error: {exc}", flush=True)
        return False, 0

    visible = 0
    for i, dev in enumerate(devs):
        inp = int(dev["max_input_channels"])
        out = int(dev["max_output_channels"])
        if inp > 0 or out > 0:
            visible += 1
            print(
                f"  [{i}] {dev['name']} in={inp}ch out={out}ch "
                f"rate={dev['default_samplerate']}",
                flush=True,
            )
    if visible == 0:
        print("  no ALSA/PortAudio devices", flush=True)
    return visible > 0, visible


def record_device(label: str, *, device: int, samplerate: int, channels: int, seconds: int) -> bool:
    print(f"\n=== {label} Recording Test ({samplerate}Hz {channels}ch, {seconds}s) ===", flush=True)
    if sd is None:
        print("  skipped: sounddevice unavailable", flush=True)
        return False
    try:
        rec = sd.rec(
            int(seconds * samplerate),
            samplerate=samplerate,
            channels=channels,
            dtype="float32",
            device=device,
        )
        sd.wait()
        peaks = []
        for ch in range(channels):
            peak = float(np.max(np.abs(rec[:, ch] if channels > 1 else rec[:, 0])))
            rms = float(np.sqrt(np.mean((rec[:, ch] if channels > 1 else rec[:, 0]) ** 2)))
            peaks.append(int(peak * 32768))
            print(f"  ch{ch}: peak={peak:.4f}({int(peak * 32768)}) rms={rms:.4f}({int(rms * 32768)})", flush=True)
        max_peak = max(peaks) if peaks else 0
        print(
            f"  {label}: {'SIGNAL DETECTED' if max_peak > 50 else 'SILENT'} "
            f"(peak={max_peak})",
            flush=True,
        )
        return max_peak > 50
    except Exception as exc:
        print(f"  {label} error: {exc}", flush=True)
        return False


def write_beep(path: Path) -> None:
    sr = 48000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    tone = (np.sin(2 * np.pi * 800 * t) * 16000).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(tone.tobytes())


def speaker_test() -> bool:
    print("\n=== Speaker Test (ALSA aplay) ===", flush=True)
    if shutil.which("aplay") is None:
        print("  skipped: aplay not found", flush=True)
        return False

    wav_path = Path(tempfile.gettempdir()) / "askme_beep_test.wav"
    write_beep(wav_path)
    ok = False
    for dev, label in [("default", "default"), ("plughw:1,0", "MCP01 direct")]:
        try:
            result = run(["aplay", "-D", dev, str(wav_path)], timeout=5)
            print_result(label, result)
            ok = ok or result.returncode == 0
        except Exception as exc:
            print(f"  {label}: ERROR {exc}", flush=True)
    return ok


def compile_usb_probe() -> Path | None:
    if not USB_PROBE_SOURCE.exists():
        print(f"  skipped: missing {USB_PROBE_SOURCE}", flush=True)
        return None
    if shutil.which("gcc") is None or shutil.which("pkg-config") is None:
        print("  skipped: gcc or pkg-config not found", flush=True)
        return None

    cflags = run(["pkg-config", "--cflags", "libusb-1.0"], timeout=5)
    libs = run(["pkg-config", "--libs", "libusb-1.0"], timeout=5)
    if cflags.returncode != 0 or libs.returncode != 0:
        print("  skipped: libusb-1.0 development files not found", flush=True)
        print((cflags.stderr + libs.stderr).strip(), flush=True)
        return None

    cmd = [
        "gcc",
        "-O2",
        "-Wall",
        str(USB_PROBE_SOURCE),
        "-o",
        str(USB_PROBE_BINARY),
        *cflags.stdout.split(),
        *libs.stdout.split(),
        "-lm",
    ]
    result = run(cmd, timeout=20)
    if result.returncode != 0:
        print_result("compile MCP01 libusb probe", result)
        return None
    return USB_PROBE_BINARY


def direct_usb_probe(
    *,
    play_ms: int,
    capture_ms: int,
    amp: int,
    min_peak: int,
    min_rms: float,
) -> bool:
    print("\n=== MCP01 Direct USB Probe (libusb, bypass ALSA) ===", flush=True)
    if not mcp01_present():
        print("  skipped: MCP01 17ef:a03b not visible in lsusb", flush=True)
        return False

    binary = compile_usb_probe()
    if binary is None:
        return False

    cmd = [
        str(binary),
        "--play-ms",
        str(play_ms),
        "--capture-ms",
        str(capture_ms),
        "--amp",
        str(amp),
    ]
    result = run(cmd, timeout=max(10, (play_ms + capture_ms) // 1000 + 10))
    print_result("MCP01 direct USB transport", result)
    signal_ok, metrics = usb_capture_signal_ok(result, min_peak=min_peak, min_rms=min_rms)
    if metrics is None:
        print("  MCP01 direct USB capture signal: not verified (missing capture_done metrics)", flush=True)
    else:
        rms, peak = metrics
        status = "OK" if signal_ok else "FAIL"
        print(
            f"  MCP01 direct USB capture signal: {status} "
            f"(rms={rms:.2f}, peak={peak}, min_rms={min_rms:.2f}, min_peak={min_peak})",
            flush=True,
        )
    return signal_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--usb-probe",
        choices=("auto", "always", "never"),
        default="auto",
        help="Run the MCP01 libusb probe. auto runs it when ALSA has no devices.",
    )
    parser.add_argument("--play-ms", type=int, default=3000, help="Direct USB probe playback duration.")
    parser.add_argument("--capture-ms", type=int, default=3000, help="Direct USB probe capture duration.")
    parser.add_argument("--amp", type=int, default=9000, help="Direct USB probe sine amplitude.")
    parser.add_argument(
        "--usb-min-peak",
        type=int,
        default=1000,
        help="Minimum direct USB capture peak required to mark capture signal healthy.",
    )
    parser.add_argument(
        "--usb-min-rms",
        type=float,
        default=30.0,
        help="Minimum direct USB capture RMS required to mark capture signal healthy.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    alsa_available, _ = list_audio_devices()

    recording_ok = False
    speaker_ok = False
    if alsa_available:
        recording_ok = record_device("HKMIC", device=0, samplerate=48000, channels=2, seconds=3)
        record_device("MCP01", device=1, samplerate=16000, channels=1, seconds=2)
        speaker_ok = speaker_test()
    else:
        print("\nSkipping ALSA record/play tests because no sound cards are exposed.", flush=True)

    usb_ok = False
    should_usb_probe = args.usb_probe == "always" or (args.usb_probe == "auto" and not alsa_available)
    if should_usb_probe:
        usb_ok = direct_usb_probe(
            play_ms=args.play_ms,
            capture_ms=args.capture_ms,
            amp=args.amp,
            min_peak=args.usb_min_peak,
            min_rms=args.usb_min_rms,
        )

    print("\n=== Summary ===", flush=True)
    print(f"  ALSA/PortAudio devices: {'available' if alsa_available else 'unavailable'}", flush=True)
    print(f"  ALSA recording: {'ok' if recording_ok else 'not verified'}", flush=True)
    print(f"  ALSA speaker: {'ok' if speaker_ok else 'not verified'}", flush=True)
    print(
        f"  MCP01 direct USB capture signal: {'ok' if usb_ok else 'failed/not verified'}",
        flush=True,
    )

    if alsa_available:
        return 0 if (recording_ok or speaker_ok) else 1
    return 0 if usb_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
