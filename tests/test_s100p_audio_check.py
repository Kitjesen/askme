"""Tests for S100P audio hardware check helpers."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


def _load_audio_check():
    path = Path(__file__).resolve().parents[1] / "scripts" / "bench" / "s100p_audio_check.py"
    spec = importlib.util.spec_from_file_location("s100p_audio_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_usb_capture_metrics_reads_probe_summary() -> None:
    audio_check = _load_audio_check()

    assert audio_check.parse_usb_capture_metrics(
        "capture_done submitted_packets=3000 completed_packets=3000 "
        "samples=48000 rms=20.11 max_abs=439 errors=0 raw_errors=0"
    ) == (20.11, 439)


def test_usb_capture_signal_requires_returncode_and_capture_energy() -> None:
    audio_check = _load_audio_check()

    weak = subprocess.CompletedProcess(
        args=["probe"],
        returncode=0,
        stdout=(
            "capture_done submitted_packets=3000 completed_packets=3000 "
            "samples=48000 rms=20.11 max_abs=439 errors=0 raw_errors=0"
        ),
        stderr="",
    )
    strong = subprocess.CompletedProcess(
        args=["probe"],
        returncode=0,
        stdout=(
            "capture_done submitted_packets=3000 completed_packets=3000 "
            "samples=48000 rms=80.00 max_abs=14523 errors=0 raw_errors=0"
        ),
        stderr="",
    )
    failed = subprocess.CompletedProcess(args=["probe"], returncode=1, stdout=strong.stdout, stderr="")

    assert audio_check.usb_capture_signal_ok(weak, min_peak=1000, min_rms=30.0) == (
        False,
        (20.11, 439),
    )
    assert audio_check.usb_capture_signal_ok(strong, min_peak=1000, min_rms=30.0) == (
        True,
        (80.0, 14523),
    )
    assert audio_check.usb_capture_signal_ok(failed, min_peak=1000, min_rms=30.0) == (
        False,
        (80.0, 14523),
    )
