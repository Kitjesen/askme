"""Local audio adapter used by the Voice Lab setup steps.

The adapter deliberately stops at device diagnostics and microphone noise
calibration. It does not claim to provide the isolated speaker monitor needed
for product-grade overlap-stop evidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from askme.voice.diagnostics.audio_devices import query_audio_devices, run_audio_loopback
from askme.voice.diagnostics.hardware_audio_capture import (
    HardwareAudioCaptureError,
    calibrate_noise_floor,
)


class VoiceLabAudioBackend(Protocol):
    """Small hardware boundary so HTTP/state tests never open real devices."""

    def inventory(self) -> dict[str, Any]: ...

    def capabilities(self) -> dict[str, bool]: ...

    def run_device_check(
        self,
        *,
        run_dir: Path,
        device_binding: dict[str, Any],
    ) -> dict[str, Any]: ...

    def calibrate_microphone(
        self,
        *,
        device_binding: dict[str, Any],
        duration_s: float,
    ) -> dict[str, Any]: ...


class SoundDeviceVoiceLabBackend:
    """Windows-friendly diagnostics backed by the existing sounddevice path."""

    def inventory(self) -> dict[str, Any]:
        return query_audio_devices()

    def capabilities(self) -> dict[str, bool]:
        return {
            "automatic_device_check": True,
            "automatic_microphone_calibration": True,
            "physical_first_sound_collector": False,
            "physical_overlap_stop_collector": False,
            "render_loopback_collector": False,
        }

    def run_device_check(
        self,
        *,
        run_dir: Path,
        device_binding: dict[str, Any],
    ) -> dict[str, Any]:
        # Keep the run directory in the boundary for future opt-in artifacts,
        # but do not persist room audio during the default diagnostic check.
        del run_dir
        return run_audio_loopback(
            input_device=device_binding["input_device_id"],
            output_device=device_binding["output_device_id"],
            sample_rate=int(device_binding["input_sample_rate_hz"]),
            record_seconds=2.0,
            tone_seconds=0.8,
            frequency_hz=880.0,
            output_gain=0.20,
            wav_out=None,
            play_recording=False,
        )

    def calibrate_microphone(
        self,
        *,
        device_binding: dict[str, Any],
        duration_s: float,
    ) -> dict[str, Any]:
        try:
            import sounddevice as sd
        except ModuleNotFoundError as exc:  # pragma: no cover - installation dependent
            raise HardwareAudioCaptureError("sounddevice is not installed") from exc

        sample_rate = int(device_binding["input_sample_rate_hz"])
        frame_count = max(1, int(sample_rate * duration_s))
        try:
            samples = sd.rec(
                frame_count,
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=device_binding["input_device_id"],
                blocking=True,
            )
        except Exception as exc:  # pragma: no cover - target hardware path
            raise HardwareAudioCaptureError(
                f"microphone calibration capture failed: {exc.__class__.__name__}"
            ) from exc

        mono = np.asarray(samples, dtype=np.float32).reshape(-1)
        chunk_size = max(1, int(sample_rate * 0.02))
        frames = [mono[index : index + chunk_size] for index in range(0, len(mono), chunk_size)]
        calibration = calibrate_noise_floor(
            frames,
            sample_rate_hz=sample_rate,
            source_label="microphone",
            margin_db=12.0,
        )
        return {
            "status": "ok",
            "duration_s": round(float(duration_s), 3),
            "calibration": calibration.to_dict(),
        }


__all__ = ["SoundDeviceVoiceLabBackend", "VoiceLabAudioBackend"]
