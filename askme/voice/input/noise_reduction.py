"""Audio noise reduction via spectral subtraction.

Pure numpy implementation — no external dependencies. Estimates the noise
floor from silence segments and subtracts it from speech, reducing
steady-state noise (fan hum, USB mic noise floor, electrical interference).

For industrial environments, this is a stepping stone. The long-term path
is hardware AEC + RNNoise / Krisp neural denoiser.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SpectralSubtractor:
    """Frequency-domain noise reduction using spectral subtraction.

    1. During calibration, collect noise spectrum from ambient silence.
    2. During speech, subtract the noise spectrum from each frame.
    3. Use spectral flooring to prevent musical noise artifacts.

    Config keys (under ``voice.noise_reduction``)::

        enabled: bool       - Enable/disable (default False)
        calibration_frames: int  - Frames for noise estimation (default 20)
        alpha: float        - Subtraction strength (default 2.0)
        beta: float         - Spectral floor (default 0.02)
        frame_size: int     - FFT frame size in samples (default 512)
        hop_size: int       - Frame hop in samples (default 256)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.enabled: bool = bool(cfg.get("enabled", False))
        self._calibration_frames: int = int(cfg.get("calibration_frames", 20))
        self._alpha: float = float(cfg.get("alpha", 2.0))
        self._beta: float = float(cfg.get("beta", 0.02))
        self._frame_size: int = int(cfg.get("frame_size", 512))
        self._hop_size: int = int(cfg.get("hop_size", 256))

        # Noise profile (estimated during calibration)
        self._noise_spectrum: np.ndarray | None = None
        self._calibration_buffer: list[np.ndarray] = []
        self._calibrated: bool = False

        # Hann window for STFT (pre-computed)
        self._window = np.hanning(self._frame_size).astype(np.float32)

        if self.enabled:
            logger.info(
                "SpectralSubtractor: enabled (alpha=%.1f, beta=%.3f, frame=%d)",
                self._alpha,
                self._beta,
                self._frame_size,
            )

    @property
    def calibrated(self) -> bool:
        """Whether noise profile has been estimated."""
        return self._calibrated

    def feed_calibration(self, samples: np.ndarray) -> bool:
        """Feed a silence/noise frame for calibration.

        Returns True when enough frames have been collected and the noise
        profile is ready.
        """
        if self._calibrated:
            return True
        if not self.enabled:
            return False

        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0

        self._calibration_buffer.append(samples.copy())
        if len(self._calibration_buffer) >= self._calibration_frames:
            self._estimate_noise_profile()
            return True
        return False

    def _estimate_noise_profile(self) -> None:
        """Compute average noise magnitude spectrum from calibration frames."""
        all_samples = np.concatenate(self._calibration_buffer)
        n_frames = (len(all_samples) - self._frame_size) // self._hop_size + 1

        if n_frames <= 0:
            logger.warning("SpectralSubtractor: not enough calibration data")
            self._calibrated = True
            return

        spectra: list[np.ndarray] = []
        for i in range(n_frames):
            start = i * self._hop_size
            frame = all_samples[start : start + self._frame_size] * self._window
            spectrum = np.abs(np.fft.rfft(frame))
            spectra.append(spectrum)

        self._noise_spectrum = np.mean(spectra, axis=0).astype(np.float32)
        self._calibrated = True
        self._calibration_buffer.clear()

        noise_db = 20 * np.log10(np.mean(self._noise_spectrum) + 1e-10)
        logger.info(
            "SpectralSubtractor: calibrated (noise floor: %.1f dB, %d frames)",
            noise_db,
            len(spectra),
        )

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction to audio samples.

        Returns cleaned audio. If not calibrated or disabled, returns
        the input unchanged.
        """
        if not self.enabled or not self._calibrated or self._noise_spectrum is None:
            return samples

        is_int16 = samples.dtype == np.int16
        if is_int16:
            audio = samples.astype(np.float32) / 32768.0
        else:
            audio = samples.astype(np.float32)

        n_frames = (len(audio) - self._frame_size) // self._hop_size + 1
        if n_frames <= 0:
            return samples

        output = np.zeros(len(audio), dtype=np.float32)
        window_sum = np.zeros(len(audio), dtype=np.float32)

        for i in range(n_frames):
            start = i * self._hop_size
            end = start + self._frame_size
            frame = audio[start:end] * self._window

            # Forward FFT
            spectrum = np.fft.rfft(frame)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)

            # Spectral subtraction with flooring
            clean_mag = magnitude - self._alpha * self._noise_spectrum
            floor = self._beta * magnitude
            clean_mag = np.maximum(clean_mag, floor)

            # Inverse FFT with original phase
            clean_spectrum = clean_mag * np.exp(1j * phase)
            clean_frame = np.fft.irfft(clean_spectrum, n=self._frame_size)

            # Overlap-add
            output[start:end] += clean_frame.astype(np.float32) * self._window
            window_sum[start:end] += self._window**2

        # Normalize by window overlap
        nonzero = window_sum > 1e-8
        output[nonzero] /= window_sum[nonzero]
        # Copy untouched tail
        tail_start = n_frames * self._hop_size
        if tail_start < len(audio):
            output[tail_start:] = audio[tail_start:]

        if is_int16:
            return (output * 32768.0).clip(-32768, 32767).astype(np.int16)
        return output

    def reset(self) -> None:
        """Reset calibration state."""
        self._noise_spectrum = None
        self._calibration_buffer.clear()
        self._calibrated = False


class NoiseGateCalibrator:
    """Auto-calibrate noise_gate_peak from ambient mic samples.

    Collects N frames of audio, computes a peak threshold at
    ``mean + sigma_factor * std``, and returns a recommended
    noise_gate_peak value.

    Usage::

        calibrator = NoiseGateCalibrator()
        for chunk in mic_chunks:
            result = calibrator.feed(chunk)
            if result is not None:
                noise_gate_peak = result
                break
    """

    def __init__(self, num_frames: int = 20, sigma_factor: float = 2.5) -> None:
        self._num_frames = num_frames
        self._sigma_factor = sigma_factor
        self._peaks: list[int] = []

    def feed(self, samples_int16: np.ndarray) -> int | None:
        """Feed a chunk of int16 audio samples.

        Returns the recommended noise_gate_peak when enough samples
        have been collected, or None if more data is needed.
        """
        peak = int(np.max(np.abs(samples_int16)))
        self._peaks.append(peak)

        if len(self._peaks) < self._num_frames:
            return None

        arr = np.array(self._peaks, dtype=np.float64)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        threshold = int(mean_val + self._sigma_factor * std_val)

        # Clamp to reasonable range
        threshold = max(100, min(threshold, 2000))

        logger.info(
            "NoiseGateCalibrator: mean_peak=%.0f std=%.0f → threshold=%d",
            mean_val,
            std_val,
            threshold,
        )
        return threshold

    def reset(self) -> None:
        self._peaks.clear()
