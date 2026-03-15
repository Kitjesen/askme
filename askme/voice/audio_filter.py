"""Audio signal filters for the voice pipeline.

Lightweight filters applied to raw microphone input before VAD/ASR.
All pure numpy — no external dependencies.

Filters:
    - DC offset removal (always on)
    - High-pass Butterworth (removes motor hum, HVAC, ground noise < 80Hz)
    - Optional band-pass (300-3400Hz, aggressive speech-only mode)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AudioFilter:
    """Cascaded audio filter chain for microphone input.

    Applied to raw float32 samples before peak calculation, noise gate, and VAD.
    Removes low-frequency noise (motor hum, HVAC) that causes false VAD triggers
    without affecting speech quality.

    Config keys (under ``voice.audio_filter``)::

        enabled: bool        - Enable/disable (default True)
        highpass_freq: float - High-pass cutoff in Hz (default 80)
        sample_rate: int     - Expected sample rate (default 16000)
        mode: str            - "highpass" (default) or "bandpass"
        bandpass_low: float  - Band-pass low cutoff (default 300, mode=bandpass only)
        bandpass_high: float - Band-pass high cutoff (default 3400, mode=bandpass only)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.enabled: bool = bool(cfg.get("enabled", True))
        self._mode: str = cfg.get("mode", "highpass")
        self._sample_rate: int = int(cfg.get("sample_rate", 16000))
        self._highpass_freq: float = float(cfg.get("highpass_freq", 80.0))
        self._bandpass_low: float = float(cfg.get("bandpass_low", 300.0))
        self._bandpass_high: float = float(cfg.get("bandpass_high", 3400.0))

        # Biquad filter state (2nd order IIR, cascaded sections)
        # Each filter stores [x1, x2, y1, y2] state
        self._hp_state = np.zeros(4, dtype=np.float64)
        self._hp_coeffs = self._butterworth_hp(self._highpass_freq, self._sample_rate)

        self._bp_lo_state = np.zeros(4, dtype=np.float64)
        self._bp_lo_coeffs = self._butterworth_hp(self._bandpass_low, self._sample_rate)
        self._bp_hi_state = np.zeros(4, dtype=np.float64)
        self._bp_hi_coeffs = self._butterworth_lp(self._bandpass_high, self._sample_rate)

        # DC removal state
        self._dc_alpha = 0.995  # time constant ~200ms at 16kHz
        self._dc_prev_in: float = 0.0
        self._dc_prev_out: float = 0.0

        if self.enabled:
            logger.info(
                "AudioFilter: enabled mode=%s hp=%dHz sr=%d",
                self._mode, int(self._highpass_freq), self._sample_rate,
            )

    @staticmethod
    def _butterworth_hp(fc: float, fs: float) -> tuple[float, ...]:
        """2nd order Butterworth high-pass biquad coefficients."""
        w0 = 2.0 * np.pi * fc / fs
        alpha = np.sin(w0) / (2.0 * np.sqrt(2.0))  # Q = sqrt(2)/2
        cos_w0 = np.cos(w0)
        b0 = (1.0 + cos_w0) / 2.0
        b1 = -(1.0 + cos_w0)
        b2 = (1.0 + cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
        return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)

    @staticmethod
    def _butterworth_lp(fc: float, fs: float) -> tuple[float, ...]:
        """2nd order Butterworth low-pass biquad coefficients."""
        w0 = 2.0 * np.pi * fc / fs
        alpha = np.sin(w0) / (2.0 * np.sqrt(2.0))
        cos_w0 = np.cos(w0)
        b0 = (1.0 - cos_w0) / 2.0
        b1 = 1.0 - cos_w0
        b2 = (1.0 - cos_w0) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
        return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)

    def _apply_biquad(
        self, samples: np.ndarray, coeffs: tuple[float, ...], state: np.ndarray
    ) -> np.ndarray:
        """Apply a biquad IIR filter with state preservation across calls."""
        b0, b1, b2, a1, a2 = coeffs
        x1, x2, y1, y2 = state
        out = np.empty_like(samples, dtype=np.float64)
        for i in range(len(samples)):
            x0 = float(samples[i])
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            out[i] = y0
            x2, x1 = x1, x0
            y2, y1 = y1, y0
        state[:] = [x1, x2, y1, y2]
        return out.astype(samples.dtype)

    def _remove_dc(self, samples: np.ndarray) -> np.ndarray:
        """Remove DC offset using a simple 1-pole high-pass."""
        alpha = self._dc_alpha
        out = np.empty_like(samples)
        prev_in = self._dc_prev_in
        prev_out = self._dc_prev_out
        for i in range(len(samples)):
            x = float(samples[i])
            y = x - prev_in + alpha * prev_out
            out[i] = y
            prev_in = x
            prev_out = y
        self._dc_prev_in = prev_in
        self._dc_prev_out = prev_out
        return out

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Apply the filter chain to float32 audio samples.

        Returns filtered samples. If disabled, returns input unchanged.
        """
        if not self.enabled:
            return samples

        # Step 1: DC offset removal (always)
        filtered = self._remove_dc(samples)

        # Step 2: Mode-dependent filtering
        if self._mode == "bandpass":
            filtered = self._apply_biquad(filtered, self._bp_lo_coeffs, self._bp_lo_state)
            filtered = self._apply_biquad(filtered, self._bp_hi_coeffs, self._bp_hi_state)
        else:
            # Default: high-pass only
            filtered = self._apply_biquad(filtered, self._hp_coeffs, self._hp_state)

        return filtered.astype(np.float32)

    def reset(self) -> None:
        """Reset all filter states."""
        self._hp_state[:] = 0
        self._bp_lo_state[:] = 0
        self._bp_hi_state[:] = 0
        self._dc_prev_in = 0.0
        self._dc_prev_out = 0.0
