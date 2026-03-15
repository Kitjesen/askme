"""Unified audio preprocessing: HPF + noise reduction + noise gate + echo gate.

Wraps AudioFilter, SpectralSubtractor, and NoiseGateCalibrator into a single
pipeline.  Each feature is independently configurable.  Extracted from
audio_agent.py for independent testing and debugging.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .audio_filter import AudioFilter
from .noise_reduction import NoiseGateCalibrator, SpectralSubtractor

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Unified audio preprocessing: HPF + noise reduction + noise gate + echo gate.

    Each feature independently configurable.  Extracted from audio_agent.py
    for independent testing and debugging.

    Config keys (under ``voice``)::

        audio_filter:
            enabled: bool          (default True)
            highpass_freq: float   (default 80)
            mode: str              (default "highpass")
        noise_reduction:
            enabled: bool          (default False)
            alpha: float           (default 2.0)
            beta: float            (default 0.02)
            calibration_frames: int (default 20)
        noise_gate_peak: int | "auto"  (default 0, 0=disabled)
        echo_gate_peak: int            (default 0, 0=disabled)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}

        # --- Audio filter (DC removal + HPF / bandpass) ---
        filter_cfg = cfg.get("audio_filter")
        if filter_cfg and filter_cfg.get("enabled", True):
            self._filter = AudioFilter(filter_cfg)
        else:
            self._filter: AudioFilter | None = None

        # --- Spectral subtraction noise reduction ---
        nr_cfg = cfg.get("noise_reduction")
        if nr_cfg and nr_cfg.get("enabled", False):
            self._subtractor = SpectralSubtractor(nr_cfg)
        else:
            self._subtractor: SpectralSubtractor | None = None

        # --- Noise gate ---
        _raw_gate = cfg.get("noise_gate_peak", 0)
        self._auto_calibrate_gate: bool = str(_raw_gate).lower() == "auto"
        self._noise_gate_peak_val: int = 0 if self._auto_calibrate_gate else int(_raw_gate)
        self._noise_calibrator: NoiseGateCalibrator | None = None
        if self._auto_calibrate_gate:
            self._noise_calibrator = NoiseGateCalibrator()

        # --- Echo gate ---
        self._echo_gate_peak: int = int(cfg.get("echo_gate_peak", 0))

    # ------------------------------------------------------------------
    # Main processing entry point
    # ------------------------------------------------------------------

    def process(
        self,
        samples: np.ndarray,
        tts_active: bool = False,
        speech_active: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, int, bool]:
        """Process raw float32 mic samples through the filter chain.

        Args:
            samples: float32 audio samples from the microphone.
            tts_active: True while TTS is playing back audio.
            speech_active: True once VAD has confirmed ongoing speech.

        Returns:
            (filtered_float32, filtered_int16, peak, echo_gated)

            - *filtered_float32*: after HPF + noise reduction (float32).
            - *filtered_int16*: int16 version of the filtered audio.
            - *peak*: int16 peak amplitude of the filtered frame.
            - *echo_gated*: True if this frame was suppressed by the echo gate.
        """
        # Step 1: HPF / bandpass filter
        if self._filter is not None:
            samples = self._filter.process(samples)

        # Step 2: float32 -> int16 + peak
        samples_int16 = (samples * 32768).astype(np.int16)
        peak = int(np.max(np.abs(samples_int16)))

        # Step 3: Spectral subtraction (operates on int16, returns int16)
        if self._subtractor is not None and self._subtractor.calibrated:
            samples_int16 = self._subtractor.process(samples_int16)

        # Step 4: Echo gate — suppress low-energy mic during TTS playback.
        # Do NOT gate once speech_active=True (barge-in already confirmed).
        echo_gated = (
            tts_active
            and self._echo_gate_peak > 0
            and peak < self._echo_gate_peak
            and not speech_active
        )

        return samples, samples_int16, peak, echo_gated

    # ------------------------------------------------------------------
    # Calibration helpers
    # ------------------------------------------------------------------

    def feed_calibration(self, samples: np.ndarray) -> None:
        """Feed a silence frame for noise reduction calibration."""
        if self._subtractor is not None and not self._subtractor.calibrated:
            self._subtractor.feed_calibration(samples)

    def auto_calibrate_gate(self, samples_int16: np.ndarray) -> int | None:
        """Feed a frame for noise gate auto-calibration.

        Returns the recommended threshold when calibration is complete,
        or ``None`` if more data is needed (or auto-calibration is not active).
        """
        if self._noise_calibrator is None:
            return None

        result = self._noise_calibrator.feed(samples_int16)
        if result is not None:
            self._noise_gate_peak_val = result
            self._noise_calibrator = None
            logger.info("Noise gate auto-calibrated: %d", result)
        return result

    # ------------------------------------------------------------------
    # Noise gate
    # ------------------------------------------------------------------

    @property
    def noise_gate_peak(self) -> int:
        """Current noise gate threshold."""
        return self._noise_gate_peak_val

    @noise_gate_peak.setter
    def noise_gate_peak(self, value: int) -> None:
        self._noise_gate_peak_val = value

    def is_noise_gated(self, peak: int) -> bool:
        """Check if *peak* falls below the noise gate threshold.

        Returns ``True`` when the noise gate is enabled (>0) and *peak* is
        below the threshold — meaning this frame should be skipped for the
        ``speech_active`` transition.
        """
        return self._noise_gate_peak_val > 0 and peak < self._noise_gate_peak_val
