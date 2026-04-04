"""VAD Engine - Voice Activity Detection via sherpa-onnx Silero VAD."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
try:
    import sherpa_onnx
except ModuleNotFoundError:
    sherpa_onnx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class VADEngine:
    """Voice Activity Detector backed by sherpa-onnx Silero VAD.

    Config dict expected keys (under voice.vad):
        model: str                  - path to silero_vad.onnx (default "models/vad/silero_vad.onnx")
        threshold: float            - detection threshold (default 0.5)
        min_silence_duration: float - min silence to end speech segment (default 0.5)
        min_speech_duration: float  - min speech to start segment (default 0.25)
        sample_rate: int            - audio sample rate (default 16000)
        buffer_size_in_seconds: int - ring buffer size (default 30)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = config.get("model", "models/vad/silero_vad.onnx")
        vad_config.silero_vad.threshold = float(config.get("threshold", 0.5))
        vad_config.silero_vad.min_silence_duration = float(config.get("min_silence_duration", 0.5))
        vad_config.silero_vad.min_speech_duration = float(config.get("min_speech_duration", 0.25))
        vad_config.sample_rate = int(config.get("sample_rate", 16000))

        buffer_size = int(config.get("buffer_size_in_seconds", 30))
        self.detector = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=buffer_size)

        logger.info("VAD initialized.")

    def accept_waveform(self, samples_int16: np.ndarray) -> None:
        """Feed int16 audio samples to the VAD."""
        self.detector.accept_waveform(samples_int16)

    def is_speech_detected(self) -> bool:
        """Return True if speech is currently detected."""
        return self.detector.is_speech_detected()
