"""VAD Engine - Voice Activity Detection via sherpa-onnx Silero VAD."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import sherpa_onnx
except ModuleNotFoundError:
    class _SherpaOnnxStub:
        VadModelConfig = None
        SileroVadModelConfig = None
        VoiceActivityDetector = None
    sherpa_onnx = _SherpaOnnxStub()  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class VADEngine:
    """Voice Activity Detector backed by sherpa-onnx Silero VAD.

    Config dict expected keys (under voice.vad):
        model_path: str             - path to silero_vad.onnx (``model`` is a legacy alias)
        threshold: float            - detection threshold (default 0.5)
        min_silence_duration: float - min silence to end speech segment (default 0.5)
        min_speech_duration: float  - min speech to start segment (default 0.25)
        sample_rate: int            - audio sample rate (default 16000)
        buffer_size_in_seconds: int - ring buffer size (default 30)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        if sherpa_onnx.VadModelConfig is None:
            self.detector = None
            logger.warning("VAD unavailable — sherpa_onnx not installed")
            return

        vad_config = sherpa_onnx.VadModelConfig()
        model_path = (
            config.get("model")
            or config.get("model_path")
            or "models/vad/silero_vad.onnx"
        )
        vad_config.silero_vad.model = str(model_path)
        vad_config.silero_vad.threshold = float(config.get("threshold", 0.5))
        vad_config.silero_vad.min_silence_duration = float(config.get("min_silence_duration", 0.5))
        vad_config.silero_vad.min_speech_duration = float(config.get("min_speech_duration", 0.25))
        vad_config.sample_rate = int(config.get("sample_rate", 16000))
        vad_config.provider = str(config.get("provider", "cpu"))
        vad_config.num_threads = int(config.get("num_threads", 1))

        buffer_size = int(config.get("buffer_size_in_seconds", 30))
        self.detector = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=buffer_size)

        logger.info("VAD initialized.")

    def accept_waveform(self, samples_int16: np.ndarray) -> None:
        """Normalize PCM16 samples to sherpa-onnx's float32 waveform contract."""
        waveform = np.asarray(samples_int16, dtype=np.float32) / 32768.0
        self.detector.accept_waveform(waveform)

    def is_speech_detected(self) -> bool:
        """Return True if speech is currently detected."""
        return self.detector.is_speech_detected()
