"""ASR Engine - streaming speech recognition via sherpa-onnx OnlineRecognizer."""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    import sherpa_onnx
except ModuleNotFoundError:
    sherpa_onnx = None  # type: ignore[assignment]

from askme.interfaces.asr import ASRBackend

logger = logging.getLogger(__name__)


class ASREngine(ASRBackend):
    """Automatic Speech Recognition engine backed by sherpa-onnx transducer models.

    Config dict expected keys (under voice.asr):
        model_dir: str                  - path to the ASR model directory
        tokens: str                     - relative filename for tokens (default "tokens.txt")
        encoder: str                    - encoder ONNX filename (auto-detects int8 vs float32)
        decoder: str                    - decoder ONNX filename
        joiner: str                     - joiner ONNX filename
        num_threads: int                - inference threads (default 1)
        sample_rate: int                - audio sample rate (default 16000)
        feature_dim: int                - feature dimension (default 80)
        enable_endpoint_detection: bool - enable endpoint detection (default True)
        rule1_min_trailing_silence: float - endpoint rule1 (default 2.4)
        rule2_min_trailing_silence: float - endpoint rule2 (default 1.2)
        rule3_min_utterance_length: float - endpoint rule3 (default inf)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        model_dir: str = config.get(
            "model_dir",
            "models/asr/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
        )

        tokens = os.path.join(model_dir, config.get("tokens", "tokens.txt"))

        # Auto-detect model files: try config overrides first, then common names
        encoder = self._find_model_file(model_dir, config.get("encoder"), [
            "encoder.int8.onnx",
            "encoder-epoch-99-avg-1.int8.onnx",
            "encoder.onnx",
            "encoder-epoch-99-avg-1.onnx",
        ])
        decoder = self._find_model_file(model_dir, config.get("decoder"), [
            "decoder.onnx",
            "decoder-epoch-99-avg-1.int8.onnx",
            "decoder-epoch-99-avg-1.onnx",
        ])
        joiner = self._find_model_file(model_dir, config.get("joiner"), [
            "joiner.int8.onnx",
            "joiner-epoch-99-avg-1.int8.onnx",
            "joiner.onnx",
            "joiner-epoch-99-avg-1.onnx",
        ])

        self.sample_rate: int = int(config.get("sample_rate", 16000))

        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=int(config.get("num_threads", 1)),
            sample_rate=self.sample_rate,
            feature_dim=int(config.get("feature_dim", 80)),
            enable_endpoint_detection=config.get("enable_endpoint_detection", True),
            rule1_min_trailing_silence=float(config.get("rule1_min_trailing_silence", 2.4)),
            rule2_min_trailing_silence=float(config.get("rule2_min_trailing_silence", 1.2)),
            rule3_min_utterance_length=float(config.get("rule3_min_utterance_length", float("inf"))),
        )

        logger.info("ASR initialized.")

    @staticmethod
    def _find_model_file(
        model_dir: str,
        config_override: str | None,
        candidates: list[str],
    ) -> str:
        """Return the first existing model file from config override or candidates."""
        if config_override:
            return os.path.join(model_dir, config_override)
        for name in candidates:
            path = os.path.join(model_dir, name)
            if os.path.exists(path):
                return path
        # Fallback to first candidate (will fail at init with clear error)
        return os.path.join(model_dir, candidates[0])

    def create_stream(self) -> sherpa_onnx.OnlineStream:
        """Create and return a new ASR stream."""
        return self.recognizer.create_stream()

    def is_ready(self, stream: sherpa_onnx.OnlineStream) -> bool:
        """Check if the recognizer is ready to decode the given stream."""
        return self.recognizer.is_ready(stream)

    def decode_stream(self, stream: sherpa_onnx.OnlineStream) -> None:
        """Decode one step on the given stream."""
        self.recognizer.decode_stream(stream)

    def is_endpoint(self, stream: sherpa_onnx.OnlineStream) -> bool:
        """Check if an endpoint has been detected in the stream."""
        return self.recognizer.is_endpoint(stream)

    def get_result(self, stream: sherpa_onnx.OnlineStream) -> str:
        """Get the current recognition result text from the stream."""
        return self.recognizer.get_result(stream)

    def reset(self, stream: sherpa_onnx.OnlineStream) -> None:
        """Reset the given stream."""
        self.recognizer.reset(stream)
