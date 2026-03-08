"""ASR Engine - streaming speech recognition via sherpa-onnx OnlineRecognizer."""

from __future__ import annotations

import logging
import os
from typing import Any

import sherpa_onnx

logger = logging.getLogger(__name__)


class ASREngine:
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

        # Auto-detect int8 vs float32 model files
        encoder_name = config.get("encoder", "encoder-epoch-99-avg-1.int8.onnx")
        decoder_name = config.get("decoder", "decoder-epoch-99-avg-1.int8.onnx")
        joiner_name = config.get("joiner", "joiner-epoch-99-avg-1.int8.onnx")

        encoder = os.path.join(model_dir, encoder_name)
        decoder = os.path.join(model_dir, decoder_name)
        joiner = os.path.join(model_dir, joiner_name)

        # Fallback to float32 if int8 not found
        if not os.path.exists(encoder):
            encoder = os.path.join(model_dir, "encoder-epoch-99-avg-1.onnx")
            decoder = os.path.join(model_dir, "decoder-epoch-99-avg-1.onnx")
            joiner = os.path.join(model_dir, "joiner-epoch-99-avg-1.onnx")

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
