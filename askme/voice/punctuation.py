"""Punctuation restoration for ASR output via sherpa-onnx CT-Transformer."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class PunctuationRestorer:
    """Add punctuation to raw ASR text using a CT-Transformer model.

    Config dict expected keys (under voice.punctuation):
        model_path: str   - path to the CT-Transformer ONNX model file
        num_threads: int   - inference threads (default 2)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        model_path: str = config.get(
            "model_path",
            "models/punct/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx",
        )

        if not os.path.exists(model_path):
            logger.info("Punctuation model not found at %s, disabled.", model_path)
            self._punct = None
            return

        try:
            import sherpa_onnx

            punct_config = sherpa_onnx.OfflinePunctuationConfig(
                model=sherpa_onnx.OfflinePunctuationModelConfig(
                    ct_transformer=model_path,
                    num_threads=int(config.get("num_threads", 2)),
                )
            )
            self._punct = sherpa_onnx.OfflinePunctuation(punct_config)
            logger.info("Punctuation restorer initialized.")
        except Exception as exc:
            logger.warning("Failed to load punctuation model: %s", exc)
            self._punct = None

    @property
    def available(self) -> bool:
        return self._punct is not None

    def restore(self, text: str) -> str:
        """Add punctuation to *text*. Returns original text if model unavailable."""
        if not self._punct or not text.strip():
            return text
        try:
            return self._punct.add_punctuation(text.strip())
        except Exception:
            return text
