"""KWS Engine - Keyword Spotting (wake word detection) via sherpa-onnx."""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    import sherpa_onnx
except ModuleNotFoundError:
    sherpa_onnx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class KWSEngine:
    """Keyword Spotter backed by sherpa-onnx zipformer KWS models.

    Config dict expected keys (under voice.kws):
        model_dir: str       - path to the KWS model directory
        tokens: str          - tokens filename (default "tokens.txt")
        encoder: str         - encoder ONNX filename
        decoder: str         - decoder ONNX filename
        joiner: str          - joiner ONNX filename
        num_threads: int     - inference threads (default 1)
        keywords_file: str   - keywords filename (default "keywords.txt")
        keywords: list[str]  - keyword lines to write if keywords_file does not exist
    """

    def __init__(self, config: dict[str, Any]) -> None:
        if sherpa_onnx is None:
            self.spotter = None
            logger.warning("KWS unavailable — sherpa_onnx not installed")
            return

        model_dir: str = config.get(
            "model_dir",
            "models/kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01",
        )

        if not os.path.exists(model_dir):
            logger.warning("KWS model directory not found: %s, skipping.", model_dir)
            self.spotter = None
            return

        tokens = os.path.join(model_dir, config.get("tokens", "tokens.txt"))
        encoder = os.path.join(
            model_dir,
            config.get("encoder", "encoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
        )
        decoder = os.path.join(
            model_dir,
            config.get("decoder", "decoder-epoch-12-avg-2-chunk-16-left-64.onnx"),
        )
        joiner = os.path.join(
            model_dir,
            config.get("joiner", "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"),
        )

        keywords_file = os.path.join(
            model_dir,
            config.get("keywords_file", "keywords.txt"),
        )

        configured_keywords: list[str] = config.get("keywords", [])
        # Empty keywords list = skip KWS entirely (always-on listening)
        if not configured_keywords:
            logger.info("KWS disabled: no keywords configured (always-on mode).")
            self.spotter = None
            return

        # Configured keywords take precedence over any existing keywords file so
        # wake-word changes apply immediately without manual file edits.
        if configured_keywords:
            normalized_keywords = [
                self._normalize_keyword(str(kw))
                for kw in configured_keywords
                if str(kw).strip()
            ]
            with open(keywords_file, "w", encoding="utf-8") as f:
                for kw in normalized_keywords:
                    f.write(kw + "\n")
            configured_keywords = []
        elif not os.path.exists(keywords_file):
            with open(keywords_file, "w", encoding="utf-8") as f:
                for kw in (
                    self._normalize_keyword("雷霆"),
                    self._normalize_keyword("Thunder"),
                ):
                    f.write(kw + "\n")

        # Backward-compatible fallback path for older configs that expect
        # auto-generation only when the keywords file is missing.
        if configured_keywords:
            default_keywords: list[str] = config.get("keywords", [
                "\u4f60\u597d @\u4f60\u597d",   # 你好 @你好
                "\u5c0f\u667a @\u5c0f\u667a",   # 小智 @小智
            ])
            with open(keywords_file, "w", encoding="utf-8") as f:
                for kw in default_keywords:
                    f.write(kw + "\n")

        self.spotter = sherpa_onnx.KeywordSpotter(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=int(config.get("num_threads", 1)),
            keywords_file=keywords_file,
        )

        logger.info("KWS initialized.")

    @staticmethod
    def _normalize_keyword(keyword: str) -> str:
        keyword = keyword.strip()
        if not keyword:
            return ""
        if "@" in keyword:
            return keyword
        return f"{keyword} @{keyword}"

    @property
    def available(self) -> bool:
        """Return True if the keyword spotter was loaded successfully."""
        return self.spotter is not None

    def create_stream(self) -> sherpa_onnx.OnlineStream | None:
        """Create and return a new KWS stream, or None if spotter unavailable."""
        if self.spotter is None:
            return None
        return self.spotter.create_stream()
