"""KWS Engine - Keyword Spotting (wake word detection) via sherpa-onnx."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

try:
    import sherpa_onnx
except ModuleNotFoundError:
    class _SherpaOnnxStub:
        KeywordSpotter = None
        OnlineStream = None
    sherpa_onnx = _SherpaOnnxStub()  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def normalize_keyword_line(keyword: str) -> str:
    """Return a sherpa keyword line while preserving pre-tokenized input."""
    keyword = keyword.strip()
    if not keyword:
        return ""
    if "@" in keyword:
        return keyword
    return f"{keyword} @{keyword}"


def validate_keyword_lines(keyword_lines: list[str], tokens_file: str | Path) -> list[str]:
    """Validate keyword tokens before native sherpa initialization.

    sherpa-onnx terminates the process when a keyword contains a token absent
    from ``tokens.txt``. Checking in Python turns that fatal startup into a
    normal readiness error.
    """
    path = Path(tokens_file)
    if not path.is_file():
        return [f"tokens file does not exist: {path}"]

    vocabulary = {
        line.split(maxsplit=1)[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    errors: list[str] = []
    for line in keyword_lines:
        token_text = line.split("@", 1)[0].strip()
        for token in token_text.split():
            if token.startswith(("#", ":")):
                continue
            if token not in vocabulary:
                errors.append(f"unsupported token {token!r} in keyword {line!r}")
    return errors


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
        self.spotter = None
        if sherpa_onnx.KeywordSpotter is None:
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

        configured_keywords = [
            normalize_keyword_line(str(keyword))
            for keyword in config.get("keywords", [])
            if str(keyword).strip()
        ]
        # Empty keywords list = skip KWS entirely (always-on listening)
        if not configured_keywords:
            logger.info("KWS disabled: no keywords configured (always-on mode).")
            return

        # Configured keywords take precedence over any existing keywords file so
        # wake-word changes apply immediately without manual file edits.
        keyword_errors = validate_keyword_lines(configured_keywords, tokens)
        if keyword_errors:
            logger.error("KWS keyword configuration invalid: %s", "; ".join(keyword_errors))
            return
        with open(keywords_file, "w", encoding="utf-8") as f:
            for keyword in configured_keywords:
                f.write(keyword + "\n")

        self.spotter = sherpa_onnx.KeywordSpotter(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=int(config.get("num_threads", 1)),
            provider=str(config.get("provider", "cpu")),
            device=int(config.get("device", 0)),
            keywords_file=keywords_file,
        )

        logger.info("KWS initialized.")

    @staticmethod
    def _normalize_keyword(keyword: str) -> str:
        return normalize_keyword_line(keyword)

    @property
    def available(self) -> bool:
        """Return True if the keyword spotter was loaded successfully."""
        return self.spotter is not None

    def create_stream(self) -> sherpa_onnx.OnlineStream | None:
        """Create and return a new KWS stream, or None if spotter unavailable."""
        if self.spotter is None:
            return None
        return self.spotter.create_stream()
