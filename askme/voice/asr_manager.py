"""ASR backend manager with local/cloud fallback and noise filtering.

Manages both local sherpa-onnx and cloud Paraformer ASR backends.
Provides noise utterance filtering and punctuation restoration.
Extracted from audio_agent.py for independent testing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from .asr import ASREngine
except ModuleNotFoundError:
    ASREngine = None  # type: ignore[assignment,misc]
from .cloud_asr import CloudASR
from .punctuation import PunctuationRestorer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Noise / confirmation constants (moved from audio_agent.py)
# ---------------------------------------------------------------------------

# ASR results that are clearly noise or feedback sounds, not commands.
# When matched, silently discard and re-listen (no TTS response).
_NOISE_UTTERANCES: frozenset[str] = frozenset([
    "\u55ef", "\u54e6", "\u554a", "\u5462", "\u54c8", "\u5662", "\u54c7", "\u5440", "\u55e8",
    "\u55ef\u55ef", "\u54e6\u54e6", "\u554a\u554a", "\u55ef\uff1f", "\u54e6\uff1f", "\u554a\uff1f",
    "\u7684", "\u4e86", "\u5427", "\u561b", "\u90a3", "\u8fd9", "\u5c31",
    "\u90a3\u4e2a", "\u8fd9\u4e2a", "\u5c31\u662f", "\u7136\u540e", "\u6240\u4ee5", "\u4f46\u662f",
])

# Words that are normally noise BUT become valid when the system is
# awaiting user confirmation (e.g. "要执行巡检吗？" -> "好的").
# These are only passed through when awaiting_confirmation is True.
_CONFIRMATION_WORDS: frozenset[str] = frozenset([
    "\u5bf9", "\u597d", "\u884c", "\u662f", "\u4e0d", "\u6ca1", "\u6709",
    "\u5bf9\u5bf9", "\u597d\u597d", "\u662f\u7684", "\u597d\u7684", "\u6ca1\u6709", "\u4e0d\u662f",
    "\u786e\u8ba4", "\u53d6\u6d88", "\u53ef\u4ee5", "\u4e0d\u884c", "\u7b97\u4e86", "\u6267\u884c",
    "\u5bf9\u7684", "\u6ca1\u9519", "\u597d\u5427", "\u4e0d\u8981", "\u522b", "\u62d2\u7edd",
    "\u540c\u610f", "\u6279\u51c6", "\u7ee7\u7eed", "\u653e\u5f03", "ok", "yes", "no",
])

# Minimum text length (in characters) to consider as valid speech.
# Shorter results are silently discarded unless they match known commands.
_MIN_VALID_TEXT_LEN = 2

# Single-char words that ARE valid commands (bypass length filter).
_SINGLE_CHAR_COMMANDS: frozenset[str] = frozenset([
    "\u505c", "\u8d70", "\u7ad9", "\u6765", "\u53bb", "\u5f00", "\u5173", "\u8d77", "\u5750", "\u9000",
])


@dataclass
class ASRResult:
    """Result from ASR recognition."""

    text: str
    source: str  # "local" | "cloud"
    is_noise: bool = False
    latency_ms: float = 0.0


class ASRManager:
    """ASR backend manager with local/cloud fallback and noise filtering.

    Manages both local sherpa-onnx and cloud Paraformer ASR backends.
    Provides noise utterance filtering and punctuation restoration.
    Extracted from audio_agent.py for independent testing.

    Config keys::

        asr: {model_dir, sample_rate, ...}  - local ASR config
        cloud_asr: {enabled, api_key, ...}  - cloud ASR config
        punctuation: {model_path, ...}      - punctuation restorer
    """

    def __init__(self, config: dict[str, Any]) -> None:
        # Local ASR (always available)
        self._asr = ASREngine(config.get("asr", {}))
        self._stream = self._asr.create_stream()
        self._punct = PunctuationRestorer(config.get("punctuation", {}))

        # Cloud ASR (optional)
        self._cloud = CloudASR(config.get("cloud_asr", {}))
        self._cloud_active: bool = False

        # State
        self._recognition_active: bool = False
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def preconnect_cloud(self) -> None:
        """Pre-establish Cloud ASR WebSocket before speech starts.

        Called at listen-start so the connection is warm when speech arrives.
        Audio fed before start_session() is silently accepted by the cloud.
        """
        if self._cloud.available and not self._cloud_active:
            self._cloud_active = self._cloud.start_session()

    def start_session(self) -> None:
        """Start a new recognition session (both local + cloud if available)."""
        self._recognition_active = True
        self._start_time = time.monotonic()

        # Always reset local stream for a clean slate
        self._asr.reset(self._stream)
        self._stream = self._asr.create_stream()

        # Start cloud session if not already pre-connected
        if self._cloud.available and not self._cloud_active:
            self._cloud_active = self._cloud.start_session()
        # If preconnect already opened, just keep it

    def feed_cloud_only(self, samples_int16: np.ndarray) -> None:
        """Feed audio to cloud ASR only (during silence, keeps connection warm)."""
        if self._cloud_active:
            try:
                self._cloud.feed(samples_int16.tobytes())
            except Exception:
                pass

    def feed_audio(
        self,
        samples_float32: np.ndarray,
        samples_int16: np.ndarray,
        sample_rate: int,
    ) -> None:
        """Feed audio to both ASR backends.

        Args:
            samples_float32: Float32 samples for local sherpa-onnx ASR.
            samples_int16: Int16 samples for cloud ASR (converted to PCM bytes).
            sample_rate: Audio sample rate (e.g. 16000).
        """
        if not self._recognition_active:
            return

        # Feed local ASR
        self._stream.accept_waveform(sample_rate, samples_float32)
        while self._asr.is_ready(self._stream):
            self._asr.decode_stream(self._stream)

        # Feed cloud ASR
        if self._cloud_active:
            pcm_bytes = samples_int16.tobytes()
            try:
                self._cloud.feed(pcm_bytes)
            except Exception as exc:
                logger.warning("Cloud ASR feed error: %s", exc)

    def check_endpoint(self) -> ASRResult | None:
        """Check if local ASR has detected an endpoint with text.

        Returns ASRResult if endpoint detected, None otherwise.
        """
        if not self._recognition_active:
            return None

        if not self._asr.is_endpoint(self._stream):
            return None

        text = self._asr.get_result(self._stream).strip()
        if not text:
            return None

        latency = (time.monotonic() - self._start_time) * 1000
        return ASRResult(text=text, source="local", latency_ms=latency)

    def finish_and_get_result(
        self, awaiting_confirmation: bool = False
    ) -> ASRResult | None:
        """Finish cloud session and return best result (cloud preferred, local fallback).

        Called when VAD detects speech end. Cloud ASR is preferred when it
        returns a non-empty result; otherwise local ASR result is used.

        Args:
            awaiting_confirmation: When True, confirmation words bypass noise filter.

        Returns:
            ASRResult with the best transcription, or None if no speech detected.
        """
        if not self._recognition_active:
            return None

        latency = (time.monotonic() - self._start_time) * 1000
        cloud_text = ""
        local_text = ""

        # Get local ASR result FIRST — so we can decide cloud timeout
        while self._asr.is_ready(self._stream):
            self._asr.decode_stream(self._stream)
        local_text = self._asr.get_result(self._stream).strip()

        # Finish cloud session — if local already has a result, only give cloud
        # 300ms to confirm/override it; otherwise wait the full 3s fallback window.
        if self._cloud_active:
            self._cloud_active = False
            try:
                cloud_timeout = 0.3 if local_text else 3.0
                cloud_text = self._cloud.finish_session(timeout=cloud_timeout).strip()
                if cloud_timeout == 0.3 and cloud_text:
                    logger.debug("Cloud ASR confirmed in fast path (local was ready)")
            except Exception as exc:
                logger.warning("Cloud ASR finish failed, using local: %s", exc)
                cloud_text = ""

        # Pick best: cloud preferred when available
        if cloud_text:
            text = cloud_text
            source = "cloud"
        elif local_text:
            text = local_text
            source = "local"
        else:
            self._recognition_active = False
            return None

        self._recognition_active = False

        # Noise filter
        is_noise = self._filter_noise(text, awaiting_confirmation)
        if is_noise:
            logger.info("ASR noise filtered: '%s'", text)
            return ASRResult(
                text=text, source=source, is_noise=True, latency_ms=latency,
            )

        # Punctuation restoration
        text = self._restore_punctuation(text)

        return ASRResult(text=text, source=source, latency_ms=latency)

    def force_endpoint(self) -> ASRResult | None:
        """Force an ASR endpoint (for max speech duration guard).

        Drains the local ASR decoder and returns the result. Cloud session
        is cancelled since forced endpoints are typically noise.
        """
        if not self._recognition_active:
            return None

        latency = (time.monotonic() - self._start_time) * 1000

        # Cancel cloud (forced endpoints are usually noise)
        if self._cloud_active:
            self._cloud_active = False
            self._cloud.cancel_session()

        # Drain local ASR
        while self._asr.is_ready(self._stream):
            self._asr.decode_stream(self._stream)
        text = self._asr.get_result(self._stream).strip()

        self._recognition_active = False

        if not text or len(text) <= 1:
            return None

        text = self._restore_punctuation(text)
        return ASRResult(text=text, source="local", latency_ms=latency)

    def reset(self) -> None:
        """Reset ASR streams for next utterance."""
        self._recognition_active = False
        self._cloud_active = False
        self._start_time = 0.0
        self._asr.reset(self._stream)
        self._stream = self._asr.create_stream()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_noise(self, text: str, awaiting_confirmation: bool = False) -> bool:
        """Check if *text* is noise/feedback that should be discarded.

        Public wrapper around :meth:`_filter_noise`.
        """
        return self._filter_noise(text, awaiting_confirmation)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_noise(self, text: str, awaiting_confirmation: bool) -> bool:
        """Check if text is noise/feedback. Returns True if noise (should discard).

        Context-aware: when *awaiting_confirmation* is True, confirmation words
        ("好的", "是的", "不" etc.) bypass the filter so the user can confirm/reject.
        """
        is_confirmation_word = awaiting_confirmation and text in _CONFIRMATION_WORDS

        if is_confirmation_word:
            return False

        if text in _NOISE_UTTERANCES:
            return True
        if text in _CONFIRMATION_WORDS:
            return True
        if len(text) == 1 and text not in _SINGLE_CHAR_COMMANDS:
            return True
        if len(text) < _MIN_VALID_TEXT_LEN and text not in _SINGLE_CHAR_COMMANDS:
            return True

        return False

    def _restore_punctuation(self, text: str) -> str:
        """Add punctuation to raw ASR text."""
        if self._punct.available:
            return self._punct.restore(text)
        return text
