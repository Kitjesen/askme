"""ASR (Automatic Speech Recognition) backend interface."""

from __future__ import annotations

from abc import ABC
from typing import Any

from askme.runtime.registry import BackendRegistry


class ASRBackend(ABC):
    """Abstract ASR interface — speech recognition.

    Two concrete flavours exist:

    * **Frame-by-frame** (:class:`~askme.voice.asr.ASREngine`): caller drives
      a stream with ``create_stream`` / ``decode_stream`` / ``get_result``.
    * **Session-based** (:class:`~askme.voice.cloud_asr.CloudASR`): caller
      opens a session, feeds audio, and collects a final transcript.

    Methods below provide default ``NotImplementedError`` stubs so each
    subclass only overrides the methods relevant to its flavour.
    """

    # ── Frame-by-frame API (ASREngine) ──────────────────

    def create_stream(self) -> Any:
        """Create and return a new ASR stream."""
        raise NotImplementedError

    def is_ready(self, stream: Any) -> bool:
        """Check if the recognizer is ready to decode the given stream."""
        raise NotImplementedError

    def decode_stream(self, stream: Any) -> None:
        """Decode one step on the given stream."""
        raise NotImplementedError

    def is_endpoint(self, stream: Any) -> bool:
        """Check if an endpoint has been detected in the stream."""
        raise NotImplementedError

    def get_result(self, stream: Any) -> str:
        """Get the current recognition result text from the stream."""
        raise NotImplementedError

    def reset(self, stream: Any) -> None:
        """Reset the given stream."""
        raise NotImplementedError

    # ── Session-based API (CloudASR) ────────────────────

    @property
    def available(self) -> bool:
        """Whether this backend is available for use."""
        return False

    def start_session(self) -> bool:
        """Open a recognition session. Returns True on success."""
        raise NotImplementedError

    def feed(self, pcm16_bytes: bytes) -> None:
        """Send a chunk of PCM16 audio to the recognizer."""
        raise NotImplementedError

    def finish_session(self, timeout: float = 5.0) -> str:
        """Finish the session and return the transcribed text."""
        raise NotImplementedError

    def cancel_session(self) -> None:
        """Cancel the current session without waiting for results."""
        raise NotImplementedError


asr_registry = BackendRegistry("asr", ASRBackend, default="sherpa")
