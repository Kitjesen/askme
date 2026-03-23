"""TTS (Text-to-Speech) backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from askme.runtime.registry import BackendRegistry


class TTSBackend(ABC):
    """Abstract TTS interface — text to audio."""

    @abstractmethod
    def __init__(self, cfg: dict[str, Any]) -> None: ...

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Synthesize speech from text. Returns PCM audio bytes."""

    @abstractmethod
    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Streaming synthesis. Yields audio chunks."""

    @property
    @abstractmethod
    def voice_name(self) -> str:
        """Current voice identifier."""

    def supports_cloning(self) -> bool:
        """Whether this backend supports voice cloning."""
        return False


tts_registry = BackendRegistry("tts", TTSBackend, default="minimax")
