"""TTS (Text-to-Speech) backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from askme.runtime.registry import BackendRegistry


class TTSBackend(ABC):
    """Abstract TTS interface — text to audio.

    Matches the public API of :class:`askme.voice.tts.TTSEngine`.
    """

    @abstractmethod
    def speak(self, text: str) -> None:
        """Queue text for speech synthesis and playback."""

    @abstractmethod
    def stop_immediately(self) -> None:
        """Stop playback immediately and clear all queued audio."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release all resources."""

    @property
    @abstractmethod
    def backend(self) -> str:
        """Current backend identifier (e.g. 'local', 'edge', 'minimax')."""

    def is_active(self) -> bool:
        """Whether audio is currently being played."""
        return False


tts_registry = BackendRegistry("tts", TTSBackend, default="minimax")
