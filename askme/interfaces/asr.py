"""ASR (Automatic Speech Recognition) backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from askme.runtime.registry import BackendRegistry


class ASRBackend(ABC):
    """Abstract ASR interface — audio to text."""

    @abstractmethod
    def __init__(self, cfg: dict[str, Any]) -> None: ...

    @abstractmethod
    def recognize(self, audio_samples: bytes, sample_rate: int = 16000) -> str:
        """Recognize speech from audio samples. Returns transcribed text."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Whether the recognizer is initialized and ready."""

    def supports_streaming(self) -> bool:
        """Whether this backend supports streaming recognition."""
        return False


asr_registry = BackendRegistry("asr", ASRBackend, default="sherpa")
