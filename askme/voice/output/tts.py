"""Re-exports from the canonical flat module askme.voice.tts.

The flat askme.voice.* modules are the authoritative implementations.
Sub-packages exist for organisational clarity only.
"""
from askme.voice.tts import TTSEngine  # noqa: F401

__all__ = ['TTSEngine']
