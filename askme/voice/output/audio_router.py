"""Re-exports from the canonical flat module askme.voice.audio_router.

The flat askme.voice.* modules are the authoritative implementations.
Sub-packages exist for organisational clarity only.
"""
from askme.voice.audio_router import AudioRouter, AudioErrorKind  # noqa: F401

__all__ = ['AudioRouter', 'AudioErrorKind']
