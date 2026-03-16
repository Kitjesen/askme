"""Re-exports from the canonical flat module askme.voice.audio_filter.

The flat askme.voice.* modules are the authoritative implementations.
Sub-packages exist for organisational clarity only.
"""
from askme.voice.audio_filter import AudioFilter  # noqa: F401

__all__ = ['AudioFilter']
