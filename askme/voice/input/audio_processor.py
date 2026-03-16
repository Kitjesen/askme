"""Re-exports from the canonical flat module askme.voice.audio_processor.

The flat askme.voice.* modules are the authoritative implementations.
Sub-packages exist for organisational clarity only.
"""
from askme.voice.audio_processor import AudioProcessor  # noqa: F401

__all__ = ['AudioProcessor']
