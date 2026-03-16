"""Re-exports from the canonical flat module askme.voice.vad.

The flat askme.voice.* modules are the authoritative implementations.
Sub-packages exist for organisational clarity only.
"""
from askme.voice.vad import VADEngine  # noqa: F401

__all__ = ['VADEngine']
