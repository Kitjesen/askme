"""Re-exports from the canonical flat module askme.voice.asr.

The flat askme.voice.* modules are the authoritative implementations.
Sub-packages exist for organisational clarity only.
"""
from askme.voice.asr import ASREngine  # noqa: F401

__all__ = ['ASREngine']
