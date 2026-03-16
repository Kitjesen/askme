"""Re-exports from the canonical flat module askme.voice.mic_input.

The flat askme.voice.* modules are the authoritative implementations.
Sub-packages exist for organisational clarity only.
"""
from askme.voice.mic_input import MicInput  # noqa: F401

__all__ = ['MicInput']
