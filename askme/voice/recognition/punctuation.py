"""Re-exports from the canonical flat module askme.voice.punctuation.

The flat askme.voice.* modules are the authoritative implementations.
Sub-packages exist for organisational clarity only.
"""
from askme.voice.punctuation import PunctuationRestorer  # noqa: F401

__all__ = ['PunctuationRestorer']
