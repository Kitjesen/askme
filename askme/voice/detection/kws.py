"""Re-exports from the canonical flat module askme.voice.kws.

The flat askme.voice.* modules are the authoritative implementations.
Sub-packages exist for organisational clarity only.
"""
from askme.voice.kws import KWSEngine  # noqa: F401

__all__ = ['KWSEngine']
