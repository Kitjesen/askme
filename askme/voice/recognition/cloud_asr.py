"""Re-exports from the canonical flat module askme.voice.cloud_asr.

The flat askme.voice.* modules are the authoritative implementations.
Sub-packages exist for organisational clarity only.
"""
from askme.voice.cloud_asr import CloudASR  # noqa: F401

__all__ = ['CloudASR']
