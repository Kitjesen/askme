"""Re-exports from the canonical flat module askme.voice.address_detector.

The flat askme.voice.* modules are the authoritative implementations.
Sub-packages exist for organisational clarity only.
"""
from askme.voice.address_detector import AddressDetector  # noqa: F401

__all__ = ['AddressDetector']
