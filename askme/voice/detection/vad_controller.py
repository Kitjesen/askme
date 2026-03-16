"""Re-exports from the canonical flat module askme.voice.vad_controller.

The flat askme.voice.* modules are the authoritative implementations.
Sub-packages exist for organisational clarity only.
"""
from askme.voice.vad_controller import VADController, VADEvent, _BARGE_IN_HOLD_S, _MAX_SPEECH_DURATION  # noqa: F401

__all__ = ['VADController', 'VADEvent', '_BARGE_IN_HOLD_S', '_MAX_SPEECH_DURATION']
