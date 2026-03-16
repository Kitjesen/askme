"""Re-exports from the canonical flat module askme.voice.asr_manager.

The flat askme.voice.* modules are the authoritative implementations.
Sub-packages exist for organisational clarity only.
"""
from askme.voice.asr_manager import ASRManager, ASRResult, _CONFIRMATION_WORDS, _MIN_VALID_TEXT_LEN, _NOISE_UTTERANCES, _SINGLE_CHAR_COMMANDS  # noqa: F401

__all__ = ['ASRManager', 'ASRResult', '_CONFIRMATION_WORDS', '_MIN_VALID_TEXT_LEN', '_NOISE_UTTERANCES', '_SINGLE_CHAR_COMMANDS']
