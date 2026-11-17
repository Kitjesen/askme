"""Xiaozhi-aligned speech helpers for AskMe voice output.

These functions are pure and do not open audio devices or network sockets.
TTS and the voice gateway call them; board code can call them directly.
"""

from askme.voice.speech.correct_words import (
    CORRECT_WORDS_ENV,
    DEFAULT_CORRECT_WORDS,
    PronunciationTable,
    StreamCorrector,
    apply_pronunciation,
    correct_stream_chunk,
    parse_correct_words,
    resolve_correct_words,
    restore_display_text,
)
from askme.voice.speech.emotion import (
    DEFAULT_EMOJI,
    DEFAULT_EMOTION,
    EMOJI_MAP,
    Emotion,
    build_llm_emotion_message,
    dumps_llm_emotion_message,
    extract_emotion,
    push_device_emotion,
)
from askme.voice.speech.tts_retry import (
    DEFAULT_TTS_MAX_ATTEMPTS,
    TTSRetryResult,
    TTSSynthesisFailure,
    synthesize_with_retry,
)

__all__ = [
    "CORRECT_WORDS_ENV",
    "DEFAULT_CORRECT_WORDS",
    "DEFAULT_EMOTION",
    "DEFAULT_EMOJI",
    "DEFAULT_TTS_MAX_ATTEMPTS",
    "EMOJI_MAP",
    "Emotion",
    "PronunciationTable",
    "StreamCorrector",
    "TTSRetryResult",
    "TTSSynthesisFailure",
    "apply_pronunciation",
    "build_llm_emotion_message",
    "correct_stream_chunk",
    "dumps_llm_emotion_message",
    "extract_emotion",
    "parse_correct_words",
    "push_device_emotion",
    "resolve_correct_words",
    "restore_display_text",
    "synthesize_with_retry",
]
