"""StreamSplitter - split streaming LLM output into speakable sentence chunks.

Implements a 5-rule sentence splitting strategy optimized for Chinese + English TTS:

1. **Strong punctuation** - always split on `.?!。？！\\n`
2. **Medium punctuation** - split on `;；:：` when buffer > 8 chars
3. **First sentence aggressive** - split on comma after 5 chars when total < 40 chars
4. **Normal comma split** - split on comma when buffer > 15 chars
5. **Emergency split** - prevent infinite buffering when buffer > 60 chars
"""

from __future__ import annotations

# Strong punctuation characters that always trigger a split
_STRONG_PUNCT = frozenset(".?!。？！\n")
# Medium punctuation — split after short accumulation
_MEDIUM_PUNCT = frozenset(";；:：")
# Comma characters for softer splitting
_COMMAS = frozenset(",，、")


class StreamSplitter:
    """Accumulates streaming text and yields complete sentences to speak.

    Usage::

        splitter = StreamSplitter()
        for token in llm_stream:
            for sentence in splitter.feed(token):
                tts.speak(sentence)
        leftover = splitter.flush()
        if leftover:
            tts.speak(leftover)
    """

    def __init__(
        self,
        first_sentence_threshold: int = 40,
        first_sentence_min_len: int = 5,
        normal_comma_min_len: int = 15,
        emergency_max_len: int = 60,
    ) -> None:
        self._buffer: str = ""
        self._total_chars: int = 0

        # Tunable thresholds
        self._first_threshold = first_sentence_threshold
        self._first_min = first_sentence_min_len
        self._comma_min = normal_comma_min_len
        self._emergency_max = emergency_max_len

    def feed(self, content: str) -> list[str]:
        """Feed a new token/chunk and return a list of complete sentences to speak.

        Returns an empty list if no sentence boundary was found yet.
        """
        self._buffer += content
        self._total_chars += len(content)

        should_split = False

        # Rule 1: Strong punctuation - always split
        if any(p in content for p in _STRONG_PUNCT):
            should_split = True

        # Rule 2: Medium punctuation (;:) - split after 8+ chars
        elif any(p in content for p in _MEDIUM_PUNCT) and len(self._buffer) > 8:
            should_split = True

        # Rule 3: First sentence - split aggressively on comma after 5 chars
        elif (
            self._total_chars < self._first_threshold
            and any(c in content for c in _COMMAS)
            and len(self._buffer) > self._first_min
        ):
            should_split = True

        # Rule 4: Normal comma split after 15 chars
        elif len(self._buffer) > self._comma_min and any(c in content for c in _COMMAS):
            should_split = True

        # Rule 5: Emergency split — split at best boundary, keep remainder
        elif len(self._buffer) > self._emergency_max:
            pos = self._find_split_point(self._buffer)
            head = self._buffer[:pos].strip()
            self._buffer = self._buffer[pos:]
            return [head] if head else []

        if should_split and self._buffer.strip():
            sentence = self._buffer.strip()
            self._buffer = ""
            return [sentence]

        return []

    @staticmethod
    def _find_split_point(text: str) -> int:
        """Find the best split position within a long text.

        Prefers splitting at: space > CJK boundary near 2/3 mark > hard limit.
        """
        limit = len(text) * 2 // 3
        if limit < 10:
            limit = min(len(text), 30)

        # Try last space before the 2/3 mark
        last_space = text.rfind(" ", 0, limit + 10)
        if last_space > limit // 2:
            return last_space + 1

        # For CJK text (no spaces), split at the 2/3 mark
        return limit

    def flush(self) -> str | None:
        """Return any remaining buffered text, or None if empty."""
        leftover = self._buffer.strip()
        self._buffer = ""
        self._total_chars = 0
        return leftover if leftover else None

    def reset(self) -> None:
        """Clear all internal state."""
        self._buffer = ""
        self._total_chars = 0
