"""Pronunciation corrections applied to text that is about to be synthesized.

Config entries use the official Xiaozhi form ``original|spoken`` (see
``TTSProviderBase`` in xinnan-tech/xiaozhi-esp32-server). One regex is built
from the keys, longest first, so a short name cannot steal the prefix of a
longer one. The spoken form is what TTS receives. ``restore`` maps that form
back to the original for captions and ``sentence_start``.

The table does not clean Markdown. Call it after the existing speech cleanup.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)

CORRECT_WORDS_ENV = "ASKME_TTS_CORRECT_WORDS"

# Small product list. 算 and 蒜 are both suàn, so the spoken form keeps the
# name's sound when an engine misreads 算, while captions restore 小算 / 大算.
# 科创e谷 keeps the Latin letter; 伊 is the spoken name of the letter E.
# Extend this in config or via ASKME_TTS_CORRECT_WORDS. Do not grow it in code.
DEFAULT_CORRECT_WORDS: tuple[str, ...] = (
    "小算|小蒜",
    "大算|大蒜",
    "聚龙科创e谷|聚龙科创伊谷",
    "巨龙科创e谷|巨龙科创伊谷",
)


@dataclass(frozen=True, slots=True)
class PronunciationTable:
    """Compiled forward and reverse pronunciation maps."""

    mapping: Mapping[str, str]
    reverse: Mapping[str, str]
    pattern: re.Pattern[str] | None
    reverse_pattern: re.Pattern[str] | None
    by_first: Mapping[str, tuple[str, ...]]
    skipped: tuple[str, ...] = ()

    def apply(self, text: str) -> str:
        """Replace originals in a complete string. Empty input is unchanged."""
        if not text or self.pattern is None:
            return text
        return self.pattern.sub(lambda match: self.mapping[match.group(0)], text)

    def restore(self, text: str) -> str:
        """Map TTS-side text back to the original writing, longest match first."""
        if not text or self.reverse_pattern is None:
            return text
        return self.reverse_pattern.sub(lambda match: self.reverse[match.group(0)], text)


def parse_correct_words(entries: Iterable[object] | None) -> PronunciationTable:
    """Build a table from ``original|spoken`` strings.

    Malformed lines, blank sides, and non-strings are skipped. A later entry
    replaces an earlier one with the same original. When two originals share a
    spoken form, the reverse map keeps the later original, matching the
    official ``{spoken: original}`` comprehension.
    """
    mapping: dict[str, str] = {}
    skipped: list[str] = []
    for item in entries or ():
        if not isinstance(item, str):
            skipped.append(repr(item))
            continue
        if "|" not in item:
            skipped.append(item)
            continue
        original, spoken = item.split("|", 1)
        original = original.strip()
        spoken = spoken.strip()
        if original == "" or spoken == "":
            skipped.append(item)
            continue
        mapping[original] = spoken
    return _compile(mapping, tuple(skipped))


def resolve_correct_words(
    configured: object = None,
    *,
    env: str | None = None,
    use_defaults_when_missing: bool = True,
) -> PronunciationTable:
    """Resolve yaml entries plus ``ASKME_TTS_CORRECT_WORDS``.

    A missing config value uses :data:`DEFAULT_CORRECT_WORDS`. An empty list
    disables those defaults. The environment variable is semicolon- or
    newline-separated and is appended, so it extends the list without a code
    change. Pass ``env`` explicitly in tests; otherwise the process environment
    is read.
    """
    if configured is None:
        base: list[object] = list(DEFAULT_CORRECT_WORDS) if use_defaults_when_missing else []
    elif isinstance(configured, str):
        base = list(_split_env_entries(configured))
    elif isinstance(configured, (list, tuple)):
        base = list(configured)
    else:
        base = []
    if env is None:
        env = os.environ.get(CORRECT_WORDS_ENV)
    extra = list(_split_env_entries(env))
    table = parse_correct_words([*base, *extra])
    for line in table.skipped:
        logger.warning("Ignoring malformed correct_words entry: %s", line)
    return table


def apply_pronunciation(text: str, table: PronunciationTable) -> str:
    """Apply ``table`` to a complete string that is about to be synthesized."""
    return table.apply(text)


def restore_display_text(text: str, table: PronunciationTable) -> str:
    """Restore original writing for a caption or ``sentence_start`` payload."""
    return table.restore(text)


def correct_stream_chunk(
    text: str,
    table: PronunciationTable,
    pending: str = "",
) -> tuple[str, str]:
    """Correct one streaming fragment.

    Words are grouped by their first character, longest first. A fragment that
    ends in a prefix of a correction stays in ``pending`` until a later chunk
    completes or rejects it. Returns ``(ready_text, pending)``.
    """
    pieces, pending_out = _match_stream(text, table, pending)
    return "".join(pieces), pending_out


class StreamCorrector:
    """Stateful wrapper around :func:`correct_stream_chunk` for one utterance."""

    def __init__(self, table: PronunciationTable) -> None:
        self._table = table
        self._pending = ""

    @property
    def pending(self) -> str:
        return self._pending

    def feed(self, text: str) -> str:
        ready, self._pending = correct_stream_chunk(text, self._table, self._pending)
        return ready

    def flush(self) -> str:
        """Release an unmatched tail. Call this at the end of a stream."""
        tail = self._pending
        self._pending = ""
        return tail

    def reset(self) -> None:
        self._pending = ""


def _split_env_entries(raw: str | None) -> tuple[str, ...]:
    if raw is None or not str(raw).strip():
        return ()
    parts: list[str] = []
    for chunk in str(raw).replace("\n", ";").split(";"):
        item = chunk.strip()
        if item:
            parts.append(item)
    return tuple(parts)


def _compile(mapping: dict[str, str], skipped: tuple[str, ...]) -> PronunciationTable:
    if not mapping:
        return PronunciationTable(
            mapping={},
            reverse={},
            pattern=None,
            reverse_pattern=None,
            by_first={},
            skipped=skipped,
        )
    sorted_keys = sorted(mapping, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(key) for key in sorted_keys))
    reverse: dict[str, str] = {}
    for key in mapping:
        reverse[mapping[key]] = key
    reverse_keys = sorted(reverse, key=len, reverse=True)
    reverse_pattern = re.compile("|".join(re.escape(key) for key in reverse_keys))
    by_first: dict[str, list[str]] = {}
    for key in sorted_keys:
        first = key[0]
        by_first.setdefault(first, []).append(key)
    frozen_first = {char: tuple(keys) for char, keys in by_first.items()}
    return PronunciationTable(
        mapping=dict(mapping),
        reverse=reverse,
        pattern=pattern,
        reverse_pattern=reverse_pattern,
        by_first=frozen_first,
        skipped=skipped,
    )


def _match_stream(
    text: str,
    table: PronunciationTable,
    pending: str,
) -> tuple[list[str], str]:
    """Official sliding-window match: longest key per first character wins.

    If the buffered text is only a prefix of a longer key, this waits, same as
    ``TTSProviderBase._match_stream_text``. A later character that breaks that
    prefix flushes the buffer literally. An empty chunk keeps ``pending``.
    """
    if not table.mapping:
        combined = f"{pending}{text}"
        return ([combined] if combined else []), ""
    if not text:
        return [], pending
    result: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        test = pending + char
        matched = False
        if pending:
            candidates = table.by_first.get(pending[0], ())
        else:
            candidates = table.by_first.get(char, ())
        for key in candidates:
            if test == key:
                result.append(table.mapping[key])
                pending = ""
                matched = True
                break
            if key.startswith(test):
                pending = test
                matched = True
                break
        if matched:
            index += 1
            continue
        if pending:
            result.append(pending)
            pending = ""
        if char in table.by_first:
            pending = char
        else:
            result.append(char)
        index += 1
    return result, pending
