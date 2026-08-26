"""Deterministic intents allowed on the voice fast path.

The fast path is intentionally narrow.  It may end ASR early only for replies
that are audio-only, read-only status queries, or an exact emergency stop.  It
never authorizes robot motion; the emergency path can only revoke motion.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

from askme.robot_interaction.routing_policy import DEFAULT_ESTOP_KEYWORDS


class FastVoiceIntentKind(Enum):
    ESTOP = "estop"
    QUICK_REPLY = "quick_reply"
    READ_ONLY_SKILL = "read_only_skill"


@dataclass(frozen=True)
class FastVoiceIntent:
    intent_id: str
    kind: FastVoiceIntentKind
    normalized_text: str
    reply_text: str | None = None
    skill_name: str | None = None
    preface_text: str | None = None
    cache_key: str | None = None


_SPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCTUATION = "\t\r\n ,.!?;:\uff0c\u3002\uff01\uff1f\uff1b\uff1a"

# A bare location/status request is read-only.  Navigation and movement words
# are deliberately absent from this table.
_READ_ONLY_LOCATION_PHRASES: frozenset[str] = frozenset(
    {
        "\u5b9a\u4f4d",
        "\u5b9a\u4f4d\u72b6\u6001",
        "\u5f53\u524d\u4f4d\u7f6e",
        "\u6211\u5728\u54ea\u91cc",
        "\u6211\u73b0\u5728\u5728\u54ea\u91cc",
        "\u4f60\u5728\u54ea\u91cc",
        "\u4f60\u73b0\u5728\u5728\u54ea\u91cc",
        "\u67e5\u8be2\u5f53\u524d\u4f4d\u7f6e",
    }
)

_LOCATION_PREFACE = "\u6b63\u5728\u8bfb\u53d6\u5f53\u524d\u4f4d\u7f6e\uff0c\u8bf7\u7a0d\u5019\u3002"
_SYSTEM_CACHED_PHRASES: Mapping[str, str] = {
    "system-please-yield": "\u60a8\u597d\uff0c\u8bf7\u8ba9\u4e00\u4e0b\uff0c\u8c22\u8c22\u3002",
}
_PLEASE_YIELD_PHRASES: frozenset[str] = frozenset(
    {
        "\u8bf7\u8ba9\u4e00\u4e0b",
        "\u9ebb\u70e6\u8ba9\u4e00\u4e0b",
        "\u60a8\u597d\u8bf7\u8ba9\u4e00\u4e0b",
    }
)


def normalize_fast_voice_text(text: str) -> str:
    """Normalize only formatting noise; do not perform fuzzy matching."""

    clean = _SPACE_RE.sub("", str(text or "").strip().lower())
    return clean.strip(_TRAILING_PUNCTUATION)


def _cache_key(prefix: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def match_fast_voice_intent(
    text: str,
    *,
    quick_replies: Mapping[str, str],
    estop_keywords: Iterable[str] = DEFAULT_ESTOP_KEYWORDS,
) -> FastVoiceIntent | None:
    """Return a fast-path decision for exact, explicitly safe utterances."""

    normalized = normalize_fast_voice_text(text)
    if not normalized:
        return None

    if any(
        normalize_fast_voice_text(keyword) == normalized
        for keyword in estop_keywords
    ):
        return FastVoiceIntent(
            intent_id="estop",
            kind=FastVoiceIntentKind.ESTOP,
            normalized_text=normalized,
        )

    for phrase, reply_text in quick_replies.items():
        if normalize_fast_voice_text(phrase) != normalized:
            continue
        return FastVoiceIntent(
            intent_id="quick_reply",
            kind=FastVoiceIntentKind.QUICK_REPLY,
            normalized_text=normalized,
            reply_text=reply_text,
            cache_key=_cache_key("quick", reply_text),
        )

    if normalized in _PLEASE_YIELD_PHRASES:
        return FastVoiceIntent(
            intent_id="please_yield",
            kind=FastVoiceIntentKind.QUICK_REPLY,
            normalized_text=normalized,
            reply_text=_SYSTEM_CACHED_PHRASES["system-please-yield"],
            cache_key="system-please-yield",
        )

    if normalized in _READ_ONLY_LOCATION_PHRASES:
        return FastVoiceIntent(
            intent_id="location_status",
            kind=FastVoiceIntentKind.READ_ONLY_SKILL,
            normalized_text=normalized,
            skill_name="nav_query",
            preface_text=_LOCATION_PREFACE,
            cache_key=_cache_key("location", _LOCATION_PREFACE),
        )

    return None


def default_cached_phrases(quick_replies: Mapping[str, str]) -> dict[str, str]:
    """Return the phrases that deployment tooling should pre-synthesize."""

    phrases = dict(_SYSTEM_CACHED_PHRASES)
    for phrase in quick_replies:
        intent = match_fast_voice_intent(phrase, quick_replies=quick_replies)
        if intent is not None and intent.reply_text and intent.cache_key:
            phrases[intent.cache_key] = intent.reply_text
    location = match_fast_voice_intent(
        "\u5f53\u524d\u4f4d\u7f6e",
        quick_replies=quick_replies,
    )
    if location is not None and location.preface_text and location.cache_key:
        phrases[location.cache_key] = location.preface_text
    return phrases


__all__ = [
    "FastVoiceIntent",
    "FastVoiceIntentKind",
    "default_cached_phrases",
    "match_fast_voice_intent",
    "normalize_fast_voice_text",
]
