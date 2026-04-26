"""Typed slot analysis: extract slot values and detect vague placeholders.

Design:
- Each SlotSpec in a skill's required_slots has a type (location / text / referent / …)
- For each slot, we try to extract the value from user_text
- We then check whether the extracted value is a vague placeholder word

Vague placeholder examples:
  - "那个", "这个"  — referential but unspecified
  - "那里", "这里"  — location unspecified
  - "一下", "查查"  — filler with no semantic content
  - "东西", "什么"  — generic, non-specific

A slot containing only such words is treated as VAGUE (= still missing).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .slot_types import SlotAnalysis, SlotFill

if TYPE_CHECKING:
    from askme.skills.skill_model import SkillDefinition, SlotSpec

# ── Vague word sets ───────────────────────────────────────────────────────────

#: Pure referential placeholders — the target/object is not named
_VAGUE_REFERENTS: frozenset[str] = frozenset([
    "那个", "这个", "那件", "这件", "那种", "这种",
    "那些", "这些", "那个东西", "这个东西",
    "它", "他", "她",
])

#: Vague location markers
_VAGUE_LOCATIONS: frozenset[str] = frozenset([
    "那里", "这里", "那边", "这边", "那处", "此处",
    "过去", "过来", "那里面", "这里面",
])

#: Filler content — utterance has no information
_VAGUE_FILLERS: frozenset[str] = frozenset([
    "一下", "一下吧", "一下子", "一会儿",
    "看看", "查查", "试试", "搜搜",
    "某个", "某处", "某地",
    "东西", "什么东西", "啥", "什么",
    "不知道", "随便", "都行", "无所谓",
])

_ALL_VAGUE: frozenset[str] = _VAGUE_REFERENTS | _VAGUE_LOCATIONS | _VAGUE_FILLERS


def _strip_vague_words(text: str) -> str:
    """Remove all known vague words from *text* and return the remainder.

    Longer entries are removed first to avoid partial overlaps
    (e.g. "那个东西" removed before "那个").
    """
    result = text
    for v in sorted(_ALL_VAGUE, key=len, reverse=True):
        result = result.replace(v, "")
    return result.strip()


def is_vague(text: str) -> bool:
    """Return True if *text* is a vague placeholder (after stripping whitespace).

    Called on the *extracted slot value*, not the full utterance. Two checks:
    1. Exact match against the vague word set (e.g. "那个", "随便", "那里").
    2. For short values (≤ 8 chars): strip all vague words; if ≤ 1 real
       character remains, the whole value is effectively a vague placeholder.
       Examples:
         "抓那个"   → strip "那个" → "抓" (1 char)  → vague ✓
         "那里看看" → strip "那里"+"看看" → "" (0)  → vague ✓
         "那个瓶子" → strip "那个" → "瓶子" (2)     → NOT vague ✓
         "那个红色积木" → strip "那个" → "红色积木" → NOT vague ✓
    """
    t = text.strip()
    if not t or len(t) <= 1:
        return True
    if t in _ALL_VAGUE:
        return True
    if len(t) <= 8:
        return len(_strip_vague_words(t)) <= 1
    return False


# ── Extraction helpers ────────────────────────────────────────────────────────


def _strip_triggers(user_text: str, skill: SkillDefinition) -> str:
    """Remove the longest matching voice trigger from user_text and return the rest."""
    if not skill.voice_trigger:
        return user_text.strip()

    triggers = sorted(
        [t.strip() for t in skill.voice_trigger.split(",") if t.strip()],
        key=len,
        reverse=True,
    )
    for trigger in triggers:
        pos = user_text.find(trigger)
        if pos >= 0:
            # Trigger found — return whatever is left (may be empty string).
            # Do NOT fall through to the full-text fallback.
            return (user_text[:pos] + user_text[pos + len(trigger) :]).strip()

    # No trigger matched at all — return the full text as best-effort value.
    return user_text.strip()


def _extract_slot_value(
    spec: SlotSpec,
    user_text: str,
    skill: SkillDefinition,
    pipeline: Any = None,
) -> str:
    """Extract the value for a slot from user_text.

    Returns an empty string when nothing was found.
    """
    if spec.type == "location" and pipeline is not None:
        try:
            target = pipeline.extract_semantic_target(user_text)
        except Exception:  # noqa: BLE001
            # Pipeline unavailable or raised — treat as no extraction
            return ""
        # If extraction returned the full input unchanged, no specific target
        if target and target.strip() != user_text.strip():
            return target.strip()
        return ""

    # For text / referent / datetime / enum: return content after trigger removal
    return _strip_triggers(user_text, skill)


# ── Public API ────────────────────────────────────────────────────────────────


def analyze_slots(
    skill: SkillDefinition,
    user_text: str,
    pipeline: Any = None,
) -> SlotAnalysis:
    """Analyze user_text against a skill's required_slots.

    Returns a :class:`SlotAnalysis` with status per slot:
    - "filled"  — value extracted and non-vague
    - "vague"   — value extracted but is a placeholder word
    - "missing" — no value could be extracted
    """
    fills: list[SlotFill] = []

    for spec in skill.required_slots:
        if spec.optional:
            continue  # optional slots are never blocking

        value = _extract_slot_value(spec, user_text, skill, pipeline)

        if not value:
            status = "missing"
        elif is_vague(value):
            status = "vague"
        else:
            status = "filled"

        fills.append(SlotFill(spec=spec, value=value or None, status=status))

    return SlotAnalysis(skill_name=skill.name, slots=fills)
