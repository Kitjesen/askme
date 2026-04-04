"""Shared pipeline utilities — small helpers used by multiple sub-modules."""

from __future__ import annotations

import re

# Compiled once; shared across StreamProcessor, SkillGate, BrainPipeline, etc.
_RE_THINK = re.compile(r"<think>[\s\S]*?</think>", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove all ``<think>...</think>`` blocks from a complete string."""
    return _RE_THINK.sub("", text).strip()
