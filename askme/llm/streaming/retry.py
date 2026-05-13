"""Retry constants and backoff policy for LLM transport calls."""

from __future__ import annotations

import random

RETRYABLE_STATUS = {500, 502, 503, 504, 529}


def default_backoff(attempt: int) -> float:
    """Exponential backoff with jitter.

    The first retry stays fast because conversational voice turns cannot spend
    several seconds waiting before trying the same provider again.
    """

    if attempt == 0:
        return 0.3 + random.uniform(0, 0.2)
    base = min(2 ** (attempt - 1), 8)
    return base + random.uniform(0, 0.5)
