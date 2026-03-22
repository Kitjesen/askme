"""Memory Admission Control — 5-factor gate for episodic memory.

Decides whether an event is worth remembering before it enters the episode
buffer. Filters routine/duplicate events to keep memory focused on novel,
high-utility information.

Reference: A-MAC (arXiv 2603.04549, March 2026)

Usage::

    from askme.memory.admission import MemoryAdmissionControl

    mac = MemoryAdmissionControl(threshold=0.4)
    admitted, score = mac.should_admit("error", "Motor overcurrent", importance=0.8)
    if admitted:
        episodic.log("error", "Motor overcurrent")
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AdmissionScore:
    """Breakdown of the 5-factor admission score."""

    utility: float      # 0-1: will this be useful in the future?
    confidence: float   # 0-1: is this factual/certain?
    novelty: float      # 0-1: is this new information?
    recency: float      # 0-1: how recent? (always 1.0 for new events)
    type_prior: float   # 0-1: base score by event type (errors=high, routine=low)

    @property
    def total(self) -> float:
        """Weighted sum — utility and novelty weighted highest."""
        weights = [0.3, 0.15, 0.25, 0.1, 0.2]
        values = [self.utility, self.confidence, self.novelty, self.recency, self.type_prior]
        return sum(w * v for w, v in zip(weights, values))


# Event type → base prior (errors/anomalies high, routine low)
_TYPE_PRIORS: dict[str, float] = {
    "error": 0.9,
    "anomaly": 0.9,
    "command": 0.6,
    "action": 0.5,
    "outcome": 0.5,
    "perception": 0.3,
    "system": 0.2,
}

# Direct-observation event types get high confidence
_HIGH_CONFIDENCE_TYPES = frozenset(("command", "error", "action"))


class MemoryAdmissionControl:
    """5-factor gate: decide if an event is worth remembering.

    Factors:
        1. utility — based on importance score
        2. confidence — direct observation vs. inferred
        3. novelty — character-level Jaccard against recent texts
        4. recency — always 1.0 for new events
        5. type_prior — base score by event type

    Reference: A-MAC (arXiv 2603.04549, March 2026)
    """

    def __init__(self, threshold: float = 0.4) -> None:
        self._threshold = threshold
        self._recent_texts: deque[str] = deque(maxlen=50)

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    def should_admit(
        self, kind: str, text: str, importance: float = 0.0
    ) -> tuple[bool, AdmissionScore]:
        """Returns (should_store, score_breakdown)."""
        type_prior = _TYPE_PRIORS.get(kind, 0.4)
        novelty = self._compute_novelty(text)
        utility = min(1.0, max(0.0, importance))
        confidence = 0.9 if kind in _HIGH_CONFIDENCE_TYPES else 0.7
        recency = 1.0

        score = AdmissionScore(utility, confidence, novelty, recency, type_prior)
        admitted = score.total >= self._threshold

        if admitted:
            self._recent_texts.append(text)

        return admitted, score

    def _compute_novelty(self, text: str) -> float:
        """Simple character-level Jaccard novelty against recent texts."""
        if not self._recent_texts:
            return 1.0
        text_chars = set(text)
        if not text_chars:
            return 1.0
        max_overlap = 0.0
        for recent in self._recent_texts:
            recent_chars = set(recent)
            if not recent_chars:
                continue
            union = text_chars | recent_chars
            if not union:
                continue
            overlap = len(text_chars & recent_chars) / len(union)
            max_overlap = max(max_overlap, overlap)
        return 1.0 - max_overlap
