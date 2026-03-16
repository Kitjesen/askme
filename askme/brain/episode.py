"""Episode dataclass and importance scoring — extracted from episodic_memory.py.

These are the core data structures for the episodic memory system, separated
so they can be imported without pulling in the full EpisodicMemory machinery.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from typing import Any

# ── Constants ──────────────────────────────────────────────

# Activation / Decay (Ebbinghaus + ACT-R inspired)
DEFAULT_STABILITY_S = 3600.0   # Initial stability = 1 hour (Ebbinghaus S parameter)
STABILITY_GROWTH_FACTOR = 2.0  # Each access doubles stability (spacing effect / SM-2 inspired)
MAX_STABILITY_S = 7 * 86400.0  # Cap at 7 days

# Park 2023 retrieval weights (relevance-dominant)
WEIGHT_RECENCY = 0.5
WEIGHT_IMPORTANCE = 2.0
WEIGHT_RELEVANCE = 3.0

# Importance scoring rules (event_type → base importance)
IMPORTANCE_RULES: dict[str, float] = {
    "perception": 0.3,   # Routine sensing
    "action":     0.2,   # Routine action
    "outcome":    0.3,   # Action result
    "command":    0.7,   # User interaction — high value
    "error":      0.8,   # Errors are important to remember
    "system":     0.1,   # System housekeeping — low importance
}

# Context-based importance boosts
IMPORTANCE_BOOSTS: dict[str, float] = {
    "person":     0.5,   # Detecting a person is significant
    "new":        0.4,   # New entity / first-time detection
    "danger":     0.6,   # Safety-related events
    "fail":       0.3,   # Failure in action
    "success":    0.1,   # Expected success — small boost
    "surprise":   0.4,   # Unexpected event (novelty / prediction error)
}


class Episode:
    """A single event in the robot's experience stream with importance, stability, and decay.

    Memory model based on Ebbinghaus forgetting curve (R = e^(-t/S)):
      - stability (S) starts at DEFAULT_STABILITY_S
      - Each access doubles S (spacing effect / SM-2 inspired)
      - Higher importance → higher initial stability
      - Retrievability R decays exponentially but slows with each retrieval
    """

    __slots__ = (
        "timestamp", "event_type", "description", "context",
        "importance", "stability", "access_count", "last_accessed",
    )

    def __init__(
        self,
        event_type: str,
        description: str,
        context: dict[str, Any] | None = None,
        importance: float = 0.0,
    ) -> None:
        self.timestamp: float = time.time()
        self.event_type: str = event_type
        self.description: str = description
        self.context: dict[str, Any] = context or {}
        self.importance: float = importance
        # Ebbinghaus stability: higher importance → higher initial S
        self.stability: float = DEFAULT_STABILITY_S * (1.0 + importance)
        self.access_count: int = 0
        self.last_accessed: float = self.timestamp

    def access(self) -> None:
        """Record an access (retrieval). Each access doubles stability (spacing effect)."""
        self.access_count += 1
        self.last_accessed = time.time()
        # SM-2 inspired: each successful retrieval doubles stability
        self.stability = min(self.stability * STABILITY_GROWTH_FACTOR, MAX_STABILITY_S)

    def retrievability(self, now: float | None = None) -> float:
        """Ebbinghaus forgetting curve: R = e^(-t/S).

        R ∈ [0, 1]: 1.0 = perfect recall, 0.0 = forgotten.
        S grows on each access → memory becomes more durable over time.
        """
        now = now or time.time()
        t = now - self.last_accessed
        return math.exp(-t / self.stability)

    # Keep compute_activation as alias for backward compatibility
    def compute_activation(self, now: float | None = None) -> float:
        """Alias for retrievability() — backward compatible."""
        return self.retrievability(now)

    def retrieval_score(self, query_keywords: set[str] | None = None, now: float | None = None) -> float:
        """Park 2023 weighted retrieval: recency(0.5) + importance(2.0) + relevance(3.0).

        Weights from Stanford Generative Agents paper — relevance dominates.
        """
        recency = self.retrievability(now)
        relevance = 0.0
        if query_keywords:
            desc_lower = self.description.lower()
            matches = sum(1 for kw in query_keywords if kw in desc_lower)
            relevance = min(matches / max(len(query_keywords), 1), 1.0)
        return (WEIGHT_RECENCY * recency
                + WEIGHT_IMPORTANCE * self.importance
                + WEIGHT_RELEVANCE * relevance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.timestamp,
            "time": datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S"),
            "type": self.event_type,
            "desc": self.description,
            "ctx": self.context,
            "importance": round(self.importance, 2),
            "stability": round(self.stability, 1),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "retrievability": round(self.retrievability(), 3),
        }

    def to_log_line(self) -> str:
        """Human-readable format for LLM reflection prompt."""
        time_str = datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")
        imp_str = f" [imp={self.importance:.1f}]"
        ctx_str = f" ({json.dumps(self.context, ensure_ascii=False)})" if self.context else ""
        return f"[{time_str}] [{self.event_type}]{imp_str} {self.description}{ctx_str}"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Episode:
        """Recreate an episode from persisted JSON data."""
        episode = cls(
            str(payload.get("type", "system")),
            str(payload.get("desc", "")),
            payload.get("ctx") or {},
            importance=float(payload.get("importance", 0.0)),
        )
        episode.timestamp = float(payload.get("ts", episode.timestamp))
        episode.stability = float(payload.get("stability", episode.stability))
        episode.access_count = int(payload.get("access_count", 0))
        episode.last_accessed = float(payload.get("last_accessed", episode.timestamp))
        return episode


def decay_importance(importance: float, age_hours: float, half_life_hours: float = 72.0) -> float:
    """Apply Ebbinghaus forgetting curve to importance score.

    importance decays exponentially: effective = importance * 2^(-age/half_life)
    Default half-life: 72 hours (3 days). After 3 days, importance halves.
    Reinforced memories (accessed again) reset their timestamp.

    Reference: Ebbinghaus (1885); A-MEM (Xu et al., NeurIPS 2025)
    """
    if age_hours <= 0:
        return importance
    return importance * math.pow(2, -age_hours / half_life_hours)


def score_importance(event_type: str, description: str, context: dict[str, Any] | None = None) -> float:
    """Rule-based importance scoring for a robot episode.

    Scoring layers:
      1. Base score from event type (command > error > perception > action > system)
      2. Context boosts (person detected, new entity, danger, failure)
      3. Confidence weighting for perception events
      4. Clamped to [0, 1]
    """
    base = IMPORTANCE_RULES.get(event_type, 0.3)
    boost = 0.0
    ctx = context or {}

    # Detection-based boosts
    detections = ctx.get("detections", [])
    desc_lower = description.lower()

    for det in detections:
        label = det.get("label", "").lower()
        conf = det.get("conf", det.get("confidence", 0.5))
        if label == "person":
            boost += IMPORTANCE_BOOSTS["person"] * conf
        elif label in ("fire", "smoke", "weapon"):
            boost += IMPORTANCE_BOOSTS["danger"] * conf

    # Keyword-based boosts
    if any(kw in desc_lower for kw in ("新", "first", "new", "未知", "unknown")):
        boost += IMPORTANCE_BOOSTS["new"]
    if any(kw in desc_lower for kw in ("失败", "fail", "error", "异常")):
        boost += IMPORTANCE_BOOSTS["fail"]
    if any(kw in desc_lower for kw in ("危险", "danger", "警告", "warning")):
        boost += IMPORTANCE_BOOSTS["danger"]
    # Person detection from text descriptions (e.g., vision "我看到了: 1个person")
    if not detections and any(kw in desc_lower for kw in ("person", "人", "行人")):
        boost += IMPORTANCE_BOOSTS["person"]

    # Surprise/novelty boost from context
    if ctx.get("surprise") or ctx.get("novel"):
        boost += IMPORTANCE_BOOSTS["surprise"]

    # High confidence perception gets a small boost
    if event_type == "perception" and detections:
        avg_conf = sum(d.get("conf", d.get("confidence", 0.5)) for d in detections) / len(detections)
        if avg_conf > 0.9:
            boost += 0.1

    return min(base + boost, 1.0)
