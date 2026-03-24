"""Trend analysis over episodic event streams.

Groups episodes by hour and event_type, compares recent windows against
historical baselines, and reports spikes (ratio > 2.0).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from askme.memory.episode import Episode

logger = logging.getLogger(__name__)

# Spike threshold: recent count / baseline must exceed this to be a trend
_SPIKE_THRESHOLD = 2.0


@dataclass(frozen=True)
class Trend:
    """A detected frequency spike in an event stream."""

    window_start: float
    window_end: float
    event_type: str
    count: int
    baseline: float
    spike_ratio: float
    description_zh: str


class TrendAnalyzer:
    """Detects frequency spikes in episodic event streams."""

    def analyze(
        self,
        episodes: list[Episode],
        window_hours: int = 1,
        baseline_hours: int = 24,
    ) -> list[Trend]:
        """Analyze episodes for frequency spikes.

        Compares the most recent *window_hours* against the average hourly
        rate over the preceding *baseline_hours*.

        Returns a list of Trend objects for event types that spike above
        ``_SPIKE_THRESHOLD``.
        """
        if not episodes:
            return []

        now = time.time()
        window_start = now - window_hours * 3600
        baseline_start = now - baseline_hours * 3600

        # Count events in recent window vs baseline period per event_type
        recent_counts: dict[str, int] = defaultdict(int)
        baseline_counts: dict[str, int] = defaultdict(int)

        for ep in episodes:
            ts = ep.timestamp
            if ts >= window_start:
                recent_counts[ep.event_type] += 1
            elif ts >= baseline_start:
                baseline_counts[ep.event_type] += 1

        # Compute baseline hourly rate
        baseline_window_hours = max((now - window_start - (now - baseline_start)) / 3600, 1.0)
        # Hours in baseline period excluding the recent window
        baseline_span = max(baseline_hours - window_hours, 1)

        trends: list[Trend] = []
        for event_type, count in recent_counts.items():
            bl_count = baseline_counts.get(event_type, 0)
            # Hourly rate in baseline period
            bl_rate = bl_count / baseline_span if baseline_span > 0 else 0.0

            # If baseline is zero but we have recent events, use 0.5 as floor
            effective_baseline = max(bl_rate, 0.5)
            # Recent hourly rate
            recent_rate = count / max(window_hours, 1)
            spike_ratio = recent_rate / effective_baseline

            if spike_ratio >= _SPIKE_THRESHOLD:
                desc = (
                    f"过去{window_hours}小时检测到{event_type}事件{count}次，"
                    f"是平均水平的{spike_ratio:.1f}倍"
                )
                trends.append(Trend(
                    window_start=window_start,
                    window_end=now,
                    event_type=event_type,
                    count=count,
                    baseline=effective_baseline,
                    spike_ratio=spike_ratio,
                    description_zh=desc,
                ))

        return trends

    def get_summary(self, episodes: list[Episode]) -> str:
        """Return a Chinese text summary of active trends for prompt injection.

        Returns empty string when no trends detected.
        """
        trends = self.analyze(episodes)
        if not trends:
            return ""
        lines = [t.description_zh for t in trends]
        return "\n".join(lines)
