"""Procedural memory — learns which approaches work best.

Tracks task procedures (patrol routes, inspection sequences) and their
success rates using Bayesian Beta posteriors. Over time, the robot
learns to prefer procedures that have historically succeeded.

Reference: MACLA (arXiv 2512.18950, Dec 2025)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Procedure:
    name: str                    # "navigate_warehouse_a_north_corridor"
    task_type: str               # "navigate", "inspect", "patrol"
    description: str             # "Go to Warehouse A via north corridor"
    alpha: float = 1.0           # Beta distribution: successes + 1
    beta: float = 1.0            # Beta distribution: failures + 1
    total_uses: int = 0
    avg_duration_s: float = 0.0
    last_used: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)  # conditions when this works

    @property
    def success_rate(self) -> float:
        """Expected success rate: E[Beta(alpha, beta)] = alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def confidence(self) -> float:
        """How confident are we? More data = higher confidence."""
        return 1.0 - 1.0 / (1.0 + self.total_uses)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task_type": self.task_type,
            "description": self.description,
            "alpha": self.alpha,
            "beta": self.beta,
            "total_uses": self.total_uses,
            "avg_duration_s": self.avg_duration_s,
            "last_used": self.last_used,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Procedure:
        return cls(
            name=data["name"],
            task_type=data.get("task_type", ""),
            description=data.get("description", ""),
            alpha=data.get("alpha", 1.0),
            beta=data.get("beta", 1.0),
            total_uses=data.get("total_uses", 0),
            avg_duration_s=data.get("avg_duration_s", 0.0),
            last_used=data.get("last_used", 0.0),
            context=data.get("context", {}),
        )


class ProceduralMemory:
    """Learns which procedures work best for which tasks.

    Usage::

        pm = ProceduralMemory()
        pm.record_outcome("nav_warehouse_a_north", "navigate", success=True, duration=120)
        pm.record_outcome("nav_warehouse_a_south", "navigate", success=False, duration=0)
        best = pm.get_best_procedure("navigate")
        # -> "nav_warehouse_a_north" (higher success rate)
    """

    def __init__(self, data_dir: str = "data/memory/procedures") -> None:
        self._procedures: dict[str, Procedure] = {}
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def record_outcome(self, name: str, task_type: str, *,
                       success: bool, duration: float = 0.0,
                       description: str = "", context: dict[str, Any] | None = None) -> None:
        """Record a procedure outcome. Creates procedure if new, updates Beta posterior."""
        proc = self._procedures.get(name)
        if proc is None:
            proc = Procedure(name=name, task_type=task_type, description=description)
            self._procedures[name] = proc
        if success:
            proc.alpha += 1.0
        else:
            proc.beta += 1.0
        proc.total_uses += 1
        proc.last_used = time.time()
        if duration > 0:
            # Incremental mean update
            if proc.avg_duration_s == 0.0:
                proc.avg_duration_s = duration
            else:
                proc.avg_duration_s += (duration - proc.avg_duration_s) / proc.total_uses
        if description:
            proc.description = description
        if context:
            proc.context.update(context)
        self._save()

    def get_best_procedure(self, task_type: str,
                           min_confidence: float = 0.3) -> Procedure | None:
        """Get the highest-scoring procedure for a task type.

        Score = success_rate * confidence (exploitation + exploration balance).
        """
        candidates = [p for p in self._procedures.values()
                       if p.task_type == task_type and p.confidence >= min_confidence]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.success_rate * p.confidence)

    def get_procedures(self, task_type: str = "") -> list[Procedure]:
        """Get all procedures, optionally filtered by task_type."""
        if not task_type:
            return list(self._procedures.values())
        return [p for p in self._procedures.values() if p.task_type == task_type]

    def get_context(self) -> str:
        """Get procedural context for system prompt: known procedures + success rates."""
        if not self._procedures:
            return ""
        parts: list[str] = ["[已学习的操作流程]"]
        for proc in self._procedures.values():
            rate_pct = proc.success_rate * 100
            line = f"- {proc.name} ({proc.task_type}): 成功率{rate_pct:.0f}%, 使用{proc.total_uses}次"
            if proc.description:
                line += f" — {proc.description}"
            parts.append(line)
        return "\n".join(parts)

    def _save(self) -> None:
        """Persist to JSON."""
        path = self._data_dir / "procedures.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump([p.to_dict() for p in self._procedures.values()],
                          f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("ProceduralMemory save failed: %s", e)

    def _load(self) -> None:
        """Load from JSON."""
        path = self._data_dir / "procedures.json"
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    for item in json.load(f):
                        proc = Procedure.from_dict(item)
                        self._procedures[proc.name] = proc
            except Exception as e:
                logger.warning("ProceduralMemory load failed: %s", e)
