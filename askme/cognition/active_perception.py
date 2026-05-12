"""Active perception refresh loop for cognition planning.

The resolver is deliberately local and dependency-free: it records the need for
fresh scene facts, calls an injected refresh function, and lets the caller rerun
planning with the updated world state.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from askme.cognition.planner import CognitivePlan
from askme.cognition.world_state import WorldStateService

RefreshCallback = Callable[[dict[str, Any]], Any]
ReplanCallback = Callable[[], CognitivePlan]


@dataclass(frozen=True)
class ActivePerceptionRequest:
    """A request for fresh perception facts needed by planning."""

    request_id: str
    reason: str
    goal: str
    missing_inputs: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "reason": self.reason,
            "goal": self.goal,
            "missing_inputs": list(self.missing_inputs),
            "created_at": self.created_at,
        }


class ActivePerceptionResolver:
    """Request fresh perception and rerun planning when scene facts are stale."""

    def __init__(
        self,
        *,
        world_state: WorldStateService,
        refresh: RefreshCallback | None = None,
        source: str = "active_perception_resolver",
    ) -> None:
        self.world_state = world_state
        self.refresh = refresh
        self.source = source
        self._requests: list[ActivePerceptionRequest] = []

    @property
    def requests(self) -> list[dict[str, Any]]:
        return [request.to_dict() for request in self._requests]

    def needs_refresh(self, plan: CognitivePlan) -> bool:
        reference = plan.reference or {}
        return (
            "scene_reference" in plan.missing_inputs
            and reference.get("reason") == "no_fresh_scene_object"
        )

    async def resolve(
        self,
        initial_plan: CognitivePlan,
        *,
        replan: ReplanCallback,
        refresh_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the current plan, or refresh perception and return a rerun plan."""
        if not self.needs_refresh(initial_plan):
            return {
                "plan": initial_plan,
                "active_perception": {
                    "requested": False,
                    "reason": "fresh_perception_not_required",
                    "requests": self.requests,
                },
            }

        request = ActivePerceptionRequest(
            request_id=f"apr-{uuid4().hex[:12]}",
            reason=str(initial_plan.reference.get("reason") or "missing_fresh_scene_fact"),
            goal=initial_plan.goal,
            missing_inputs=list(initial_plan.missing_inputs),
        )
        self._requests.append(request)
        self.world_state.record_event(
            "perception_refresh_requested",
            request.to_dict(),
            source=self.source,
            observed_at=request.created_at,
        )
        self.world_state.update_fact(
            "task.last_perception_refresh_request",
            request.to_dict(),
            source=self.source,
            observed_at=request.created_at,
            stale_after_s=300.0,
        )

        refresh_result = await self._call_refresh({
            "request": request.to_dict(),
            **dict(refresh_context or {}),
        })
        rerun_plan = replan()
        return {
            "plan": rerun_plan,
            "active_perception": {
                "requested": True,
                "request": request.to_dict(),
                "requests": self.requests,
                "refresh": refresh_result,
                "replanned": True,
                "resolved_after_refresh": not rerun_plan.requires_clarification,
                "missing_inputs_after_refresh": list(rerun_plan.missing_inputs),
            },
        }

    async def _call_refresh(self, context: dict[str, Any]) -> dict[str, Any]:
        if self.refresh is None:
            return {"refreshed": False, "reason": "refresh_unavailable"}
        try:
            result = self.refresh(context)
            value = await result if inspect.isawaitable(result) else result
        except Exception as exc:
            return {"refreshed": False, "reason": str(exc)}
        return value if isinstance(value, dict) else {"refreshed": bool(value), "value": value}
