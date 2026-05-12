"""Robot-aware planning adapter.

The planner turns operator language plus fresh context into auditable planning
sessions and mission drafts. It never dispatches hardware actions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from askme.cognition.planning_session import PlanningSession
from askme.cognition.working_memory import WorkingMemory
from askme.cognition.world_state import WorldStateService

_CONFIRMATION_INPUT = "operator_confirmation"
_PHYSICAL_INTENTS = frozenset({
    "capture_evidence",
    "inspection_patrol",
    "manipulation",
    "navigation",
})


@dataclass(frozen=True)
class CognitivePlan:
    """A safe, operator-facing plan proposal."""

    plan_id: str
    goal: str
    intent: str
    interaction_state: str
    requires_clarification: bool
    clarification_question: str = ""
    reference: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    safety_constraints: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    mission: dict[str, Any] | None = None
    planning_session_id: str = ""
    stage: str = ""
    missing_inputs: list[str] = field(default_factory=list)
    next_prompt: str = ""
    handoff_ready: bool = False
    session: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "planning_session_id": self.planning_session_id,
            "goal": self.goal,
            "intent": self.intent,
            "stage": self.stage or self.interaction_state,
            "interaction_state": self.interaction_state,
            "requires_clarification": self.requires_clarification,
            "clarification_question": self.clarification_question,
            "missing_inputs": list(self.missing_inputs),
            "next_prompt": self.next_prompt,
            "handoff_ready": self.handoff_ready,
            "reference": dict(self.reference),
            "context": dict(self.context),
            "safety_constraints": list(self.safety_constraints),
            "steps": [dict(step) for step in self.steps],
            "mission": self.mission,
            "session": dict(self.session),
            "created_at": self.created_at,
        }


class CognitivePlanner:
    """Deterministic planning scaffold for voice/text robot requests."""

    def __init__(
        self,
        *,
        world_state: WorldStateService,
        working_memory: WorkingMemory,
        mission_service: Any | None = None,
        max_sessions: int = 20,
    ) -> None:
        self.world_state = world_state
        self.working_memory = working_memory
        self.mission_service = mission_service
        self.max_sessions = max(1, int(max_sessions))
        self._sessions: dict[str, PlanningSession] = {}

    def plan_from_text(
        self,
        text: str,
        *,
        operator_id: str | None = None,
        robot_id: str | None = None,
        site_id: str | None = None,
        channel: str = "cognition",
        metadata: dict[str, Any] | None = None,
        planning_session_id: str | None = None,
        reply_to_plan_id: str | None = None,
        operator_confirmation: Any | None = None,
        cancel: bool = False,
        revise_goal: str | None = None,
    ) -> CognitivePlan:
        requested_goal = str(revise_goal or text or "").strip()
        session = self._session_for(planning_session_id)
        if not requested_goal and session is not None:
            requested_goal = session.goal
        if not requested_goal:
            raise ValueError("planning text is required")

        plan_id = f"cog-{uuid4().hex[:12]}"
        active_session = session or self._new_session(planning_session_id)
        intent = _intent_for(requested_goal)
        confirmation = _confirmation_value(operator_confirmation)
        metadata = metadata or {}

        if cancel:
            return self._cancel_plan(
                active_session,
                plan_id=plan_id,
                goal=requested_goal,
                intent=intent,
                operator_id=operator_id,
                robot_id=robot_id,
                site_id=site_id,
                channel=channel,
            )

        reference = self.world_state.resolve_reference(requested_goal)
        missing_inputs = _missing_inputs_for(requested_goal, intent, reference)
        requires_clarification = bool(missing_inputs)
        stage = "clarifying" if requires_clarification else "drafting"
        mission = None

        if not requires_clarification and self.mission_service is not None:
            mission = self._draft_mission(
                requested_goal,
                plan_id=plan_id,
                intent=intent,
                operator_id=operator_id,
                robot_id=robot_id,
                site_id=site_id,
                channel=channel,
                metadata=metadata,
                session_id=active_session.session_id,
            )

        if not requires_clarification:
            if _requires_operator_confirmation(intent):
                if confirmation is True:
                    stage = "ready_for_arbiter"
                else:
                    stage = "awaiting_confirmation"
                    missing_inputs = [_CONFIRMATION_INPUT]
            else:
                stage = "ready_for_arbiter"

        handoff_ready = stage == "ready_for_arbiter" and not missing_inputs
        confirmation_status = (
            "confirmed"
            if handoff_ready
            else "cancelled"
            if stage == "cancelled"
            else "unconfirmed"
        )
        next_prompt = _next_prompt(stage, intent, missing_inputs, reference)
        clarification_question = next_prompt if requires_clarification else ""

        active_session.update(
            plan_id=plan_id,
            goal=requested_goal,
            intent=intent,
            stage=stage,
            missing_inputs=missing_inputs,
            reference=reference,
            mission=mission,
            confirmation_status=confirmation_status,
            operator_id=operator_id,
            robot_id=robot_id,
            site_id=site_id,
            channel=channel,
        )
        self._trim_sessions()

        context = {
            "world_summary": self.world_state.context_summary(),
            "working_memory": self.working_memory.summary(),
            "reply_to_plan_id": str(reply_to_plan_id or ""),
        }
        self.working_memory.record_turn(requested_goal, task_id=plan_id)
        self.working_memory.set_focus(
            last_plan_id=plan_id,
            last_intent=intent,
            planning_session_id=active_session.session_id,
            planning_stage=stage,
        )

        plan = CognitivePlan(
            plan_id=plan_id,
            planning_session_id=active_session.session_id,
            goal=requested_goal,
            intent=intent,
            stage=stage,
            interaction_state=stage,
            requires_clarification=requires_clarification,
            clarification_question=clarification_question,
            missing_inputs=missing_inputs,
            next_prompt=next_prompt,
            handoff_ready=handoff_ready,
            reference=reference,
            context=context,
            safety_constraints=_safety_constraints(intent),
            steps=_steps_for(
                intent,
                stage=stage,
                missing_inputs=missing_inputs,
                mission=mission,
                handoff_ready=handoff_ready,
            ),
            mission=mission,
            session=active_session.to_dict(),
        )
        self.world_state.update_fact(
            "task.last_plan",
            {
                "plan_id": plan.plan_id,
                "planning_session_id": plan.planning_session_id,
                "intent": plan.intent,
                "state": plan.interaction_state,
                "handoff_ready": plan.handoff_ready,
            },
            source="cognitive_planner",
            stale_after_s=300.0,
        )
        return plan

    def context_payload(self) -> dict[str, Any]:
        return {
            "world_state": self.world_state.snapshot(),
            "working_memory": self.working_memory.snapshot(),
            "planning_sessions": [
                session.to_dict()
                for session in sorted(
                    self._sessions.values(),
                    key=lambda item: item.updated_at,
                    reverse=True,
                )
            ],
        }

    def _draft_mission(
        self,
        goal: str,
        *,
        plan_id: str,
        intent: str,
        operator_id: str | None,
        robot_id: str | None,
        site_id: str | None,
        channel: str,
        metadata: dict[str, Any],
        session_id: str,
    ) -> dict[str, Any]:
        payload = {
            "text": goal,
            "operator_id": operator_id,
            "robot_id": robot_id,
            "site_id": site_id,
            "channel": channel,
            "metadata": {
                **metadata,
                "_cognition": {
                    "plan_id": plan_id,
                    "planning_session_id": session_id,
                    "intent": intent,
                    "world_summary": self.world_state.context_summary(),
                },
            },
        }
        clean_payload = {key: value for key, value in payload.items() if value not in (None, "")}
        draft = getattr(self.mission_service, "draft_from_payload", None)
        if not callable(draft):
            return {}
        result = draft(clean_payload)
        return result if isinstance(result, dict) else {}

    def _session_for(self, planning_session_id: str | None) -> PlanningSession | None:
        cleaned = str(planning_session_id or "").strip()
        return self._sessions.get(cleaned) if cleaned else None

    def _new_session(self, planning_session_id: str | None) -> PlanningSession:
        session = PlanningSession.create(str(planning_session_id or "").strip() or None)
        self._sessions[session.session_id] = session
        return session

    def _trim_sessions(self) -> None:
        if len(self._sessions) <= self.max_sessions:
            return
        oldest = sorted(self._sessions.values(), key=lambda item: item.updated_at)
        for session in oldest[: len(self._sessions) - self.max_sessions]:
            self._sessions.pop(session.session_id, None)

    def _cancel_plan(
        self,
        session: PlanningSession,
        *,
        plan_id: str,
        goal: str,
        intent: str,
        operator_id: str | None,
        robot_id: str | None,
        site_id: str | None,
        channel: str,
    ) -> CognitivePlan:
        session.update(
            plan_id=plan_id,
            goal=goal,
            intent=intent,
            stage="cancelled",
            missing_inputs=[],
            reference={},
            mission=None,
            confirmation_status="cancelled",
            operator_id=operator_id,
            robot_id=robot_id,
            site_id=site_id,
            channel=channel,
        )
        self.working_memory.set_focus(
            last_plan_id=plan_id,
            last_intent=intent,
            planning_session_id=session.session_id,
            planning_stage="cancelled",
        )
        return CognitivePlan(
            plan_id=plan_id,
            planning_session_id=session.session_id,
            goal=goal,
            intent=intent,
            stage="cancelled",
            interaction_state="cancelled",
            requires_clarification=False,
            missing_inputs=[],
            next_prompt="已取消当前规划。",
            handoff_ready=False,
            reference={},
            context={
                "world_summary": self.world_state.context_summary(),
                "working_memory": self.working_memory.summary(),
            },
            safety_constraints=_safety_constraints(intent),
            steps=[{"step": "cancel_planning_session", "status": "done"}],
            mission=None,
            session=session.to_dict(),
        )


def _intent_for(text: str) -> str:
    lowered = text.lower()
    if any(marker in lowered for marker in ("急停", "停止", "停下", "stop", "estop", "e-stop")):
        return "safety_stop"
    if any(marker in lowered for marker in ("巡检", "检查", "巡逻", "patrol", "inspect")):
        return "inspection_patrol"
    if any(marker in lowered for marker in ("拍照", "截图", "抓拍", "取证", "photo", "snapshot", "capture")):
        return "capture_evidence"
    if any(marker in lowered for marker in ("状态", "电量", "status", "battery")):
        return "status_report"
    if any(marker in lowered for marker in ("导航", "去", "到", "go to", "navigate")):
        return "navigation"
    if any(marker in lowered for marker in ("拿", "抓", "夹取", "pick", "grab")):
        return "manipulation"
    return "operator_assist"


def _missing_inputs_for(
    text: str,
    intent: str,
    reference: dict[str, Any],
) -> list[str]:
    missing: list[str] = []
    if reference.get("needs_clarification"):
        missing.append("scene_reference")
    if _is_underspecified(text, intent):
        if intent == "navigation":
            missing.append("target_location")
        elif intent == "manipulation":
            missing.append("target_object")
        else:
            missing.append("task_details")
    return _unique(missing)


def _is_underspecified(text: str, intent: str) -> bool:
    compact = "".join(str(text).split())
    if len(compact) < 2:
        return True
    return intent in {"navigation", "manipulation"} and len(compact) < 4


def _requires_operator_confirmation(intent: str) -> bool:
    return intent in _PHYSICAL_INTENTS


def _confirmation_value(value: Any | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "confirm", "confirmed", "ok", "确认", "是", "可以"}:
        return True
    if text in {"0", "false", "no", "n", "cancel", "取消", "否"}:
        return False
    return None


def _next_prompt(
    stage: str,
    intent: str,
    missing_inputs: list[str],
    reference: dict[str, Any],
) -> str:
    if stage == "cancelled":
        return "已取消当前规划。"
    if stage == "awaiting_confirmation":
        return _confirmation_prompt(intent)
    if "scene_reference" in missing_inputs or reference.get("reason") == "no_fresh_scene_object":
        return "我还不能确定你指的是哪个目标，请说出目标名称，或让我先看一眼当前画面。"
    if "target_location" in missing_inputs:
        return "请补充目标位置或路线名称。"
    if "target_object" in missing_inputs:
        return "请补充要操作的目标物和期望动作。"
    if missing_inputs:
        return "请补充目标、位置或约束条件。"
    if stage == "ready_for_arbiter":
        return "计划已确认，可以交给运行时仲裁器继续处理。"
    return ""


def _confirmation_prompt(intent: str) -> str:
    if intent in {"navigation", "inspection_patrol"}:
        return "已生成移动/巡检任务草案，请确认后再交给运行时仲裁器。"
    if intent == "manipulation":
        return "已生成操作任务草案，请确认目标和动作后再交给运行时仲裁器。"
    if intent == "capture_evidence":
        return "已生成取证任务草案，请确认后再执行采集。"
    return "请确认是否继续。"


def _steps_for(
    intent: str,
    *,
    stage: str,
    missing_inputs: list[str],
    mission: dict[str, Any] | None,
    handoff_ready: bool,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = [
        {"step": "observe_context", "status": "done"},
        {
            "step": "bind_world_reference",
            "status": "needs_input" if "scene_reference" in missing_inputs else "done",
        },
    ]
    if stage == "cancelled":
        steps.append({"step": "cancel_planning_session", "status": "done"})
        return steps
    if missing_inputs and missing_inputs != [_CONFIRMATION_INPUT]:
        steps.append({"step": "ask_clarifying_question", "status": "ready"})
        return steps
    if mission is not None:
        steps.append({
            "step": "mission_draft",
            "status": "created",
            "mission_id": mission.get("mission", {}).get("mission_id"),
        })
    if _requires_operator_confirmation(intent):
        steps.append({
            "step": "await_operator_confirmation",
            "status": "done" if handoff_ready else "required",
        })
    steps.append({
        "step": "submit_to_arbiter",
        "status": "ready" if handoff_ready else "blocked_until_confirmed",
    })
    if intent in {"navigation", "inspection_patrol", "manipulation"}:
        steps.append({"step": "runtime_safety_evaluation", "status": "required"})
    return steps


def _safety_constraints(intent: str) -> list[str]:
    constraints = [
        "Do not dispatch hardware actions from the LLM or voice layer.",
        "Submit only high-level mission drafts to the runtime arbiter.",
        "Require operator confirmation before physical movement or manipulation.",
    ]
    if intent in {"navigation", "inspection_patrol"}:
        constraints.append("Navigation must pass geofence, estop, and path safety checks.")
    if intent == "manipulation":
        constraints.append("Manipulation requires target confirmation and control-service arbitration.")
    if intent == "safety_stop":
        constraints.append("Use the dedicated safety/E-STOP path, not a general mission draft.")
    return constraints


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values
