"""Robot-aware planning adapter.

The planner turns operator language plus fresh context into auditable planning
sessions and mission drafts. It never dispatches hardware actions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from askme.cognition.memory import WorkingMemory
from askme.cognition.planning.planning_session import PlanningSession
from askme.cognition.world import WorldStateService

_CONFIRMATION_INPUT = "operator_confirmation"
_MISSION_DRAFT_INPUT = "mission_draft"
_PHYSICAL_INTENTS = frozenset({
    "capture_evidence",
    "inspection_patrol",
    "manipulation",
    "navigation",
    "visitor_escort",
})
_INFORMATION_RESPONSE_INTENTS = frozenset({
    "visitor_wayfinding",
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
    readiness: dict[str, Any] = field(default_factory=dict)
    handoff_contract: dict[str, Any] = field(default_factory=dict)
    world_state_snapshot_id: str = ""
    session: dict[str, Any] = field(default_factory=dict)
    conversation_session_id: str = ""
    operator_id: str = ""
    robot_id: str = ""
    site_id: str = ""
    channel: str = "cognition"
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
            "readiness": dict(self.readiness),
            "handoff_contract": dict(self.handoff_contract),
            "world_state_snapshot_id": self.world_state_snapshot_id,
            "reference": dict(self.reference),
            "context": dict(self.context),
            "safety_constraints": list(self.safety_constraints),
            "steps": [dict(step) for step in self.steps],
            "mission": self.mission,
            "session": dict(self.session),
            "conversation_session_id": self.conversation_session_id,
            "operator_id": self.operator_id,
            "robot_id": self.robot_id,
            "site_id": self.site_id,
            "channel": self.channel,
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
        conversation_session_id: str | None = None,
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
                conversation_session_id=conversation_session_id,
            )

        reference = self.world_state.resolve_reference(requested_goal)
        missing_inputs = _missing_inputs_for(requested_goal, intent, reference)
        requires_clarification = bool(missing_inputs)
        stage = "clarifying" if requires_clarification else "drafting"
        mission = None

        needs_mission_draft = _should_draft_mission(intent)
        if not requires_clarification and needs_mission_draft:
            if self.mission_service is not None:
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
            if not _mission_draft_ready(mission):
                missing_inputs = _unique([*missing_inputs, _MISSION_DRAFT_INPUT])
                requires_clarification = True

        if not requires_clarification:
            if intent in _INFORMATION_RESPONSE_INTENTS:
                stage = "answer_ready"
            elif _requires_operator_confirmation(intent):
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
            else "not_required"
            if stage == "answer_ready"
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

        conversation_session = str(conversation_session_id or "")
        context = {
            "world_summary": self.world_state.context_summary(),
            "working_memory": self.working_memory.summary(
                conversation_session_id=conversation_session or None
            ),
            "reply_to_plan_id": str(reply_to_plan_id or ""),
            "conversation_session_id": conversation_session,
        }
        self.working_memory.record_turn(
            requested_goal,
            task_id=plan_id,
            conversation_session_id=conversation_session,
        )
        self.working_memory.set_focus(
            conversation_session_id=conversation_session,
            last_plan_id=plan_id,
            last_intent=intent,
            planning_session_id=active_session.session_id,
            planning_stage=stage,
        )
        world_snapshot = self.world_state.snapshot(include_stale=False)
        world_snapshot_id = _world_state_snapshot_id(world_snapshot)

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
            readiness=_readiness_for(
                stage,
                intent,
                missing_inputs=missing_inputs,
                handoff_ready=handoff_ready,
                next_prompt=next_prompt,
            ),
            handoff_contract=_handoff_contract_for(
                plan_id,
                active_session.session_id,
                intent,
                stage=stage,
                missing_inputs=missing_inputs,
                handoff_ready=handoff_ready,
                world_state_snapshot=world_snapshot,
            ),
            world_state_snapshot_id=world_snapshot_id,
            session=active_session.to_dict(),
            conversation_session_id=str(conversation_session_id or ""),
            operator_id=str(operator_id or ""),
            robot_id=str(robot_id or ""),
            site_id=str(site_id or ""),
            channel=str(channel or "cognition"),
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
        conversation_session_id: str | None = None,
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
            conversation_session_id=str(conversation_session_id or ""),
            last_plan_id=plan_id,
            last_intent=intent,
            planning_session_id=session.session_id,
            planning_stage="cancelled",
        )
        world_snapshot = self.world_state.snapshot(include_stale=False)
        world_snapshot_id = _world_state_snapshot_id(world_snapshot)
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
                "working_memory": self.working_memory.summary(
                    conversation_session_id=str(conversation_session_id or "") or None
                ),
                "conversation_session_id": str(conversation_session_id or ""),
            },
            safety_constraints=_safety_constraints(intent),
            steps=[{"step": "cancel_planning_session", "status": "done"}],
            mission=None,
            readiness=_readiness_for(
                "cancelled",
                intent,
                missing_inputs=[],
                handoff_ready=False,
                next_prompt="planning_cancelled",
            ),
            handoff_contract=_handoff_contract_for(
                plan_id,
                session.session_id,
                intent,
                stage="cancelled",
                missing_inputs=[],
                handoff_ready=False,
                world_state_snapshot=world_snapshot,
            ),
            world_state_snapshot_id=world_snapshot_id,
            session=session.to_dict(),
            conversation_session_id=str(conversation_session_id or ""),
            operator_id=str(operator_id or ""),
            robot_id=str(robot_id or ""),
            site_id=str(site_id or ""),
            channel=str(channel or "cognition"),
        )


def _intent_for(text: str) -> str:
    lowered = text.lower()
    if any(marker in lowered for marker in ("急停", "停止", "停下")):
        return "safety_stop"
    if any(marker in lowered for marker in ("带我去", "带路", "请带我", "跟你走", "escort me", "lead me")):
        return "visitor_escort"
    if _looks_like_wayfinding_question(lowered):
        return "visitor_wayfinding"
    if any(marker in lowered for marker in ("巡检", "检查", "巡逻")):
        return "inspection_patrol"
    if any(marker in lowered for marker in ("拍照", "截图", "抓拍", "取证")):
        return "capture_evidence"
    if any(marker in lowered for marker in ("状态", "电量", "报告")):
        return "status_report"
    if any(marker in lowered for marker in ("导航", "前往", "去", "到")):
        return "navigation"
    if any(marker in lowered for marker in ("拿", "抓", "夹取")):
        return "manipulation"
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


def _looks_like_wayfinding_question(lowered: str) -> bool:
    if any(marker in lowered for marker in ("怎么走", "在哪", "在哪里", "路线", "怎么去", "指路", "问路", "nearest", "where is", "how do i get")):
        return True
    if any(marker in lowered for marker in ("厕所", "卫生间", "咖啡", "停车场", "出口", "西门", "东门", "南门", "北门")):
        return any(marker in lowered for marker in ("在哪", "怎么", "找", "where", "route"))
    return False


def _missing_inputs_for(
    text: str,
    intent: str,
    reference: dict[str, Any],
) -> list[str]:
    missing: list[str] = []
    if reference.get("needs_clarification"):
        missing.append("scene_reference")
    if _is_underspecified(text, intent):
        if intent in {"navigation", "visitor_escort", "visitor_wayfinding"}:
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
    return intent in {"navigation", "visitor_escort", "visitor_wayfinding", "manipulation"} and len(compact) < 4


def _requires_operator_confirmation(intent: str) -> bool:
    return intent in _PHYSICAL_INTENTS


def _should_draft_mission(intent: str) -> bool:
    return intent not in _INFORMATION_RESPONSE_INTENTS


def _mission_draft_ready(mission: dict[str, Any] | None) -> bool:
    if not isinstance(mission, dict):
        return False
    body = mission.get("mission")
    return bool(mission.get("drafted") is True and isinstance(body, dict) and body.get("mission_id"))


def _confirmation_value(value: Any | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"确认", "可以", "是", "继续"}:
        return True
    if text in {"取消", "不是", "不用", "否"}:
        return False
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
        return "已取消当前任务规划。"
    if stage == "answer_ready":
        return _information_response_prompt(intent)
    if stage == "awaiting_confirmation":
        return _confirmation_prompt(intent)
    if "scene_reference" in missing_inputs or reference.get("reason") == "no_fresh_scene_object":
        return "我还不能确定你指的是哪个目标，请说出目标名称，或让我先刷新现场画面。"
    if "target_location" in missing_inputs:
        return "请补充目标位置或路线名称。"
    if "target_object" in missing_inputs:
        return "请补充要操作的目标物和期望动作。"
    if _MISSION_DRAFT_INPUT in missing_inputs:
        return "任务草案还没有生成，暂不能交给运行系统。请检查任务规划服务后再确认。"
    if missing_inputs:
        return "请补充目标、位置或约束条件。"
    if stage == "ready_for_arbiter":
        return "计划已确认，可以交给运行系统做安全检查和任务调度。"
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
    if _MISSION_DRAFT_INPUT in missing_inputs:
        return "任务草案还没有生成，暂不能交给运行系统。请检查任务规划服务后再确认。"
    if missing_inputs:
        return "请补充目标、位置或约束条件。"
    if stage == "ready_for_arbiter":
        return "计划已确认，可以交给运行时仲裁器继续处理。"
    return ""


def _information_response_prompt(intent: str) -> str:
    if intent == "visitor_wayfinding":
        return "这是问路咨询，我会根据园区知识库回答，不会启动机器狗移动；如需带路，请明确说“请带我去目的地”。"
    return "这是信息咨询，我会直接回答，不会启动机器人任务。"


def _confirmation_prompt(intent: str) -> str:
    if intent in {"navigation", "inspection_patrol", "visitor_escort"}:
        return "已生成移动/巡检任务草案，请确认后再交给运行系统做安全检查和调度。"
    if intent == "manipulation":
        return "已生成操作任务草案，请确认目标和动作后再交给运行系统。"
    if intent == "capture_evidence":
        return "已生成取证任务草案，请确认后再执行采集。"
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
    if intent in _INFORMATION_RESPONSE_INTENTS:
        steps.append({
            "step": "answer_with_grounded_park_knowledge",
            "status": "ready" if stage == "answer_ready" else "blocked",
        })
        return steps
    if _MISSION_DRAFT_INPUT in missing_inputs:
        steps.append({"step": "mission_draft", "status": "unavailable"})
        steps.append({"step": "submit_to_arbiter", "status": "blocked_missing_mission_draft"})
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
    if intent == "visitor_wayfinding":
        constraints.append("Wayfinding answers must stay grounded in the park knowledge base and must not start movement.")
    if intent == "visitor_escort":
        constraints.append("Visitor escort requires destination confirmation, safe route availability, and runtime preflight.")
    if intent == "manipulation":
        constraints.append("Manipulation requires target confirmation and control-service arbitration.")
    if intent == "safety_stop":
        constraints.append("Use the dedicated safety/E-STOP path, not a general mission draft.")
    return constraints


def _readiness_for(
    stage: str,
    intent: str,
    *,
    missing_inputs: list[str],
    handoff_ready: bool,
    next_prompt: str,
) -> dict[str, Any]:
    blocked_by = list(missing_inputs)
    status = _readiness_status(stage, blocked_by, handoff_ready)
    return {
        "status": status,
        "customer_status": _customer_status_for(status),
        "handoff_ready": handoff_ready,
        "can_submit_to_runtime": handoff_ready,
        "blocked_by": blocked_by,
        "allowed_next_actions": _allowed_next_actions(status),
        "requires_operator_confirmation": _requires_operator_confirmation(intent),
        "requires_safety_preflight": _requires_safety_preflight(intent),
        "runtime_gate": "safety_preflight_required" if _requires_safety_preflight(intent) else "none",
        "next_prompt": next_prompt,
    }


def _handoff_contract_for(
    plan_id: str,
    planning_session_id: str,
    intent: str,
    *,
    stage: str,
    missing_inputs: list[str],
    handoff_ready: bool,
    world_state_snapshot: dict[str, Any],
) -> dict[str, Any]:
    return {
        "contract_version": "task_plan.v1",
        "producer": "askme.cognition",
        "consumer": "runtime_handoff",
        "plan_id": plan_id,
        "planning_session_id": planning_session_id,
        "handoff_state": stage,
        "handoff_ready": handoff_ready,
        "confirmed": handoff_ready,
        "blocked_by": list(missing_inputs),
        "world_state_snapshot_id": _world_state_snapshot_id(world_state_snapshot),
        "dispatch_authority": "runtime_arbiter_only",
        "can_dispatch_hardware": False,
        "requires_operator_confirmation": _requires_operator_confirmation(intent),
        "requires_safety_preflight": _requires_safety_preflight(intent),
        "submit_conditions": _submit_conditions(intent),
    }


def _readiness_status(stage: str, blocked_by: list[str], handoff_ready: bool) -> str:
    if stage == "cancelled":
        return "cancelled"
    if stage == "answer_ready":
        return "ready_to_answer"
    if handoff_ready:
        return "ready_for_runtime_handoff"
    if "scene_reference" in blocked_by:
        return "needs_fresh_perception"
    if blocked_by == [_CONFIRMATION_INPUT]:
        return "awaiting_operator_confirmation"
    if blocked_by:
        return "needs_operator_input"
    return "drafting"


def _customer_status_for(status: str) -> str:
    if status == "ready_to_answer":
        return "这是园区问询回答，不会启动机器人任务。"
    messages = {
        "cancelled": "计划已取消。",
        "ready_for_runtime_handoff": "已确认，可以进入运行前安全检查。",
        "needs_fresh_perception": "需要先刷新现场画面，确认目标后再继续。",
        "awaiting_operator_confirmation": "等待操作员确认，确认后才会交给运行系统。",
        "needs_operator_input": "还需要补充目标、位置或约束信息。",
        "drafting": "正在生成任务草案。",
    }
    return messages.get(status, messages["drafting"])


def _allowed_next_actions(status: str) -> list[str]:
    if status == "ready_to_answer":
        return ["answer_with_grounded_knowledge", "offer_escort_after_confirmation", "start_new_plan"]
    if status == "ready_for_runtime_handoff":
        return ["submit_to_runtime_arbiter", "revise_plan", "cancel_plan"]
    if status == "awaiting_operator_confirmation":
        return ["confirm_plan", "revise_plan", "cancel_plan"]
    if status == "needs_fresh_perception":
        return ["refresh_perception", "answer_clarifying_question", "cancel_plan"]
    if status == "needs_operator_input":
        return ["answer_clarifying_question", "revise_plan", "cancel_plan"]
    if status == "cancelled":
        return ["start_new_plan"]
    return ["continue_planning", "cancel_plan"]


def _requires_safety_preflight(intent: str) -> bool:
    return intent in _PHYSICAL_INTENTS or intent == "safety_stop"


def _submit_conditions(intent: str) -> list[str]:
    if intent in _INFORMATION_RESPONSE_INTENTS:
        return [
            "no_runtime_handoff_for_information_response",
            "answer_must_use_grounded_park_knowledge",
        ]
    conditions = [
        "operator_confirmation_status_confirmed",
        "missing_inputs_empty",
        "task_steps_are_high_level_skills",
    ]
    if _requires_safety_preflight(intent):
        conditions.append("runtime_safety_preflight_passed")
    return conditions


def _world_state_snapshot_id(snapshot: dict[str, Any]) -> str:
    updated_at = float(snapshot.get("updated_at", time.time()) or time.time())
    fact_count = int(snapshot.get("fact_count", 0) or 0)
    return f"world-{int(updated_at * 1000)}-{fact_count}"


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values
