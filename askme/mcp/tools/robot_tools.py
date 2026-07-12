"""MCP tools for robot arm control.

Safety architecture (Codex review, 2026-06-01):
  Low-level tools (move/pick/place/home/wave) require ``ASKME_LAB_UNSAFE_TOOLS=true``.
  Production callers (ZeroClaw) must use ``robot_submit_task``, which routes through
  TaskHandoff → SafetyPreflight → runtime arbiter when the full runtime is available.
  ``robot_state`` is always read-only.  ``robot_estop`` is always permitted.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os

from mcp.server.fastmcp import Context

from askme.errors import ROBOT_NOT_CONNECTED, error_response
from askme.mcp.context import AppContext
from askme.mcp.registration import mcp

logger = logging.getLogger(__name__)

_LAB_UNSAFE = os.environ.get("ASKME_LAB_UNSAFE_TOOLS", "").lower() in ("1", "true", "yes")
_UNSAFE_BLOCKED_MSG = json.dumps({
    "error": "unsafe_direct_arm_access_blocked",
    "message": (
        "Direct arm motion tools are gated behind ASKME_LAB_UNSAFE_TOOLS=true. "
        "Use robot_submit_task for production workflows — it routes through "
        "TaskHandoff → SafetyPreflight → runtime arbiter."
    ),
})


def _get_app(ctx: Context) -> AppContext:
    return ctx.request_context.lifespan_context


def _no_robot() -> str:
    return error_response(ROBOT_NOT_CONNECTED, "Robot arm not connected or not enabled")


def _require_unsafe() -> str | None:
    """Return an error payload when unsafe direct-arm tools are blocked."""
    if not _LAB_UNSAFE:
        return _UNSAFE_BLOCKED_MSG
    return None


async def _execute_arm(app: AppContext, action: str, params: dict[str, float] | None = None):
    execute = app.arm_controller.execute
    args = (action,) if params is None else (action, params)
    if inspect.iscoroutinefunction(execute):
        result = await execute(*args)
    else:
        result = await asyncio.to_thread(execute, *args)
    if inspect.isawaitable(result):
        return await result
    return result


@mcp.tool(
    annotations={
        "risk": "physical-world",
        "requires": "safety-preflight",
        "production": "use robot_submit_task",
    }
)
async def robot_move(x: float, y: float, z: float, ctx: Context) -> str:
    """Move the robot arm to a target position in millimetres.

    Args:
        x: X coordinate (mm). Positive = right.
        y: Y coordinate (mm). Positive = forward.
        z: Z coordinate (mm). Positive = up.
    """
    if blocked := _require_unsafe():
        return blocked
    app = _get_app(ctx)
    if not app.arm_controller:
        return _no_robot()

    await ctx.info(f"Moving arm to ({x}, {y}, {z}) mm")
    result = await _execute_arm(app, "move", {"x": x, "y": y, "z": z})
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool(
    annotations={
        "risk": "physical-world",
        "requires": "safety-preflight",
        "production": "use robot_submit_task",
    }
)
async def robot_pick(target: str, ctx: Context) -> str:
    """Close the gripper to pick up an object.

    Args:
        target: Description of the object to pick up.
    """
    if blocked := _require_unsafe():
        return blocked
    app = _get_app(ctx)
    if not app.arm_controller:
        return _no_robot()

    await ctx.info(f"Picking up: {target}")
    result = await _execute_arm(app, "grab")
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool(
    annotations={
        "risk": "physical-world",
        "requires": "safety-preflight",
        "production": "use robot_submit_task",
    }
)
async def robot_place(location: str, ctx: Context) -> str:
    """Open the gripper to release / place an object.

    Args:
        location: Description of where to place the object.
    """
    if blocked := _require_unsafe():
        return blocked
    app = _get_app(ctx)
    if not app.arm_controller:
        return _no_robot()

    await ctx.info(f"Placing at: {location}")
    result = await _execute_arm(app, "release")
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool(
    annotations={
        "risk": "physical-world",
        "requires": "safety-preflight",
        "production": "use robot_submit_task",
    }
)
async def robot_home(ctx: Context) -> str:
    """Return the robot arm to its home (rest) position."""
    if blocked := _require_unsafe():
        return blocked
    app = _get_app(ctx)
    if not app.arm_controller:
        return _no_robot()

    result = await _execute_arm(app, "home")
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool(
    annotations={
        "risk": "physical-world",
        "requires": "safety-preflight",
        "production": "use robot_submit_task",
    }
)
async def robot_wave(ctx: Context) -> str:
    """Make the robot arm perform a wave gesture."""
    if blocked := _require_unsafe():
        return blocked
    app = _get_app(ctx)
    if not app.arm_controller:
        return _no_robot()

    result = await _execute_arm(app, "wave")
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool(annotations={"risk": "read-only"})
async def robot_state(ctx: Context) -> str:
    """Get the current robot arm state: joint angles, connection, e-stop."""
    app = _get_app(ctx)
    if not app.arm_controller:
        return _no_robot()

    state = await asyncio.to_thread(app.arm_controller.get_state)
    return json.dumps(state, ensure_ascii=False, default=str)


@mcp.tool(annotations={"risk": "emergency", "bypasses": "all-gates"})
async def robot_estop(ctx: Context) -> str:
    """EMERGENCY STOP — immediately halt all robot motion."""
    app = _get_app(ctx)
    if not app.arm_controller:
        return _no_robot()

    app.arm_controller.emergency_stop()
    logger.warning("E-STOP triggered via MCP tool")
    return json.dumps({
        "status": "emergency_stop_activated",
        "message": "All robot motion halted immediately",
    })


@mcp.tool(annotations={"risk": "physical-world", "requires": "task-handoff+safety-preflight"})
async def robot_submit_task(
    task_type: str,
    params_json: str = "{}",
    operator_id: str = "mcp-agent",
    reason: str = "",
    ctx: Context | None = None,
) -> str:
    """Submit a robot task through the safety chain.

    Production entry-point for ZeroClaw.  All physical actions are routed
    through TaskHandoff → SafetyPreflight → runtime arbiter.

    Args:
        task_type: One of ``move``, ``pick``, ``place``, ``home``, ``wave``,
                   ``navigate``, ``patrol``, ``return_home``.
        params_json: JSON string with task parameters, e.g.
                     ``{"x": 100, "y": 200, "z": 300}`` for move.
        operator_id: Who requested this task (for the audit trail).
        reason: Why this task is needed (for the audit trail).
    """
    app = _get_app(ctx) if ctx else None
    runtime = getattr(app, "runtime_app", None) if app else None

    try:
        params = json.loads(params_json) if isinstance(params_json, str) else params_json
    except json.JSONDecodeError:
        return json.dumps({"error": "invalid_params_json", "message": "params_json is not valid JSON"})

    handoff = None
    if runtime is not None:
        handoff = getattr(runtime, "handoff_service", None)

    if handoff is not None and hasattr(handoff, "submit_plan_payload"):
        plan = {
            "goal": f"{task_type}: {reason}" if reason else task_type,
            "task_type": task_type,
            "params": params,
            "operator_id": operator_id,
            "reason": reason,
            "source": "zeroclaw-mcp",
        }
        try:
            result = await handoff.submit_plan_payload(plan)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as exc:
            logger.warning("TaskHandoff failed for %s: %s", task_type, exc)
            return json.dumps({"error": "handoff_failed", "message": str(exc)})

    # Fallback: no full runtime — only allow in lab mode
    if not _LAB_UNSAFE:
        return json.dumps({
            "error": "runtime_unavailable",
            "message": (
                "Full runtime (TaskHandoff) is not available in standalone MCP mode. "
                "Set ASKME_LAB_UNSAFE_TOOLS=true to use direct arm tools instead."
            ),
        })

    if not app or not app.arm_controller:
        return _no_robot()

    # Lab-unsafe fallback — direct arm call
    action_map = {"move": "move", "pick": "grab", "place": "release", "home": "home", "wave": "wave"}
    action = action_map.get(task_type, task_type)
    result = await _execute_arm(app, action, params if action == "move" else None)
    return json.dumps(result, ensure_ascii=False, default=str)
