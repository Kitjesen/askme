"""MCP tools for robot arm control."""

from __future__ import annotations

import asyncio
import json
import logging

from mcp.server.fastmcp import Context

from askme.errors import ROBOT_NOT_CONNECTED, INTERNAL_ERROR, error_response
from askme.mcp_server import AppContext, mcp

logger = logging.getLogger(__name__)


def _get_app(ctx: Context) -> AppContext:
    return ctx.request_context.lifespan_context


def _no_robot() -> str:
    return error_response(ROBOT_NOT_CONNECTED, "Robot arm not connected or not enabled")


@mcp.tool()
async def robot_move(x: float, y: float, z: float, ctx: Context) -> str:
    """Move the robot arm to a target position in millimetres.

    Args:
        x: X coordinate (mm). Positive = right.
        y: Y coordinate (mm). Positive = forward.
        z: Z coordinate (mm). Positive = up.
    """
    app = _get_app(ctx)
    if not app.arm_controller:
        return _no_robot()

    await ctx.info(f"Moving arm to ({x}, {y}, {z}) mm")
    result = await asyncio.to_thread(
        app.arm_controller.execute, "move", {"x": x, "y": y, "z": z}
    )
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool()
async def robot_pick(target: str, ctx: Context) -> str:
    """Close the gripper to pick up an object.

    Args:
        target: Description of the object to pick up.
    """
    app = _get_app(ctx)
    if not app.arm_controller:
        return _no_robot()

    await ctx.info(f"Picking up: {target}")
    result = await asyncio.to_thread(app.arm_controller.execute, "grab")
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool()
async def robot_place(location: str, ctx: Context) -> str:
    """Open the gripper to release / place an object.

    Args:
        location: Description of where to place the object.
    """
    app = _get_app(ctx)
    if not app.arm_controller:
        return _no_robot()

    await ctx.info(f"Placing at: {location}")
    result = await asyncio.to_thread(app.arm_controller.execute, "release")
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool()
async def robot_home(ctx: Context) -> str:
    """Return the robot arm to its home (rest) position."""
    app = _get_app(ctx)
    if not app.arm_controller:
        return _no_robot()

    result = await asyncio.to_thread(app.arm_controller.execute, "home")
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool()
async def robot_wave(ctx: Context) -> str:
    """Make the robot arm perform a wave gesture."""
    app = _get_app(ctx)
    if not app.arm_controller:
        return _no_robot()

    result = await asyncio.to_thread(app.arm_controller.execute, "wave")
    return json.dumps(result, ensure_ascii=False, default=str)


@mcp.tool()
async def robot_state(ctx: Context) -> str:
    """Get the current robot arm state: joint angles, connection, e-stop."""
    app = _get_app(ctx)
    if not app.arm_controller:
        return _no_robot()

    state = await asyncio.to_thread(app.arm_controller.get_state)
    return json.dumps(state, ensure_ascii=False, default=str)


@mcp.tool()
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
