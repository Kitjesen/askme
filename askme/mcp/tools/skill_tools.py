"""MCP tool for executing SKILL.md skills."""

from __future__ import annotations

import datetime
import json
import logging
import uuid

from mcp.server.fastmcp import Context

from askme.conversation import GenerationStarted, InteractionInput, TurnOutcome
from askme.errors import INTERNAL_ERROR, SKILL_DISABLED, SKILL_NOT_FOUND, error_response
from askme.llm.core.contracts import LLMCallContext
from askme.mcp.context import AppContext
from askme.mcp.registration import mcp

logger = logging.getLogger(__name__)


def _get_app(ctx: Context) -> AppContext:
    return ctx.request_context.lifespan_context


@mcp.tool()
async def execute_skill(skill_name: str, user_input: str, ctx: Context) -> str:
    """Execute a named skill from the SKILL.md system.

    Args:
        skill_name: Name of the skill (e.g. "daily_summary", "robot_move").
        user_input: User context or instruction to pass to the skill prompt.
    """
    app = _get_app(ctx)
    if app.skill_manager is None:
        return error_response(INTERNAL_ERROR, "Skill manager not initialised")

    skill = app.skill_manager.get(skill_name)
    if skill is None:
        catalog = app.skill_manager.get_skill_catalog()
        return error_response(
            SKILL_NOT_FOUND, f"Skill '{skill_name}' not found", {"available": catalog}
        )

    if not skill.enabled:
        return error_response(SKILL_DISABLED, f"Skill '{skill_name}' is disabled")

    await ctx.info(f"Executing skill: {skill_name}")

    template_ctx: dict[str, str] = {
        "user_input": user_input,
        "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_date": datetime.datetime.now().strftime("%Y-%m-%d"),
    }
    if app.arm_controller:
        template_ctx["robot_state"] = json.dumps(app.arm_controller.get_state(), ensure_ascii=False)

    call_id = uuid.uuid4().hex
    turn_manager = app.interaction_turn_manager
    turn_context = None
    if turn_manager is not None:
        turn_context = turn_manager.open(
            InteractionInput(
                user_text=user_input,
                source="mcp",
                thread_id=app.process_session_id,
                channel="text",
                metadata={
                    "skill_name": skill_name,
                    "llm_call_id": call_id,
                },
            )
        )
        turn_context = turn_manager.advance(
            turn_context,
            GenerationStarted(
                provider="skill_executor",
                generation_id=call_id,
                metadata={"skill_name": skill_name},
            ),
        )

    try:
        result = await app.skill_executor.execute(
            skill,
            template_ctx,
            llm_call_context=LLMCallContext(
                call_id=call_id,
                session_id=app.process_session_id,
                turn_id=getattr(turn_context, "turn_id", None),
                purpose="tool_followup",
                channel="text",
                request_class="robot_action",
                privacy_class="conversation",
                allow_cache=False,
            ),
        )
    except Exception as exc:
        if turn_manager is not None and turn_context is not None:
            turn_manager.settle(
                turn_context,
                TurnOutcome.fail(
                    reason="mcp_skill_failed",
                    metadata={
                        "skill_name": skill_name,
                        "error_type": type(exc).__name__,
                    },
                ),
            )
        raise
    if turn_manager is not None and turn_context is not None:
        turn_manager.settle(
            turn_context,
            TurnOutcome.commit(
                assistant_text=str(result or ""),
                metadata={"skill_name": skill_name},
            ),
        )
    return result
