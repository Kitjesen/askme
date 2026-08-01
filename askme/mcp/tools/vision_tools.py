"""MCP vision and text tools."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from mcp.server.fastmcp import Context

from askme.conversation import GenerationStarted, InteractionInput, TurnOutcome
from askme.llm.core.contracts import LLMCallContext
from askme.mcp.context import AppContext
from askme.mcp.registration import mcp

logger = logging.getLogger(__name__)


def _get_app(ctx: Context) -> AppContext:
    return ctx.request_context.lifespan_context


def _resolve_conversation_session_id(
    app: AppContext,
    ctx: Context,
    explicit_session_id: str | None,
) -> str:
    """Resolve one stable logical conversation identity for an MCP request."""

    candidates = (
        explicit_session_id,
        getattr(ctx, "client_id", None),
        app.process_session_id,
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    app.process_session_id = f"mcp-{uuid.uuid4().hex}"
    return app.process_session_id


def _no_vision() -> str:
    return json.dumps({"error": "vision not available"})


@mcp.tool()
async def look_around(question: str = "", ctx: Context = None) -> str:
    """Describe the current scene, optionally focused by a question."""
    app = _get_app(ctx)
    vision = app.vision_bridge
    if vision is None:
        return _no_vision()

    if question:
        result = await vision.describe_scene_with_question(question)
    else:
        result = await vision.describe_scene()

    return json.dumps({"scene": result or "无检测结果"}, ensure_ascii=False)


@mcp.tool()
async def find_target(target: str, ctx: Context = None) -> str:
    """Search for a named object in the current field of view."""
    app = _get_app(ctx)
    vision = app.vision_bridge
    if vision is None:
        return _no_vision()

    result = await vision.find_object(target)
    if result is None:
        return json.dumps({"found": False, "target": target}, ensure_ascii=False)

    return json.dumps(
        {
            "found": True,
            "object": result.get("class_id", target),
            "confidence": round(result.get("confidence", 0), 2),
            "bbox": result.get("bbox"),
            "distance_m": result.get("distance_m"),
        },
        ensure_ascii=False,
    )


@mcp.tool()
async def chat(
    text: str,
    ctx: Context = None,
    conversation_session_id: str | None = None,
) -> str:
    """Send a text turn through the stable MCP context LLM surface."""
    app = _get_app(ctx)
    resolved_session_id = _resolve_conversation_session_id(app, ctx, conversation_session_id)
    if app.llm_client is None:
        return json.dumps({"error": "llm client not available"})

    call_id = uuid.uuid4().hex
    turn_manager = app.interaction_turn_manager
    turn_context = None
    try:
        if turn_manager is not None:
            turn_context = turn_manager.open(
                InteractionInput(
                    user_text=text,
                    source="mcp",
                    thread_id=resolved_session_id,
                    channel="text",
                    metadata={"llm_call_id": call_id},
                )
            )
            turn_context = turn_manager.advance(
                turn_context,
                GenerationStarted(provider="llm", generation_id=call_id),
            )

        messages: list[dict[str, Any]]
        if app.conversation is not None:
            messages = app.conversation.get_messages(
                system_prompt="You are Askme, a helpful robot assistant.",
                conversation_session_id=resolved_session_id,
            )
            messages = [*messages, {"role": "user", "content": text}]
        else:
            messages = [{"role": "user", "content": text}]

        reply = await app.llm_client.chat(
            messages,
            context=LLMCallContext(
                call_id=call_id,
                session_id=resolved_session_id,
                turn_id=getattr(turn_context, "turn_id", None),
                purpose="assistant_response",
                channel="text",
                request_class="text",
                privacy_class="conversation",
                allow_cache=False,
            ),
        )
    except Exception as exc:
        if turn_manager is not None and turn_context is not None:
            turn_manager.settle(
                turn_context,
                TurnOutcome.fail(
                    reason="mcp_chat_failed",
                    metadata={"error_type": type(exc).__name__},
                ),
            )
        raise

    if turn_manager is not None and turn_context is not None:
        turn_manager.settle(turn_context, TurnOutcome.commit(assistant_text=reply))
    if app.conversation is not None:
        try:
            app.conversation.add_user_message(
                text,
                conversation_session_id=resolved_session_id,
            )
            app.conversation.add_assistant_message(
                reply,
                conversation_session_id=resolved_session_id,
            )
        except Exception:
            logger.warning(
                "MCP chat legacy conversation projection failed",
                exc_info=True,
            )
    return json.dumps({"reply": reply, "text": text}, ensure_ascii=False)
