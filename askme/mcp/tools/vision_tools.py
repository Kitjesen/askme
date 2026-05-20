"""MCP vision and text tools."""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import Context

from askme.mcp.context import AppContext
from askme.mcp.registration import mcp


def _get_app(ctx: Context) -> AppContext:
    return ctx.request_context.lifespan_context


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

    return json.dumps({
        "found": True,
        "object": result.get("class_id", target),
        "confidence": round(result.get("confidence", 0), 2),
        "bbox": result.get("bbox"),
        "distance_m": result.get("distance_m"),
    }, ensure_ascii=False)


@mcp.tool()
async def chat(text: str, ctx: Context = None) -> str:
    """Send a text turn through the stable MCP context LLM surface."""
    app = _get_app(ctx)
    if app.llm_client is None:
        return json.dumps({"error": "llm client not available"})

    messages: list[dict[str, Any]]
    if app.conversation is not None:
        app.conversation.add_user_message(text)
        messages = app.conversation.get_messages(
            system_prompt="You are Askme, a helpful robot assistant."
        )
    else:
        messages = [{"role": "user", "content": text}]

    reply = await app.llm_client.chat(messages)
    if app.conversation is not None:
        app.conversation.add_assistant_message(reply)
    return json.dumps({"reply": reply, "text": text}, ensure_ascii=False)
