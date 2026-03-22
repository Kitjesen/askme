"""MCP vision tools — expose look_around, find_target, scan via MCP."""

from __future__ import annotations

import json
from mcp.server.fastmcp import Context

from askme.mcp.server import mcp, AppContext


def _get_app(ctx: Context) -> AppContext:
    return ctx.request_context.lifespan_context


@mcp.tool()
async def look_around(question: str = "", ctx: Context = None) -> str:
    """观察周围环境。可选 question 参数让视觉模型重点关注特定物体。"""
    app = _get_app(ctx)
    if not hasattr(app, "vision") or app.vision is None:
        return json.dumps({"error": "vision not available"})

    import asyncio
    if question:
        result = await app.vision.describe_scene_with_question(question)
    else:
        result = await app.vision.describe_scene()

    return json.dumps({"scene": result or "无检测结果"}, ensure_ascii=False)


@mcp.tool()
async def find_target(target: str, ctx: Context = None) -> str:
    """在当前视野中搜索指定物体（英文 YOLO 类别名）。"""
    app = _get_app(ctx)
    if not hasattr(app, "vision") or app.vision is None:
        return json.dumps({"error": "vision not available"})

    result = await app.vision.find_object(target)
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
    """发送文本消息给机器人并获取回复。"""
    app = _get_app(ctx)
    if not hasattr(app, "text_loop") or app.text_loop is None:
        return json.dumps({"error": "text loop not available"})

    reply = await app.text_loop.process_turn(text)
    return json.dumps({"reply": reply, "text": text}, ensure_ascii=False)
