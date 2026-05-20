"""MCP memory tools — expose L4/L5 memory search to external agents."""

from __future__ import annotations

import json

from mcp.server.fastmcp import Context

from askme.mcp.context import AppContext
from askme.mcp.registration import mcp


def _get_app(ctx: Context) -> AppContext:
    return ctx.request_context.lifespan_context


@mcp.tool()
async def memory_search(
    query: str,
    n: int = 5,
    layer: str = "all",
    ctx: Context = None,
) -> str:
    """Search robot memory across all layers.

    Args:
        query: Search text (e.g. "温度异常", "巡检记录")
        n: Max results (default 5)
        layer: "all", "knowledge", "digest", or "conversation"
    """
    app = _get_app(ctx)

    results = []

    # L4: RobotMem (conversation history)
    if layer in ("all", "conversation") and app.memory_bridge:
        try:
            text = await app.memory_bridge.retrieve(query)
            if text:
                for line in text.split("\n"):
                    line = line.strip().lstrip("- ")
                    if line:
                        results.append({"text": line, "source": "L4_conversation"})
        except Exception:
            pass

    # L5: Semantic Index (knowledge + digests)
    if layer in ("all", "knowledge", "digest"):
        try:
            from askme.memory.retrieval.semantic_index import SemanticIndex
            idx = SemanticIndex()
            sem_results = await idx.search(
                query, n=n,
                source_filter=layer if layer != "all" else None,
            )
            for r in sem_results:
                results.append({
                    "text": r["text"],
                    "source": f"L5_{r['source']}",
                    "category": r.get("category", ""),
                })
            idx.close()
        except Exception:
            pass

    # L3: Episodic knowledge (file-based)
    if layer in ("all", "knowledge") and app.episodic_memory:
        try:
            knowledge = app.episodic_memory.get_knowledge_context(max_chars=500)
            if knowledge and query.lower() in knowledge.lower():
                results.append({"text": knowledge[:300], "source": "L3_knowledge"})
        except Exception:
            pass

    if not results:
        return json.dumps({"results": [], "message": "No matching memories found"})

    return json.dumps({"results": results[:n]}, ensure_ascii=False)


@mcp.tool()
async def memory_save(text: str, source: str = "external", ctx: Context = None) -> str:
    """Save a fact to robot long-term memory.

    Args:
        text: The fact to remember (e.g. "仓库A温度传感器已校准")
        source: Origin label (default "external")
    """
    app = _get_app(ctx)
    if not app.memory_bridge:
        return json.dumps({"status": "error", "message": "Memory not available"})

    try:
        await app.memory_bridge.save(text, f"[{source}] saved")
        return json.dumps({"status": "ok", "message": f"Saved: {text[:50]}"})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
