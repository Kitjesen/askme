"""MCP memory tools — expose L4/L5 memory search to external agents."""

from __future__ import annotations

import json

from askme.mcp.server import mcp, AppContext


@mcp.tool()
async def memory_search(query: str, n: int = 5, layer: str = "all") -> str:
    """Search robot memory across all layers.

    Args:
        query: Search text (e.g. "温度异常", "巡检记录")
        n: Max results (default 5)
        layer: "all", "knowledge", "digest", or "conversation"
    """
    ctx: AppContext = mcp.get_context()

    results = []

    # L4: RobotMem (conversation history)
    if layer in ("all", "conversation") and ctx.memory_bridge:
        try:
            text = await ctx.memory_bridge.retrieve(query)
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
            from askme.memory.semantic_index import SemanticIndex
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
    if layer in ("all", "knowledge") and ctx.episodic_memory:
        try:
            knowledge = ctx.episodic_memory.get_knowledge_context(max_chars=500)
            if knowledge and query.lower() in knowledge.lower():
                results.append({"text": knowledge[:300], "source": "L3_knowledge"})
        except Exception:
            pass

    if not results:
        return json.dumps({"results": [], "message": "No matching memories found"})

    return json.dumps({"results": results[:n]}, ensure_ascii=False)


@mcp.tool()
async def memory_save(text: str, source: str = "external") -> str:
    """Save a fact to robot long-term memory.

    Args:
        text: The fact to remember (e.g. "仓库A温度传感器已校准")
        source: Origin label (default "external")
    """
    ctx: AppContext = mcp.get_context()
    if not ctx.memory_bridge:
        return json.dumps({"status": "error", "message": "Memory not available"})

    try:
        await ctx.memory_bridge.save(text, f"[{source}] saved")
        return json.dumps({"status": "ok", "message": f"Saved: {text[:50]}"})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
