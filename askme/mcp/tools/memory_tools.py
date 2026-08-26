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
async def memory_save(
    text: str,
    source: str = "external",
    customer_id: str = "",
    project_id: str = "",
    user_id: str = "",
    source_turn_id: str = "",
    confidence: float = 1.0,
    ctx: Context = None,
) -> str:
    """Submit a fact to governed durable-memory admission.

    Args:
        text: The fact to remember (e.g. "仓库A温度传感器已校准")
        source: Origin label (default "external")
        customer_id: Request-scoped customer identity for customer knowledge
        project_id: Request-scoped project identity for customer knowledge
        user_id: Request-scoped user identity for preferences/profile facts
        source_turn_id: Optional idempotency/audit identifier
        confidence: Source/ASR confidence used by local admission policy
    """
    app = _get_app(ctx)
    if not app.memory_bridge:
        return json.dumps({"status": "error", "message": "Memory not available"})

    try:
        result = await app.memory_bridge.admit_turn(
            text,
            source=source,
            source_turn_id=source_turn_id,
            confidence=confidence,
            customer_id=customer_id,
            project_id=project_id,
            user_id=user_id,
        )
        if not bool(getattr(result, "admitted", False)):
            reason = str(getattr(result, "rejected_reason", "") or "not_admitted")
            return json.dumps(
                {
                    "status": "rejected",
                    "message": "Memory admission rejected the content",
                    "reason": reason,
                },
                ensure_ascii=False,
            )
        persisted = int(getattr(result, "persisted_count", 0) or 0)
        errors = list(getattr(result, "persistence_errors", ()) or ())
        if persisted <= 0:
            return json.dumps(
                {
                    "status": "error",
                    "message": "Memory was admitted but not persisted",
                    "errors": errors,
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "status": "ok",
                "message": f"Saved: {text[:50]}",
                "persisted_count": persisted,
                "errors": errors,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
