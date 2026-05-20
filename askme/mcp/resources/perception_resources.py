"""MCP perception resources."""

from __future__ import annotations

import json
import os
import time

from askme.mcp.registration import mcp
from askme.mcp.resource_surface import get_resource_surface


@mcp.resource("askme://perception/detections")
def current_detections() -> str:
    """Return current frame-daemon detections."""
    det_path = "/tmp/askme_frame_detections.json"
    try:
        with open(det_path, encoding="utf-8") as f:
            data = json.load(f)
        age = time.time() - data.get("timestamp", 0)
        data["age_seconds"] = round(age, 1)
        data["fresh"] = age < 3.0
        return json.dumps(data, ensure_ascii=False)
    except (FileNotFoundError, json.JSONDecodeError):
        return json.dumps({"error": "frame_daemon not running", "detections": []})


@mcp.resource("askme://perception/events")
def recent_events() -> str:
    """Return recent perception change events."""
    event_path = "/tmp/askme_events.jsonl"
    events = []
    try:
        with open(event_path, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[-20:]:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    except FileNotFoundError:
        pass
    return json.dumps({"count": len(events), "events": events}, ensure_ascii=False)


@mcp.resource("askme://perception/depth")
def depth_info() -> str:
    """Return depth-daemon status and center depth."""
    return json.dumps(get_resource_surface().depth_info_payload(), ensure_ascii=False)


@mcp.resource("askme://memory/knowledge")
def memory_knowledge() -> str:
    """Return a summary of long-term knowledge files."""
    knowledge_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "data",
        "qp_memory",
        "knowledge",
    )
    files = {}
    try:
        for filename in sorted(os.listdir(knowledge_dir)):
            if filename.endswith(".md"):
                path = os.path.join(knowledge_dir, filename)
                size = os.path.getsize(path)
                files[filename] = {"size_bytes": size, "path": path}
    except FileNotFoundError:
        pass

    return json.dumps({"knowledge_dir": knowledge_dir, "files": files}, ensure_ascii=False)
