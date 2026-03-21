"""MCP perception resources — expose vision, detection, and memory state."""

from __future__ import annotations

import json
import os
import time

from askme.mcp_server import mcp


@mcp.resource("askme://perception/detections")
def current_detections() -> str:
    """当前 BPU YOLO 检测结果（实时，来自 frame_daemon）。"""
    det_path = "/tmp/askme_frame_detections.json"
    try:
        with open(det_path, "r") as f:
            data = json.load(f)
        age = time.time() - data.get("timestamp", 0)
        data["age_seconds"] = round(age, 1)
        data["fresh"] = age < 3.0
        return json.dumps(data, ensure_ascii=False)
    except (FileNotFoundError, json.JSONDecodeError):
        return json.dumps({"error": "frame_daemon not running", "detections": []})


@mcp.resource("askme://perception/events")
def recent_events() -> str:
    """最近的 ChangeDetector 感知事件（最新 20 条）。"""
    event_path = "/tmp/askme_events.jsonl"
    events = []
    try:
        with open(event_path, "r", encoding="utf-8") as f:
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
    """深度相机概要信息（中心点距离、daemon 状态）。"""
    import struct
    result = {"daemon_alive": False, "center_depth_m": None}

    # Check heartbeat
    try:
        with open("/tmp/askme_frame_daemon.heartbeat", "r") as f:
            ts = float(f.read().strip())
        result["daemon_alive"] = time.time() - ts < 3.0
    except (FileNotFoundError, ValueError):
        pass

    # Read center depth
    try:
        import numpy as np
        with open("/tmp/askme_frame_depth.bin", "rb") as f:
            header = f.read(8)
            if len(header) == 8:
                w, h = struct.unpack("II", header)
                data = f.read(w * h * 2)
                if len(data) == w * h * 2:
                    depth = np.frombuffer(data, dtype=np.uint16).reshape(h, w)
                    center = depth[h // 2, w // 2]
                    result["center_depth_m"] = round(center / 1000.0, 2) if center > 0 else None
                    result["frame_size"] = f"{w}x{h}"
    except Exception:
        pass

    return json.dumps(result, ensure_ascii=False)


@mcp.resource("askme://memory/knowledge")
def memory_knowledge() -> str:
    """qp_memory 长期知识概要（设备、事件、流程文件列表）。"""
    knowledge_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                 "data", "qp_memory", "knowledge")
    files = {}
    try:
        for f in sorted(os.listdir(knowledge_dir)):
            if f.endswith(".md"):
                path = os.path.join(knowledge_dir, f)
                size = os.path.getsize(path)
                files[f] = {"size_bytes": size, "path": path}
    except FileNotFoundError:
        pass

    return json.dumps({"knowledge_dir": knowledge_dir, "files": files}, ensure_ascii=False)
