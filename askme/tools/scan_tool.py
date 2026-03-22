"""Fast scene scan tool — captures current view + BPU detections + optional rotation.

Uses daemon pre-computed frames and detections (zero latency) for the current
direction. For 360° scan, requests rotation through dog-control-service API
(safe path), falling back to single-direction scan if control service unavailable.

Does NOT directly publish to ROS2 cmd_vel topics.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .tool_registry import BaseTool

logger = logging.getLogger(__name__)


class ScanAroundTool(BaseTool):
    """Scan current view or request 360° rotation scan via control service."""

    name = "scan_around"
    description = (
        "扫描当前方向环境。读取 BPU 检测结果 + 可选 VLM 描述（即时，<1s）。\n"
        "注意：只能看当前摄像头朝向，不能旋转（旋转需要 dog-control-service）。\n"
        "可选 question 参数让 VLM 针对当前视野回答问题。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "可选：要重点观察的问题",
            },
            "full": {
                "type": "boolean",
                "description": "是否请求 360° 旋转扫描（需要控制服务支持）",
            },
        },
    }
    safety_level = "normal"
    agent_allowed = True
    voice_label = "全方位扫描"

    def __init__(self) -> None:
        self._vision: Any = None

    def set_vision(self, vision: Any) -> None:
        self._vision = vision

    def execute(self, *, question: str = "", full: bool = False, **kwargs: Any) -> str:
        t0 = time.monotonic()

        # Always start with current direction (instant from daemon)
        lines = []

        # 1. BPU detections from daemon (0ms)
        dets = []
        if self._vision:
            dets = self._vision._read_daemon_detections() or []

        if dets:
            det_desc = ", ".join(
                f"{d['class_id']}({d.get('distance_m', '?')}m)" if d.get('distance_m')
                else d['class_id']
                for d in dets
            )
            lines.append(f"YOLO检测: {det_desc}")
        else:
            lines.append("YOLO检测: 当前方向无 COCO 物体")

        # 2. VLM question or general description
        if question and self._vision:
            import asyncio
            try:
                vlm_answer = asyncio.run(
                    self._vision.describe_scene_with_question(question)
                )
            except RuntimeError:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(1) as pool:
                    vlm_answer = pool.submit(
                        asyncio.run,
                        self._vision.describe_scene_with_question(question)
                    ).result(timeout=15)
            except Exception as exc:
                vlm_answer = f"(VLM 错误: {exc})"
            if vlm_answer:
                lines.append(f"VLM回答: {vlm_answer}")

        # 3. Full 360° rotation scan (if requested)
        if full:
            lines.append(self._request_rotation_scan())

        elapsed = (time.monotonic() - t0) * 1000
        lines.insert(0, f"扫描完成 ({elapsed:.0f}ms)")
        return "\n".join(lines)

    def _request_rotation_scan(self) -> str:
        """Request 360° scan through control service. Returns status message."""
        from .move_tool import _call_runtime_api

        # Request rotate 360° — control service handles the actual motion safely
        result = _call_runtime_api("control", "POST", "/api/v1/control/executions", {
            "mission_id": f"scan_{int(time.time())}",
            "mission_type": "motion_command",
            "requested_capability": "scan_360",
            "parameters": {"speed_deg_per_sec": 45},
        })

        if "error" in result:
            return (
                "[360°扫描不可用] dog-control-service 未运行或不支持 scan_360。"
                "已返回当前方向的扫描结果。"
            )
        return "360°旋转扫描已请求"


def register_scan_tools(registry: Any, vision: Any = None) -> None:
    """Register scan tools."""
    tool = ScanAroundTool()
    if vision:
        tool.set_vision(vision)
    registry.register(tool)
