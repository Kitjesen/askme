"""Vision tools for askme — exposes VisionBridge capabilities to the agent shell.

Provides two tools:
- look_around: describe the current scene (what objects are visible)
- find_target: search for a specific object class by name
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from .tool_registry import BaseTool

if TYPE_CHECKING:
    from askme.brain.vision_bridge import VisionBridge

logger = logging.getLogger(__name__)


class LookAroundTool(BaseTool):
    """Capture camera frame and describe what the robot currently sees."""

    name = "look_around"
    description = (
        "观察周围环境——拍照并描述当前视野中的物体、人和场景布局。"
        "返回自然语言描述。视觉不可用时返回提示信息。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
    }
    safety_level = "normal"

    def __init__(self) -> None:
        self._vision: VisionBridge | None = None

    def set_vision(self, vision: VisionBridge) -> None:
        self._vision = vision

    def execute(self, **kwargs: Any) -> str:
        if self._vision is None or not self._vision.available:
            return "[视觉不可用] 摄像头未连接或视觉模块未启用。请用其他方式（如导航到可能的位置）继续搜索。"
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    desc = pool.submit(
                        asyncio.run, self._vision.describe_scene()
                    ).result(timeout=10)
            else:
                desc = asyncio.run(self._vision.describe_scene())
        except Exception as exc:
            logger.warning("look_around failed: %s", exc)
            return f"[视觉错误] 拍照失败: {exc}"

        if not desc:
            return "[视觉] 当前视野中未检测到明显物体。"
        return desc


class FindTargetTool(BaseTool):
    """Search for a specific object in the current camera view."""

    name = "find_target"
    description = (
        "在当前视野中搜索指定物体（如 bottle, cup, person）。"
        "找到时返回物体位置信息（bbox, confidence），未找到返回 null。"
        "注意：target 使用英文 YOLO 类别名（如 bottle, cup, chair, person）。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "要搜索的物体英文类别名（YOLO class），如 bottle, cup, person, chair",
            },
        },
        "required": ["target"],
    }
    safety_level = "normal"

    def __init__(self) -> None:
        self._vision: VisionBridge | None = None

    def set_vision(self, vision: VisionBridge) -> None:
        self._vision = vision

    def execute(self, *, target: str = "", **kwargs: Any) -> str:
        if not target:
            return "[错误] 请指定要搜索的物体名称（英文，如 bottle）"
        if self._vision is None or not self._vision.available:
            return "[视觉不可用] 摄像头未连接。无法视觉搜索，请尝试导航到可能的位置。"
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run, self._vision.find_object(target)
                    ).result(timeout=10)
            else:
                result = asyncio.run(self._vision.find_object(target))
        except Exception as exc:
            logger.warning("find_target(%s) failed: %s", target, exc)
            return f"[视觉错误] 搜索失败: {exc}"

        if result is None:
            return f"[未找到] 当前视野中没有检测到 {target}。"

        return json.dumps({
            "found": True,
            "object": result.get("class", target),
            "confidence": round(result.get("confidence", 0), 2),
            "center": result.get("center"),
            "bbox": result.get("bbox"),
        }, ensure_ascii=False)


def register_vision_tools(
    registry: Any,
    vision: VisionBridge,
) -> None:
    """Register vision tools with the given VisionBridge."""
    look = LookAroundTool()
    look.set_vision(vision)
    registry.register(look)

    find = FindTargetTool()
    find.set_vision(vision)
    registry.register(find)
