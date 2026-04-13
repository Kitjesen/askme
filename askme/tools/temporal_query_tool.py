"""TemporalQueryTool — LLM tool for querying LingTu's time-indexed scene memory.

Lets the LLM answer questions like:
  - "充电桩附近最近有什么东西？"
  - "过去30分钟在厨房看到人了吗？"
  - "机器人上次看到椅子是什么时候？"

Calls GET /api/v1/memory/temporal on the LingTu gateway (NAV_GATEWAY_URL).
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .tool_registry import BaseTool

logger = logging.getLogger(__name__)


class TemporalQueryTool(BaseTool):
    """Query LingTu's time-indexed entity observation memory."""

    name = "temporal_query"
    description = (
        "查询机器人的时空记忆——某时间窗口内在某位置附近观测到了哪些实体（人/物/地标等）。\n"
        "示例用法：\n"
        "- label='person', since='30m'：过去30分钟内看到的人\n"
        "- label='charging_station', since='2h'：2小时内看到的充电桩\n"
        "- since='1h', near_x=3.4, near_y=1.8, radius=2.0：1小时内(3.4,1.8)附近2米范围的所有实体\n"
        "- label='' (空) 查询所有类别"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": "实体类别，如 person、chair、door、charging_station。留空查询所有类别",
            },
            "since": {
                "type": "string",
                "description": "时间窗口，如 '30m'(30分钟)、'1h'(1小时)、'2h'、'1d'(1天)。默认 '1h'",
                "default": "1h",
            },
            "near_x": {
                "type": "number",
                "description": "查询中心的世界坐标 X（米）。与 near_y 和 radius 同时使用",
            },
            "near_y": {
                "type": "number",
                "description": "查询中心的世界坐标 Y（米）",
            },
            "radius": {
                "type": "number",
                "description": "空间查询半径（米）。仅在 near_x/near_y 同时提供时生效",
            },
            "limit": {
                "type": "integer",
                "description": "最多返回的记录条数，默认 20，最大 100",
                "default": 20,
            },
        },
    }
    agent_allowed = True
    voice_label = "时空记忆查询"

    def execute(
        self,
        *,
        label: str = "",
        since: str = "1h",
        near_x: float | None = None,
        near_y: float | None = None,
        radius: float | None = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> str:
        nav_url = os.environ.get("NAV_GATEWAY_URL", "")
        if not nav_url:
            return "[时空记忆] NAV_GATEWAY_URL 未设置，无法查询 LingTu 记忆服务。"

        params: dict[str, Any] = {
            "since": since or "1h",
            "limit": max(1, min(int(limit), 100)),
        }
        if label:
            params["label"] = label
        if near_x is not None:
            params["near_x"] = round(float(near_x), 3)
        if near_y is not None:
            params["near_y"] = round(float(near_y), 3)
        if radius is not None:
            params["radius"] = round(float(radius), 3)

        url = f"{nav_url.rstrip('/')}/api/v1/memory/temporal?{urllib.parse.urlencode(params)}"
        try:
            raw = urllib.request.urlopen(url, timeout=5).read()
            data: dict[str, Any] = json.loads(raw.decode())
        except urllib.error.URLError as exc:
            return f"[时空记忆] LingTu 不可达 ({nav_url}): {exc.reason}"
        except Exception as exc:
            return f"[时空记忆] 查询失败: {exc}"

        observations: list[dict] = data.get("observations", [])
        if not observations:
            lbl_desc = f"「{label}」" if label else "任何实体"
            return f"在 {since} 内未发现{lbl_desc}的观测记录。"

        now = time.time()
        lines = [f"时空记忆（共 {len(observations)} 条，{since} 内）："]
        for obs in observations:
            age = now - obs.get("ts", now)
            if age < 60:
                age_str = f"{age:.0f}秒前"
            elif age < 3600:
                age_str = f"{age / 60:.0f}分钟前"
            else:
                age_str = f"{age / 3600:.1f}小时前"
            px = obs.get("pos_x") or 0.0
            py = obs.get("pos_y") or 0.0
            conf = obs.get("confidence", 1.0)
            lines.append(
                f"  [{age_str}] {obs['label']}  位置({px:.1f}, {py:.1f})  置信度{conf:.2f}"
            )
        return "\n".join(lines)


def register_temporal_tools(registry: Any) -> None:
    """Register temporal memory tools with the tool registry."""
    registry.register(TemporalQueryTool())
