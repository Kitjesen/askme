"""Robot movement tool — exposes ROS2 cmd_vel / semantic nav to agent shell.

Provides low-level motion primitives that the agent can compose:
- rotate: turn in place (scan surroundings)
- move_forward: walk straight
- go_to: semantic navigation ("去厨房")
- stop: emergency stop
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

from .tool_registry import BaseTool

logger = logging.getLogger(__name__)

# Timeout for ROS2 subprocess calls
_CMD_TIMEOUT = 10.0


def _ros2_pub(topic: str, msg_type: str, data: str, rate: float = 0, count: int = 1) -> str:
    """Publish a ROS2 message via subprocess (system Python with ROS2 sourced)."""
    if rate > 0:
        cmd = (
            f'source /opt/ros/humble/setup.bash && '
            f'timeout {_CMD_TIMEOUT} ros2 topic pub -r {rate} -t {count} '
            f'{topic} {msg_type} "{data}"'
        )
    else:
        cmd = (
            f'source /opt/ros/humble/setup.bash && '
            f'ros2 topic pub --once {topic} {msg_type} "{data}"'
        )
    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True, timeout=_CMD_TIMEOUT + 5,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[:200]
            return f"[错误] ROS2 发布失败: {stderr}"
        return "OK"
    except subprocess.TimeoutExpired:
        return "[错误] ROS2 命令超时"
    except Exception as exc:
        return f"[错误] {exc}"


class MoveRobotTool(BaseTool):
    """Control robot movement — rotate, walk forward, go to location, or stop."""

    name = "move_robot"
    description = (
        "控制机器人运动。支持以下动作：\n"
        "- action='rotate', angle=45 → 原地旋转（正=左转，负=右转，单位度）\n"
        "- action='forward', distance=1.0 → 前进（单位米，负=后退）\n"
        "- action='go_to', target='厨房' → 语义导航到指定位置\n"
        "- action='stop' → 立即停止\n"
        "搜索物体时，先 rotate 扫描四周，再 go_to 导航到可能位置。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["rotate", "forward", "go_to", "stop"],
                "description": "动作类型",
            },
            "angle": {
                "type": "number",
                "description": "旋转角度（度），正=左转，负=右转。仅 action=rotate 时有效",
            },
            "distance": {
                "type": "number",
                "description": "前进距离（米），负=后退。仅 action=forward 时有效",
            },
            "target": {
                "type": "string",
                "description": "目标位置名称（如'厨房'、'仓库'）。仅 action=go_to 时有效",
            },
        },
        "required": ["action"],
    }
    safety_level = "normal"

    def execute(
        self,
        *,
        action: str = "",
        angle: float = 0,
        distance: float = 0,
        target: str = "",
        **kwargs: Any,
    ) -> str:
        if action == "rotate":
            return self._rotate(angle)
        elif action == "forward":
            return self._forward(distance)
        elif action == "go_to":
            return self._go_to(target)
        elif action == "stop":
            return self._stop()
        else:
            return f"[错误] 未知动作: {action}"

    def _rotate(self, angle_deg: float) -> str:
        """Rotate in place by publishing cmd_vel with angular.z."""
        if abs(angle_deg) < 1:
            return "[跳过] 旋转角度太小"

        import math
        # angular velocity ~0.5 rad/s, calculate duration
        angular_speed = 0.5  # rad/s
        angle_rad = math.radians(angle_deg)
        duration = abs(angle_rad) / angular_speed
        # Publish at 10Hz for the calculated duration
        count = max(1, int(duration * 10))
        sign = 1.0 if angle_deg > 0 else -1.0

        data = (
            f"{{header: {{stamp: {{sec: 0, nanosec: 0}}, frame_id: base_link}}, "
            f"twist: {{linear: {{x: 0.0, y: 0.0, z: 0.0}}, "
            f"angular: {{x: 0.0, y: 0.0, z: {sign * angular_speed}}}}}}}"
        )
        result = _ros2_pub("/nav/cmd_vel", "geometry_msgs/msg/TwistStamped", data, rate=10, count=count)
        if result == "OK":
            # Send stop after rotation
            self._stop()
            return f"已旋转约 {abs(angle_deg):.0f} 度{'左转' if angle_deg > 0 else '右转'}"
        return result

    def _forward(self, distance: float) -> str:
        """Move forward/backward by publishing cmd_vel with linear.x."""
        if abs(distance) < 0.05:
            return "[跳过] 距离太短"

        speed = 0.3  # m/s
        duration = abs(distance) / speed
        count = max(1, int(duration * 10))
        sign = 1.0 if distance > 0 else -1.0

        data = (
            f"{{header: {{stamp: {{sec: 0, nanosec: 0}}, frame_id: base_link}}, "
            f"twist: {{linear: {{x: {sign * speed}, y: 0.0, z: 0.0}}, "
            f"angular: {{x: 0.0, y: 0.0, z: 0.0}}}}}}"
        )
        result = _ros2_pub("/nav/cmd_vel", "geometry_msgs/msg/TwistStamped", data, rate=10, count=count)
        if result == "OK":
            self._stop()
            direction = "前进" if distance > 0 else "后退"
            return f"已{direction}约 {abs(distance):.1f} 米"
        return result

    def _go_to(self, target: str) -> str:
        """Semantic navigation via /nav/semantic/instruction."""
        if not target:
            return "[错误] 请指定目标位置"

        data = f"{{data: '去{target}'}}"
        result = _ros2_pub("/nav/semantic/instruction", "std_msgs/msg/String", data)
        if result == "OK":
            return f"已发送导航指令: 去{target}（导航进行中）"
        return result

    def _stop(self) -> str:
        """Send zero velocity to stop."""
        data = (
            "{header: {stamp: {sec: 0, nanosec: 0}, frame_id: base_link}, "
            "twist: {linear: {x: 0.0, y: 0.0, z: 0.0}, "
            "angular: {x: 0.0, y: 0.0, z: 0.0}}}"
        )
        _ros2_pub("/nav/cmd_vel", "geometry_msgs/msg/TwistStamped", data)
        return "已停止"


def register_move_tools(registry: Any) -> None:
    """Register movement tools."""
    registry.register(MoveRobotTool())
