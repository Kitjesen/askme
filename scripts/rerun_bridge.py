#!/usr/bin/env python3
"""Rerun visualization bridge for Thunder.

Subscribes to ROS2 topics and streams to the Rerun viewer so you can
visualize detections, IMU orientation, and joint states from a laptop.

Topics consumed:
  /thunder/detections   — sensor_msgs/Image + custom detection overlay
  /thunder/imu          — sensor_msgs/Imu
  /thunder/joint_states — sensor_msgs/JointState
  /camera/image_raw     — sensor_msgs/Image  (optional, for detection overlay)

Usage (on S100P):
    source /opt/ros/humble/setup.bash
    pip install rerun-sdk
    python3 scripts/rerun_bridge.py [--addr 0.0.0.0:9876]

Then on laptop:
    pip install rerun-sdk
    rerun --connect 192.168.66.190:9876
"""
from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
import threading
from typing import Optional

logger = logging.getLogger("rerun_bridge")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Lazy imports — fail with a clear message if dependencies are missing
# ---------------------------------------------------------------------------
try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError:
    logger.error("rerun-sdk not installed. Run: pip install rerun-sdk")
    sys.exit(1)

try:
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from sensor_msgs.msg import Image, Imu, JointState
    from std_msgs.msg import String
except ImportError:
    logger.error("rclpy not available. Source /opt/ros/humble/setup.bash first.")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    logger.error("numpy not installed.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# QoS
# ---------------------------------------------------------------------------
_QOS_SENSOR = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
)

_QOS_RELIABLE = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)

# ---------------------------------------------------------------------------
# Thunder joint names (16 DOF — matches brainstem order)
# ---------------------------------------------------------------------------
_JOINT_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "FL_ankle", "FR_ankle", "RL_ankle", "RR_ankle",
]


# ---------------------------------------------------------------------------
# Quaternion → Euler (for IMU display)
# ---------------------------------------------------------------------------
def _quat_to_euler(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """Return (roll, pitch, yaw) in degrees from a unit quaternion."""
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.degrees(math.atan2(sinr, cosr))

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.degrees(math.asin(sinp))

    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.degrees(math.atan2(siny, cosy))

    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# ROS2 bridge node
# ---------------------------------------------------------------------------
class RerunBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("rerun_bridge")

        # /thunder/imu
        self.create_subscription(
            Imu,
            "/thunder/imu",
            self._on_imu,
            _QOS_SENSOR,
        )

        # /thunder/joint_states
        self.create_subscription(
            JointState,
            "/thunder/joint_states",
            self._on_joint_states,
            _QOS_SENSOR,
        )

        # /camera/image_raw (best-effort; only available when camera is running)
        self.create_subscription(
            Image,
            "/camera/image_raw",
            self._on_camera_image,
            _QOS_SENSOR,
        )

        # /thunder/detections — JSON-encoded detection list published by askme
        # Format: [{"label": str, "x1": int, "y1": int, "x2": int, "y2": int, "conf": float}, ...]
        self.create_subscription(
            String,
            "/thunder/detections",
            self._on_detections,
            _QOS_RELIABLE,
        )

        # Latest camera image for overlaying detections
        self._latest_image: Optional[np.ndarray] = None
        self._image_lock = threading.Lock()

        logger.info("RerunBridgeNode ready, subscribed to /thunder/{imu,joint_states,detections} and /camera/image_raw")

    # ------------------------------------------------------------------
    def _ros_time_ns(self, header) -> int:
        return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec

    # ------------------------------------------------------------------
    def _on_imu(self, msg: Imu) -> None:
        t = self._ros_time_ns(msg.header)
        rr.set_time_nanos("ros_time", t)

        q = msg.orientation
        roll, pitch, yaw = _quat_to_euler(q.x, q.y, q.z, q.w)

        # Log orientation as scalar time-series (easy to read on a graph)
        rr.log("thunder/imu/roll_deg",  rr.Scalars(roll))
        rr.log("thunder/imu/pitch_deg", rr.Scalars(pitch))
        rr.log("thunder/imu/yaw_deg",   rr.Scalars(yaw))

        # Angular velocity
        av = msg.angular_velocity
        rr.log("thunder/imu/angular_velocity/x", rr.Scalars(av.x))
        rr.log("thunder/imu/angular_velocity/y", rr.Scalars(av.y))
        rr.log("thunder/imu/angular_velocity/z", rr.Scalars(av.z))

        # Linear acceleration
        la = msg.linear_acceleration
        rr.log("thunder/imu/linear_acceleration/x", rr.Scalars(la.x))
        rr.log("thunder/imu/linear_acceleration/y", rr.Scalars(la.y))
        rr.log("thunder/imu/linear_acceleration/z", rr.Scalars(la.z))

        # 3-D transform for the robot body (visual orientation indicator)
        rr.log(
            "thunder/body_transform",
            rr.Transform3D(
                rotation=rr.Quaternion(xyzw=[q.x, q.y, q.z, q.w]),
            ),
        )

    # ------------------------------------------------------------------
    def _on_joint_states(self, msg: JointState) -> None:
        t = self._ros_time_ns(msg.header)
        rr.set_time_nanos("ros_time", t)

        names = list(msg.name) if msg.name else _JOINT_NAMES[: len(msg.position)]
        positions = list(msg.position)
        velocities = list(msg.velocity) if msg.velocity else []
        efforts = list(msg.effort) if msg.effort else []

        for i, name in enumerate(names):
            safe_name = name.replace("/", "_")
            if i < len(positions):
                rr.log(f"thunder/joints/{safe_name}/position", rr.Scalars(positions[i]))
            if i < len(velocities):
                rr.log(f"thunder/joints/{safe_name}/velocity", rr.Scalars(velocities[i]))
            if i < len(efforts):
                rr.log(f"thunder/joints/{safe_name}/effort", rr.Scalars(efforts[i]))

    # ------------------------------------------------------------------
    def _on_camera_image(self, msg: Image) -> None:
        t = self._ros_time_ns(msg.header)
        rr.set_time_nanos("ros_time", t)

        try:
            h, w = msg.height, msg.width
            enc = msg.encoding.lower()

            if enc in ("rgb8", "bgr8"):
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
                if enc == "bgr8":
                    arr = arr[:, :, ::-1]  # BGR → RGB
            elif enc in ("mono8", "8uc1"):
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
                arr = np.stack([arr] * 3, axis=-1)
            elif enc in ("yuv422", "yuyv"):
                raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 2)
                # Simple luminance-only fallback
                arr = np.stack([raw[:, :, 0]] * 3, axis=-1)
            else:
                logger.debug("Unsupported image encoding: %s", enc)
                return

            with self._image_lock:
                self._latest_image = arr.copy()

            rr.log("thunder/camera/image", rr.Image(arr))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to decode camera image: %s", exc)

    # ------------------------------------------------------------------
    def _on_detections(self, msg: String) -> None:
        import json

        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse /thunder/detections JSON: %s", exc)
            return

        if not detections:
            return

        # Log 2-D bounding boxes onto the camera image
        rr.log(
            "thunder/camera/detections",
            rr.Boxes2D(
                mins=[[d["x1"], d["y1"]] for d in detections],
                sizes=[[d["x2"] - d["x1"], d["y2"] - d["y1"]] for d in detections],
                labels=[f"{d.get('label','?')} {d.get('conf', 0):.2f}" for d in detections],
                colors=[[0, 220, 80]] * len(detections),
            ),
        )


# ---------------------------------------------------------------------------
# Blueprint — default Rerun layout
# ---------------------------------------------------------------------------
def _build_blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(name="Camera + Detections", origin="thunder/camera"),
                rrb.Spatial3DView(name="Body Orientation", origin="thunder/body_transform"),
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(name="IMU Orientation (deg)", origin="thunder/imu"),
                rrb.TimeSeriesView(name="Joint Positions (rad)", origin="thunder/joints"),
            ),
        ),
        collapse_panels=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rerun visualization bridge for Thunder")
    p.add_argument(
        "--addr",
        default="0.0.0.0:9876",
        help="Address to serve Rerun streaming on (default: 0.0.0.0:9876)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    host, _, port_str = args.addr.partition(":")
    port = int(port_str) if port_str else 9876

    # Init Rerun — serve mode so the laptop can connect
    rr.init("thunder_rerun_bridge", spawn=False)
    rr.serve_web(open_browser=False, web_port=port)
    logger.info("Rerun server listening on %s:%d", host, port)
    logger.info("On laptop: rerun --connect %s:%d", "192.168.66.190", port)

    rr.send_blueprint(_build_blueprint())

    # Init ROS2
    rclpy.init()
    node = RerunBridgeNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    def _shutdown(sig, frame):  # noqa: ARG001
        logger.info("Shutting down ...")
        executor.shutdown()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("Bridge running. Press Ctrl+C to stop.")
    try:
        executor.spin()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
