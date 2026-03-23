#!/usr/bin/env python3
"""PoC: Publish test messages on ROS2 topics for bridge testing.

Usage (on S100P):
    source /opt/ros/humble/setup.bash
    python3 scripts/poc_dds_publisher.py

Publishes:
    /thunder/detections (String, JSON) at 5Hz — simulated YOLO detections
    /thunder/estop (Bool) — toggles every 10s
"""

from __future__ import annotations

import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Bool, String


class TestPublisher(Node):
    def __init__(self) -> None:
        super().__init__("thunder_test_publisher")

        qos_sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._det_pub = self.create_publisher(String, "/thunder/detections", qos_sensor)
        self._estop_pub = self.create_publisher(Bool, "/thunder/estop", qos_latched)

        self._tick = 0
        self.create_timer(0.2, self._publish_detections)  # 5Hz
        self.create_timer(10.0, self._toggle_estop)

        # Publish initial ESTOP=false
        msg = Bool()
        msg.data = False
        self._estop_pub.publish(msg)
        self.get_logger().info("Test publisher started: detections@5Hz, estop toggle@10s")

    def _publish_detections(self) -> None:
        self._tick += 1
        det = {
            "timestamp": time.time(),
            "detections": [
                {
                    "label": "person",
                    "confidence": 0.92,
                    "bbox": [100, 50, 200, 300],
                    "distance_m": 2.3,
                },
            ],
            "frame_id": self._tick,
        }
        msg = String()
        msg.data = json.dumps(det, ensure_ascii=False)
        self._det_pub.publish(msg)
        if self._tick % 25 == 0:  # log every 5s
            self.get_logger().info(f"Published {self._tick} detection frames")

    def _toggle_estop(self) -> None:
        active = (self._tick // 50) % 2 == 1
        msg = Bool()
        msg.data = active
        self._estop_pub.publish(msg)
        self.get_logger().info(f"ESTOP toggled to {'ACTIVE' if active else 'normal'}")


def main() -> None:
    rclpy.init()
    node = TestPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
