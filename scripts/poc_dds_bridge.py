#!/usr/bin/env python3
"""PoC: ROS2 DDS bridge for askme.

Runs as a standalone process (system Python 3.10 + rclpy).
Subscribes to ROS2 topics and pushes data to askme via Unix domain socket.

Usage (on S100P):
    source /opt/ros/humble/setup.bash
    python3 scripts/poc_dds_bridge.py

askme reads from the socket without needing rclpy.

Topics subscribed:
    /thunder/detections  (std_msgs/String, JSON)  — from frame_daemon
    /thunder/estop       (std_msgs/Bool)           — from safety bridge

Socket protocol (newline-delimited JSON):
    {"topic": "/thunder/detections", "data": {...}, "ts": 1234567890.123}
    {"topic": "/thunder/estop", "data": {"active": true}, "ts": 1234567890.456}
"""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

# Must be sourced: /opt/ros/humble/setup.bash
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Bool, String

SOCKET_PATH = "/tmp/askme_dds_bridge.sock"

# QoS profiles
QOS_SENSOR = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

QOS_RELIABLE_LATCHED = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class DdsBridge(Node):
    """ROS2 node that bridges topics to a Unix domain socket."""

    def __init__(self) -> None:
        super().__init__("askme_dds_bridge")

        self._sock_clients: list[socket.socket] = []
        self._sock_clients_lock = threading.Lock()

        # Subscribe to detection topic (JSON string from frame_daemon)
        self.create_subscription(
            String,
            "/thunder/detections",
            self._on_detections,
            QOS_SENSOR,
        )
        self.get_logger().info("Subscribed to /thunder/detections")

        # Subscribe to ESTOP topic (latched bool)
        self.create_subscription(
            Bool,
            "/thunder/estop",
            self._on_estop,
            QOS_RELIABLE_LATCHED,
        )
        self.get_logger().info("Subscribed to /thunder/estop")

        # Timer to log stats every 10s
        self._msg_count = 0
        self.create_timer(10.0, self._log_stats)

    def _broadcast(self, topic: str, data: dict) -> None:
        """Send a message to all connected socket clients."""
        msg = json.dumps(
            {"topic": topic, "data": data, "ts": time.time()},
            ensure_ascii=False,
        ) + "\n"
        raw = msg.encode("utf-8")

        dead: list[socket.socket] = []
        with self._sock_clients_lock:
            for client in self._sock_clients:
                try:
                    client.sendall(raw)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    dead.append(client)
            for c in dead:
                self._sock_clients.remove(c)
                c.close()

    def _on_detections(self, msg: String) -> None:
        self._msg_count += 1
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            data = {"raw": msg.data}
        self._broadcast("/thunder/detections", data)

    def _on_estop(self, msg: Bool) -> None:
        self._msg_count += 1
        self._broadcast("/thunder/estop", {"active": msg.data})
        level = "WARN" if msg.data else "INFO"
        self.get_logger().info(f"ESTOP state: {'ACTIVE' if msg.data else 'normal'}")

    def _log_stats(self) -> None:
        with self._sock_clients_lock:
            n = len(self._sock_clients)
        self.get_logger().info(f"Bridge stats: {self._msg_count} msgs, {n} clients")

    def add_client(self, client: socket.socket) -> None:
        with self._sock_clients_lock:
            self._sock_clients.append(client)
        self.get_logger().info(f"Client connected ({len(self._sock_clients)} total)")


def run_socket_server(bridge: DdsBridge) -> None:
    """Accept Unix socket connections in a background thread."""
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(5)
    server.settimeout(1.0)
    bridge.get_logger().info(f"Socket server listening on {SOCKET_PATH}")

    while rclpy.ok():
        try:
            client, _ = server.accept()
            client.setblocking(True)
            bridge.add_client(client)
        except socket.timeout:
            continue
        except OSError:
            break

    server.close()
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)


def main() -> None:
    rclpy.init()
    bridge = DdsBridge()

    # Socket server in background thread
    sock_thread = threading.Thread(target=run_socket_server, args=(bridge,), daemon=True)
    sock_thread.start()

    # Spin ROS2 in main thread
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)

    def shutdown(sig, frame):
        bridge.get_logger().info("Shutting down...")
        rclpy.shutdown()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        executor.spin()
    finally:
        bridge.destroy_node()
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)


if __name__ == "__main__":
    main()
