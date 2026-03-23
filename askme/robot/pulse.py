"""Pulse — 脉搏数据总线.

在 askme 进程内直接订阅 ROS2 DDS 话题，后台线程 spin。
不需要 bridge 进程、socket、systemd service。

Usage::

    bus = Pulse(cfg)
    bus.on("/thunder/detections", my_callback)
    await bus.start()
    ...
    await bus.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )
    from sensor_msgs.msg import Imu, JointState
    from std_msgs.msg import Bool, String

    _RCLPY_AVAILABLE = True
except ImportError:
    _RCLPY_AVAILABLE = False

_QOS_SENSOR = None
_QOS_LATCHED = None

if _RCLPY_AVAILABLE:
    _QOS_SENSOR = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
    )
    _QOS_LATCHED = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
    )

# Topic → (msg_type, qos_key, parser)
_TOPIC_REGISTRY: dict[str, tuple] = {}

if _RCLPY_AVAILABLE:
    _TOPIC_REGISTRY = {
        "/thunder/detections": (String, "sensor", lambda m: json.loads(m.data)),
        "/thunder/estop": (Bool, "latched", lambda m: {"active": m.data}),
        "/thunder/heartbeat": (Bool, "sensor", lambda m: {"alive": m.data}),
        "/thunder/joint_states": (JointState, "sensor", lambda m: {
            "name": list(m.name),
            "position": list(m.position),
            "velocity": list(m.velocity),
            "effort": list(m.effort),
        }),
        "/thunder/imu": (Imu, "sensor", lambda m: {
            "angular_velocity": {"x": m.angular_velocity.x, "y": m.angular_velocity.y, "z": m.angular_velocity.z},
            "orientation": {"x": m.orientation.x, "y": m.orientation.y, "z": m.orientation.z, "w": m.orientation.w},
        }),
        "/thunder/cms_state": (String, "latched", lambda m: json.loads(m.data) if m.data.startswith("{") else {"state": m.data}),
    }


class Pulse:
    """脉搏数据总线 — 进程内直连 DDS.

    rclpy 在后台线程 spin，回调通过 ``call_soon_threadsafe`` 推到 asyncio。
    没有 bridge 进程、没有 socket、没有 reconnect 逻辑。
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self._enabled = cfg.get("enabled", _RCLPY_AVAILABLE)
        self._node_name = cfg.get("node_name", "askme")

        self._node: Any = None
        self._executor: Any = None
        self._spin_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

        # Topic data
        self._latest: dict[str, dict] = {}
        self._latest_lock = threading.Lock()
        self._callbacks: dict[str, list[Callable]] = {}
        self._msg_count = 0
        self._started = False

    @property
    def available(self) -> bool:
        return _RCLPY_AVAILABLE and self._enabled

    @property
    def connected(self) -> bool:
        return self._started and self._spin_thread is not None and self._spin_thread.is_alive()

    @property
    def msg_count(self) -> int:
        return self._msg_count

    async def start(self) -> None:
        """Start the bus — init rclpy, subscribe to all topics, spin in background thread."""
        if self._started:
            return
        if not self.available:
            logger.info("Pulse: disabled (rclpy not available)")
            return

        self._loop = asyncio.get_running_loop()

        if not rclpy.ok():
            rclpy.init()

        self._node = Node(self._node_name)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        # Subscribe to all registered topics
        for topic, (msg_type, qos_key, parser) in _TOPIC_REGISTRY.items():
            qos = _QOS_LATCHED if qos_key == "latched" else _QOS_SENSOR
            self._node.create_subscription(
                msg_type, topic,
                lambda msg, t=topic, p=parser: self._on_message(t, p, msg),
                qos,
            )

        # Spin in background thread
        self._stop_event.clear()
        self._spin_thread = threading.Thread(
            target=self._spin, name="pulse_spin", daemon=True,
        )
        self._spin_thread.start()
        self._started = True
        logger.info("Pulse: started (node=%s, topics=%d)", self._node_name, len(_TOPIC_REGISTRY))

    async def stop(self) -> None:
        """Stop the bus — shutdown rclpy executor and join thread."""
        if not self._started:
            return
        self._stop_event.set()
        if self._executor is not None:
            self._executor.shutdown()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=5.0)
            self._spin_thread = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        self._started = False
        logger.info("Pulse: stopped (total_msgs=%d)", self._msg_count)

    def _spin(self) -> None:
        """Background thread: spin rclpy executor until stop."""
        try:
            while not self._stop_event.is_set() and rclpy.ok():
                self._executor.spin_once(timeout_sec=0.1)
        except Exception as e:
            logger.warning("Pulse spin error: %s", e)

    def _on_message(self, topic: str, parser: Callable, raw_msg: Any) -> None:
        """Called from rclpy thread — parse and dispatch."""
        try:
            data = parser(raw_msg)
        except Exception:
            return

        data["_ts"] = time.time()
        self._msg_count += 1

        # Update latest cache (thread-safe)
        with self._latest_lock:
            self._latest[topic] = data

        # Dispatch callbacks to asyncio loop
        cbs = self._callbacks.get(topic)
        if cbs and self._loop is not None:
            for cb in cbs:
                if asyncio.iscoroutinefunction(cb):
                    self._loop.call_soon_threadsafe(
                        lambda c=cb, t=topic, d=data: asyncio.ensure_future(c(t, d)),
                    )
                else:
                    self._loop.call_soon_threadsafe(cb, topic, data)

    # ── Public API ──────────────────────────────────────

    def on(self, topic: str, callback: Callable) -> None:
        """Register a callback for a topic."""
        self._callbacks.setdefault(topic, []).append(callback)

    def get_latest(self, topic: str) -> dict | None:
        """Get the most recent message for a topic (thread-safe)."""
        with self._latest_lock:
            data = self._latest.get(topic)
            return dict(data) if data else None

    def is_estop_active(self) -> bool:
        """Read cached ESTOP state. Returns False if no data."""
        data = self.get_latest("/thunder/estop")
        if data is None:
            return False
        return bool(data.get("active", False))

    def get_detections(self) -> dict | None:
        """Get latest YOLO detections."""
        return self.get_latest("/thunder/detections")

    def get_joint_states(self) -> dict | None:
        """Get latest joint states."""
        return self.get_latest("/thunder/joint_states")

    def get_imu(self) -> dict | None:
        """Get latest IMU data."""
        return self.get_latest("/thunder/imu")

    def health(self) -> dict[str, Any]:
        """Health snapshot for runtime introspection."""
        return {
            "status": "ok" if self.connected else ("disabled" if not self.available else "disconnected"),
            "available": self.available,
            "connected": self.connected,
            "msg_count": self._msg_count,
            "topics": list(self._latest.keys()),
        }
