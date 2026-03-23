#!/usr/bin/env python3
"""Brainstem gRPC → ROS2 bridge.

Subscribes to brainstem's gRPC streaming RPCs (ListenImu, ListenJoint)
and republishes as standard ROS2 messages on /thunder/* topics.

Also subscribes to /cmd_vel and forwards to brainstem's Walk() RPC.

Usage (on S100P):
    source /opt/ros/humble/setup.bash
    export PYTHONPATH=/home/sunrise/askme/proto/python:$PYTHONPATH
    python3 scripts/brainstem_ros2_bridge.py [--host 127.0.0.1] [--port 13145]

Requires:
    - rclpy (system Python 3.10 + ROS2 Humble)
    - grpcio + protobuf
    - han_dog_message Python stubs (shared/proto/python/)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path

# Add proto stubs to path
_PROTO_DIR = str(Path(__file__).resolve().parent.parent / "proto" / "python")
if _PROTO_DIR not in sys.path:
    sys.path.insert(0, _PROTO_DIR)
# Also try shared/proto/python
_SHARED_PROTO = str(Path(__file__).resolve().parent.parent.parent / "shared" / "proto" / "python")
if _SHARED_PROTO not in sys.path:
    sys.path.insert(0, _SHARED_PROTO)

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from geometry_msgs.msg import Twist, Vector3 as RosVector3
from sensor_msgs.msg import Imu as RosImu, JointState as RosJointState
from std_msgs.msg import Bool, String

import grpc
from grpc import aio as grpc_aio

logger = logging.getLogger("brainstem_bridge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

# QoS profiles
_QOS_SENSOR = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

_QOS_RELIABLE = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

# 16 joint names for Thunder quadruped: 4 legs x 3 joints + 4 ankles
_JOINT_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "FL_ankle", "FR_ankle", "RL_ankle", "RR_ankle",
]


class BrainstemBridge(Node):
    """ROS2 node bridging brainstem gRPC streams to ROS2 topics."""

    def __init__(self, host: str = "127.0.0.1", port: int = 13145) -> None:
        super().__init__("brainstem_ros2_bridge")
        self._addr = f"{host}:{port}"
        self._connected = False
        self._imu_count = 0
        self._joint_count = 0
        self._cmd_vel_count = 0

        # Publishers
        self._pub_imu = self.create_publisher(RosImu, "/thunder/imu", _QOS_SENSOR)
        self._pub_joints = self.create_publisher(RosJointState, "/thunder/joint_states", _QOS_SENSOR)
        self._pub_state = self.create_publisher(String, "/thunder/cms_state", _QOS_RELIABLE)

        # Subscriber: /cmd_vel → Walk() RPC
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, _QOS_SENSOR)

        # gRPC channel and stub (created in async context)
        self._channel: grpc_aio.Channel | None = None
        self._stub = None
        self._walk_stub = None

        # Stats timer
        self.create_timer(10.0, self._log_stats)
        self.get_logger().info(f"BrainstemBridge targeting {self._addr}")

    def _log_stats(self) -> None:
        self.get_logger().info(
            f"Stats: imu={self._imu_count} joints={self._joint_count} "
            f"cmd_vel={self._cmd_vel_count} connected={self._connected}"
        )

    def _on_cmd_vel(self, msg: Twist) -> None:
        """Forward /cmd_vel to brainstem Walk() RPC (fire-and-forget)."""
        self._cmd_vel_count += 1
        if self._walk_stub is not None:
            # Schedule async call without blocking the ROS callback
            asyncio.ensure_future(self._send_walk(msg))

    async def _send_walk(self, twist: Twist) -> None:
        """Send Walk() RPC with Twist → Vector3 mapping."""
        try:
            import han_dog_message as proto
            vec = proto.Vector3(
                x=twist.linear.x,
                y=twist.linear.y,
                z=twist.angular.z,
            )
            await self._stub.Walk(vec)
        except Exception as e:
            self.get_logger().debug(f"Walk RPC failed: {e}")


async def _run_imu_stream(bridge: BrainstemBridge, stub) -> None:
    """Subscribe to ListenImu() and publish /thunder/imu."""
    logger.info("Starting IMU stream...")
    try:
        async for imu_msg in stub.ListenImu(
            __import__("han_dog_message").Empty()
        ):
            ros_imu = RosImu()
            ros_imu.header.stamp = bridge.get_clock().now().to_msg()
            ros_imu.header.frame_id = "base_link"

            # Angular velocity (gyroscope)
            ros_imu.angular_velocity.x = imu_msg.gyroscope.x
            ros_imu.angular_velocity.y = imu_msg.gyroscope.y
            ros_imu.angular_velocity.z = imu_msg.gyroscope.z

            # Orientation quaternion (proto: w,x,y,z → ROS: x,y,z,w)
            if imu_msg.HasField("quaternion"):
                q = imu_msg.quaternion
                ros_imu.orientation.x = q.x
                ros_imu.orientation.y = q.y
                ros_imu.orientation.z = q.z
                ros_imu.orientation.w = q.w

            bridge._pub_imu.publish(ros_imu)
            bridge._imu_count += 1
    except grpc.RpcError as e:
        logger.warning(f"IMU stream ended: {e.code()}")
    except Exception as e:
        logger.error(f"IMU stream error: {e}")


async def _run_joint_stream(bridge: BrainstemBridge, stub) -> None:
    """Subscribe to ListenJoint() and publish /thunder/joint_states."""
    logger.info("Starting Joint stream...")
    try:
        import han_dog_message as proto
        async for joint_msg in stub.ListenJoint(proto.Empty()):
            ros_js = RosJointState()
            ros_js.header.stamp = bridge.get_clock().now().to_msg()
            ros_js.header.frame_id = "base_link"

            # Handle AllJoints snapshot
            if joint_msg.HasField("all_joints"):
                aj = joint_msg.all_joints
                ros_js.name = list(_JOINT_NAMES)

                # Matrix4 is 4x4=16 floats stored as 4 rows of 4
                positions = _matrix4_to_list(aj.position)
                velocities = _matrix4_to_list(aj.velocity)
                torques = _matrix4_to_list(aj.torque)

                ros_js.position = positions
                ros_js.velocity = velocities
                ros_js.effort = torques

            # Handle SingleJoint report
            elif joint_msg.HasField("single_joint"):
                sj = joint_msg.single_joint
                idx = sj.id
                if idx < len(_JOINT_NAMES):
                    ros_js.name = [_JOINT_NAMES[idx]]
                    ros_js.position = [sj.position]
                    ros_js.velocity = [sj.velocity]
                    ros_js.effort = [sj.torque]
                else:
                    continue

            bridge._pub_joints.publish(ros_js)
            bridge._joint_count += 1
    except grpc.RpcError as e:
        logger.warning(f"Joint stream ended: {e.code()}")
    except Exception as e:
        logger.error(f"Joint stream error: {e}")


def _matrix4_to_list(m) -> list[float]:
    """Convert brainstem Matrix4 (4 rows of 4) to flat list of 16 floats."""
    result = []
    for row in [m.row0, m.row1, m.row2, m.row3]:
        result.extend([row.x, row.y, row.z, row.w])
    return result


async def _grpc_loop(bridge: BrainstemBridge) -> None:
    """Main gRPC connection loop with auto-reconnect."""
    import han_dog_message as proto

    while rclpy.ok():
        try:
            logger.info(f"Connecting to brainstem at {bridge._addr}...")
            bridge._channel = grpc_aio.insecure_channel(bridge._addr)
            bridge._stub = proto.CmsStub(bridge._channel)
            bridge._walk_stub = bridge._stub
            bridge._connected = True
            logger.info("Connected to brainstem gRPC")

            # Publish initial CMS state
            state_msg = String()
            state_msg.data = '{"state": "connected", "addr": "' + bridge._addr + '"}'
            bridge._pub_state.publish(state_msg)

            # Run IMU and Joint streams concurrently
            await asyncio.gather(
                _run_imu_stream(bridge, bridge._stub),
                _run_joint_stream(bridge, bridge._stub),
            )

        except Exception as e:
            logger.warning(f"gRPC connection failed: {e}")
            bridge._connected = False

        # Publish disconnected state
        state_msg = String()
        state_msg.data = '{"state": "disconnected"}'
        bridge._pub_state.publish(state_msg)

        logger.info("Reconnecting in 3s...")
        await asyncio.sleep(3.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Brainstem gRPC → ROS2 bridge")
    parser.add_argument("--host", default="127.0.0.1", help="brainstem gRPC host")
    parser.add_argument("--port", type=int, default=13145, help="brainstem gRPC port")
    args = parser.parse_args()

    rclpy.init()
    bridge = BrainstemBridge(host=args.host, port=args.port)

    # Run gRPC streams in asyncio, ROS2 spin in executor
    executor = MultiThreadedExecutor()
    executor.add_node(bridge)

    loop = asyncio.new_event_loop()

    def shutdown(sig, frame):
        logger.info("Shutting down...")
        loop.stop()
        rclpy.shutdown()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start gRPC loop in background
    import threading
    grpc_thread = threading.Thread(
        target=lambda: asyncio.run(_grpc_loop(bridge)),
        daemon=True,
    )
    grpc_thread.start()

    # Spin ROS2 in main thread
    try:
        executor.spin()
    finally:
        bridge.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
