#!/usr/bin/env python3
"""Persistent frame capture daemon for askme vision.

Runs as a systemd service (system Python, NOT venv).
Continuously writes the latest RGB + Depth frames to /tmp/askme_frame_*.bin
so askme can read them instantly without subprocess overhead.

Usage:
    python3 scripts/frame_daemon.py                      # default topics
    python3 scripts/frame_daemon.py --color /camera/color/image_raw --depth /camera/depth/image_raw

Install as service:
    sudo cp scripts/askme-frame-daemon.service /etc/systemd/system/
    sudo systemctl enable --now askme-frame-daemon
"""

import argparse
import struct
import time
import os
import signal
import sys

# Paths where frames are written (atomically via rename)
COLOR_PATH = "/tmp/askme_frame_color.bin"
DEPTH_PATH = "/tmp/askme_frame_depth.bin"
HEARTBEAT_PATH = "/tmp/askme_frame_daemon.heartbeat"


def main():
    parser = argparse.ArgumentParser(description="Askme frame capture daemon")
    parser.add_argument("--color", default="/camera/color/image_raw", help="Color image topic")
    parser.add_argument("--depth", default="/camera/depth/image_raw", help="Depth image topic")
    parser.add_argument("--rate", type=float, default=5.0, help="Max capture rate (Hz)")
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from sensor_msgs.msg import Image

    rclpy.init()
    node = Node("askme_frame_daemon")

    color_frame = [None]
    depth_frame = [None]

    def on_color(msg):
        color_frame[0] = msg

    def on_depth(msg):
        depth_frame[0] = msg

    node.create_subscription(Image, args.color, on_color, 1)
    node.create_subscription(Image, args.depth, on_depth, 1)

    min_interval = 1.0 / args.rate
    running = True

    def shutdown(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    print(f"[frame_daemon] Started: color={args.color} depth={args.depth} rate={args.rate}Hz")
    sys.stdout.flush()

    last_write = 0
    while running:
        executor.spin_once(timeout_sec=0.1)

        now = time.monotonic()
        if now - last_write < min_interval:
            continue

        # Write color frame atomically
        if color_frame[0] is not None:
            msg = color_frame[0]
            tmp = COLOR_PATH + ".tmp"
            try:
                with open(tmp, "wb") as f:
                    f.write(struct.pack("II", msg.width, msg.height))
                    f.write(bytes(msg.data))
                os.rename(tmp, COLOR_PATH)
            except Exception:
                pass

        # Write depth frame atomically
        if depth_frame[0] is not None:
            msg = depth_frame[0]
            tmp = DEPTH_PATH + ".tmp"
            try:
                with open(tmp, "wb") as f:
                    f.write(struct.pack("II", msg.width, msg.height))
                    f.write(bytes(msg.data))
                os.rename(tmp, DEPTH_PATH)
            except Exception:
                pass

        # Heartbeat
        try:
            with open(HEARTBEAT_PATH, "w") as f:
                f.write(f"{time.time():.3f}\n")
        except Exception:
            pass

        last_write = now

    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()
    print("[frame_daemon] Stopped")


if __name__ == "__main__":
    main()
