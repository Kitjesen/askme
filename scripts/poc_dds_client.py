#!/usr/bin/env python3
"""PoC: askme-side DDS bridge client.

Reads from the Unix domain socket published by poc_dds_bridge.py.
Demonstrates how askme can consume ROS2/DDS data without rclpy.

Usage (on S100P, in askme venv):
    python3 scripts/poc_dds_client.py

No ROS2 dependencies needed — pure Python asyncio + Unix socket.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time


SOCKET_PATH = "/tmp/askme_dds_bridge.sock"


class DdsBridgeClient:
    """Async client that reads DDS bridge messages via Unix socket."""

    def __init__(self) -> None:
        self._callbacks: dict[str, list] = {}
        self._connected = False
        self._msg_count = 0
        self._last_estop: bool | None = None
        self._last_detection_ts: float = 0.0

    def on(self, topic: str, callback) -> None:
        """Register a callback for a topic."""
        self._callbacks.setdefault(topic, []).append(callback)

    async def connect_and_read(self) -> None:
        """Connect to bridge socket and read messages forever."""
        while True:
            try:
                reader, writer = await asyncio.open_unix_connection(SOCKET_PATH)
                self._connected = True
                print(f"[DDS Client] Connected to {SOCKET_PATH}")

                buffer = b""
                while True:
                    chunk = await reader.read(4096)
                    if not chunk:
                        print("[DDS Client] Connection closed by bridge")
                        break

                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        await self._handle_message(line)

            except (ConnectionRefusedError, FileNotFoundError):
                self._connected = False
                print(f"[DDS Client] Bridge not available, retrying in 2s...")
                await asyncio.sleep(2.0)
            except Exception as e:
                self._connected = False
                print(f"[DDS Client] Error: {e}, reconnecting in 2s...")
                await asyncio.sleep(2.0)

    async def _handle_message(self, raw: bytes) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        self._msg_count += 1
        topic = msg.get("topic", "")
        data = msg.get("data", {})
        ts = msg.get("ts", 0.0)

        # Dispatch to callbacks
        for cb in self._callbacks.get(topic, []):
            if asyncio.iscoroutinefunction(cb):
                await cb(topic, data, ts)
            else:
                cb(topic, data, ts)


async def main() -> None:
    client = DdsBridgeClient()

    # Register handlers — this is what askme modules would do
    def on_detections(topic: str, data: dict, ts: float) -> None:
        dets = data.get("detections", [])
        frame_id = data.get("frame_id", "?")
        latency_ms = (time.time() - ts) * 1000
        labels = [d["label"] for d in dets]
        print(f"  [DETECT] frame={frame_id} objects={labels} latency={latency_ms:.1f}ms")

    def on_estop(topic: str, data: dict, ts: float) -> None:
        active = data.get("active", False)
        latency_ms = (time.time() - ts) * 1000
        status = "!! ACTIVE !!" if active else "normal"
        print(f"  [ESTOP]  state={status} latency={latency_ms:.1f}ms")

    client.on("/thunder/detections", on_detections)
    client.on("/thunder/estop", on_estop)

    # Stats printer
    async def print_stats():
        while True:
            await asyncio.sleep(10.0)
            print(f"  [STATS]  total_msgs={client._msg_count} connected={client._connected}")

    # Run both
    await asyncio.gather(
        client.connect_and_read(),
        print_stats(),
    )


if __name__ == "__main__":
    print("=== askme DDS Bridge Client PoC ===")
    print(f"Connecting to {SOCKET_PATH}...")
    print("No rclpy needed — pure asyncio\n")
    asyncio.run(main())
