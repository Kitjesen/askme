#!/usr/bin/env python3
"""Benchmark Pulse DDS in-process latency on S100P."""
import asyncio
import time
from askme.robot.pulse import Pulse

async def main():
    pulse = Pulse({"enabled": True, "node_name": "askme_bench"})
    await pulse.start()
    print(f"Connected: {pulse.connected}")

    latencies = []
    received = asyncio.Event()

    def on_det(topic, data):
        ts = data.get("_ts", 0)
        if ts:
            latencies.append((time.time() - ts) * 1000)
        if len(latencies) >= 50:
            received.set()

    pulse.on("/thunder/detections", on_det)
    print("Waiting for 50 detection messages...")

    try:
        await asyncio.wait_for(received.wait(), timeout=15.0)
    except asyncio.TimeoutError:
        print(f"Timeout, got {len(latencies)} msgs")

    await pulse.stop()

    if latencies:
        latencies.sort()
        n = len(latencies)
        print(f"\n=== Pulse DDS In-Process Benchmark ===")
        print(f"Messages: {n}")
        print(f"Avg:  {sum(latencies)/n:.3f}ms")
        print(f"Min:  {latencies[0]:.3f}ms")
        print(f"Max:  {latencies[-1]:.3f}ms")
        print(f"P50:  {latencies[n//2]:.3f}ms")
        print(f"P95:  {latencies[int(n*0.95)]:.3f}ms")
        print(f"P99:  {latencies[min(int(n*0.99), n-1)]:.3f}ms")
    else:
        print("No messages received (frame_daemon not running?)")

if __name__ == "__main__":
    asyncio.run(main())
