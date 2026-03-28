#!/usr/bin/env python3
"""Test raw WebSocket connection to DashScope."""
import os
import websocket
import json

api_key = os.environ.get("DASHSCOPE_API_KEY", "")
print(f"API key: {'set' if api_key else 'MISSING'}", flush=True)

url = "wss://dashscope.aliyuncs.com/api-ws/v1/inference/"
print(f"Connecting to {url}...", flush=True)

try:
    ws = websocket.WebSocket()
    ws.settimeout(10)
    ws.connect(url, header=[f"Authorization: bearer {api_key}"])
    print("Connected!", flush=True)

    # Send a minimal run-task
    msg = {
        "header": {
            "action": "run-task",
            "task_id": "test-123",
            "streaming": "duplex",
        },
        "payload": {
            "task_group": "audio",
            "task": "asr",
            "function": "recognition",
            "model": "paraformer-realtime-v2",
            "parameters": {"sample_rate": 16000, "format": "pcm"},
            "input": {},
        },
    }
    ws.send(json.dumps(msg))
    print("Sent run-task", flush=True)

    ack = ws.recv()
    ack_data = json.loads(ack)
    event = ack_data.get("header", {}).get("event", "")
    print(f"Ack: {event}", flush=True)

    ws.close()
    print("DashScope WebSocket OK!", flush=True)

except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}", flush=True)
