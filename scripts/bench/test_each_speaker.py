#!/usr/bin/env python3
"""Test each audio output device — play beep on each, user tells which one sounds."""
import subprocess
import numpy as np
import wave
import time

# Generate beep
sr = 48000
t = np.linspace(0, 1.0, int(sr * 1.0))
tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
with wave.open("/tmp/beep48k.wav", "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(tone.tobytes())

devices = [
    ("default", "default (PulseAudio/PipeWire)"),
    ("hw:0,0", "card 0: MCP01"),
    ("hw:1,0", "card 1: HK-MIC"),
]

for dev, label in devices:
    print(f"\n=== Playing on {label} ===", flush=True)
    try:
        result = subprocess.run(
            ["aplay", "-D", dev, "/tmp/beep48k.wav"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            print(f"  OK (played successfully)")
        else:
            print(f"  FAILED: {result.stderr.strip()[:100]}")
    except Exception as e:
        print(f"  ERROR: {e}")
    time.sleep(1)

print("\nWhich device did you hear the beep from?")
