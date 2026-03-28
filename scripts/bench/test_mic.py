#!/usr/bin/env python3
"""Test microphone on S100P."""
import sounddevice as sd
import numpy as np

# List input devices
devices = sd.query_devices()
for i, d in enumerate(devices):
    if d["max_input_channels"] > 0:
        print(f"  [{i}] {d['name']} (in={d['max_input_channels']}ch)")

print()
print("Recording 2 seconds...")
rec = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype="int16")
sd.wait()
peak = int(np.max(np.abs(rec)))
rms = int(np.sqrt(np.mean(rec.astype(np.float32) ** 2)))
print(f"MIC peak={peak}, rms={rms}")

if peak > 100:
    print("MIC OK!")
else:
    print("MIC still dead (peak too low)")
