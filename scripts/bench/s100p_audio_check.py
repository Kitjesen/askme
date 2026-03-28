#!/usr/bin/env python3
"""S100P audio device check: list devices, test HKMIC recording, test speaker."""
import sounddevice as sd
import numpy as np
import subprocess
import wave
import time

print("=== Audio Devices ===", flush=True)
devs = sd.query_devices()
for i, d in enumerate(devs):
    inp = d["max_input_channels"]
    out = d["max_output_channels"]
    if inp > 0 or out > 0:
        print(f"  [{i}] {d['name']} in={inp}ch out={out}ch rate={d['default_samplerate']}", flush=True)

# Test HKMIC (card 0) recording at native 48kHz
print("\n=== HKMIC Recording Test (48kHz 2ch, 3s) ===", flush=True)
try:
    rec = sd.rec(int(3 * 48000), samplerate=48000, channels=2, dtype="float32", device=0)
    sd.wait()
    for ch in range(2):
        peak = np.max(np.abs(rec[:, ch]))
        rms = np.sqrt(np.mean(rec[:, ch] ** 2))
        print(f"  ch{ch}: peak={peak:.4f}({int(peak*32768)}) rms={rms:.4f}({int(rms*32768)})", flush=True)
    max_peak = int(np.max(np.abs(rec)) * 32768)
    if max_peak > 50:
        print(f"  HKMIC: SIGNAL DETECTED (peak={max_peak})", flush=True)
    else:
        print(f"  HKMIC: SILENT (peak={max_peak})", flush=True)
except Exception as e:
    print(f"  HKMIC error: {e}", flush=True)

# Test MCP01 (card 1) recording
print("\n=== MCP01 Recording Test (16kHz 1ch, 2s) ===", flush=True)
try:
    rec2 = sd.rec(int(2 * 16000), samplerate=16000, channels=1, dtype="int16", device=1)
    sd.wait()
    peak2 = int(np.max(np.abs(rec2)))
    print(f"  MCP01: peak={peak2}", flush=True)
    if peak2 > 100:
        print("  MCP01: SIGNAL DETECTED", flush=True)
    else:
        print("  MCP01: SILENT (hardware fault?)", flush=True)
except Exception as e:
    print(f"  MCP01 error: {e}", flush=True)

# Test speaker output (generate beep)
print("\n=== Speaker Test ===", flush=True)
sr = 48000
t = np.linspace(0, 0.5, int(sr * 0.5))
tone = (np.sin(2 * np.pi * 800 * t) * 16000).astype(np.int16)
with wave.open("/tmp/beep_test.wav", "w") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(tone.tobytes())

for dev, label in [("default", "default"), ("plughw:1,0", "MCP01 direct")]:
    try:
        r = subprocess.run(["aplay", "-D", dev, "/tmp/beep_test.wav"],
                           capture_output=True, text=True, timeout=5)
        status = "OK" if r.returncode == 0 else f"FAIL: {r.stderr.strip()[:80]}"
        print(f"  {label}: {status}", flush=True)
    except Exception as e:
        print(f"  {label}: ERROR {e}", flush=True)

print("\nDone.", flush=True)
