#!/usr/bin/env python3
"""Check default audio output device."""
import sounddevice as sd

d = sd.query_devices()
default_out = sd.default.device[1]
print(f"Default output device index: {default_out}")
for i, dev in enumerate(d):
    if dev["max_output_channels"] > 0:
        name = dev["name"]
        ch = dev["max_output_channels"]
        rate = dev["default_samplerate"]
        marker = " <<< DEFAULT" if i == default_out else ""
        print(f"  [{i}] {name} out={ch}ch rate={rate}{marker}")
