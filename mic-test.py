"""Quick microphone diagnostic — run from your own terminal."""
import sounddevice as sd
import numpy as np

print("=" * 50)
print("  Microphone Diagnostic")
print("=" * 50)
print()

# List input devices
print("Input devices:")
for i, dev in enumerate(sd.query_devices()):
    if dev["max_input_channels"] > 0:
        marker = " >> DEFAULT" if i == sd.default.device[0] else ""
        print(f"  [{i}] {dev['name']} (ch={dev['max_input_channels']}){marker}")

default_idx = sd.default.device[0]
print(f"\nTesting default input device [{default_idx}]...")
print("Speak now! (recording 3 seconds...)")

audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype="int16", device=default_idx)
sd.wait()

peak = int(np.max(np.abs(audio)))
rms = float(np.sqrt(np.mean(audio.astype(float) ** 2)))

print(f"\nResult: peak={peak}, rms={rms:.1f}")

if peak < 100:
    print("\n[FAIL] Mic is silent. Possible causes:")
    print("  1. Windows Settings > Privacy > Microphone — allow Desktop apps")
    print("  2. Mic is muted in system tray / Realtek Audio Console")
    print("  3. Wrong default device selected")
    print("\nTrying all input devices...")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and i != default_idx:
            try:
                print(f"  [{i}] {dev['name']}...", end=" ", flush=True)
                a = sd.rec(int(1 * 16000), samplerate=16000, channels=1, dtype="int16", device=i)
                sd.wait()
                p = int(np.max(np.abs(a)))
                print(f"peak={p}" + (" << HAS SIGNAL!" if p > 100 else ""))
            except Exception as e:
                print(f"error: {e}")
else:
    print("\n[OK] Microphone is working!")

input("\nPress Enter to exit...")
