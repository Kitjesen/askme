"""Test voice components: ASR, VAD, KWS individually and then mic → ASR live.

Tests:
  1. ASR on bundled test wav files (offline accuracy check)
  2. VAD on synthetic audio (silence vs speech detection)
  3. KWS model loading and stream creation
  4. Live mic → VAD + ASR (speak into microphone, see real-time transcription)
"""

import io
import sys
import time

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np


def test_asr_offline():
    """Test ASR on bundled test wav files."""
    import wave
    import os

    from askme.voice.asr import ASREngine
    from askme.config import get_config

    cfg = get_config()
    asr_cfg = cfg.get("voice", {}).get("asr", {})
    model_dir = asr_cfg.get(
        "model_dir",
        "models/asr/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
    )

    print("=" * 50)
    print("[1] ASR Offline Test (test wav files)")
    print("=" * 50)

    asr = ASREngine(asr_cfg)
    print(f"  Model: {os.path.basename(model_dir)}")
    print(f"  Sample rate: {asr.sample_rate}")

    wav_dir = os.path.join(model_dir, "test_wavs")
    if not os.path.isdir(wav_dir):
        print("  SKIP: No test_wavs directory found")
        return

    wav_files = sorted(f for f in os.listdir(wav_dir) if f.endswith(".wav"))
    print(f"  Found {len(wav_files)} test files\n")

    for wav_name in wav_files:
        wav_path = os.path.join(wav_dir, wav_name)

        try:
            with wave.open(wav_path, "rb") as wf:
                sr = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

                # Resample if needed
                if sr != asr.sample_rate:
                    ratio = asr.sample_rate / sr
                    new_len = int(len(samples) * ratio)
                    indices = np.linspace(0, len(samples) - 1, new_len)
                    samples = np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)

            stream = asr.create_stream()
            t0 = time.monotonic()

            # Feed in chunks
            chunk_size = int(0.1 * asr.sample_rate)
            for i in range(0, len(samples), chunk_size):
                chunk = samples[i : i + chunk_size]
                stream.accept_waveform(asr.sample_rate, chunk)
                while asr.is_ready(stream):
                    asr.decode_stream(stream)

            # Flush remaining
            tail_padding = np.zeros(int(0.5 * asr.sample_rate), dtype=np.float32)
            stream.accept_waveform(asr.sample_rate, tail_padding)
            while asr.is_ready(stream):
                asr.decode_stream(stream)

            result = asr.get_result(stream).strip()
            dur = time.monotonic() - t0
            audio_dur = len(samples) / asr.sample_rate

            print(f"  {wav_name}: \"{result}\"")
            print(f"    audio={audio_dur:.1f}s  process={dur:.2f}s  RTF={dur / audio_dur:.2f}")

        except Exception as e:
            print(f"  {wav_name}: ERROR - {e}")

    print()


def test_vad():
    """Test VAD on synthetic audio."""
    from askme.voice.vad import VADEngine
    from askme.config import get_config

    cfg = get_config()
    vad_cfg = cfg.get("voice", {}).get("vad", {})

    print("=" * 50)
    print("[2] VAD Test (synthetic audio)")
    print("=" * 50)

    vad = VADEngine(vad_cfg)
    sr = int(vad_cfg.get("sample_rate", 16000))

    # Test 1: Pure silence → should NOT detect speech
    silence = np.zeros(sr, dtype=np.int16)  # 1 second
    vad.accept_waveform(silence)
    speech_in_silence = vad.is_speech_detected()
    print(f"  Silence (1s): speech_detected = {speech_in_silence} (expected: False)")

    # Test 2: Loud tone → should detect speech
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    tone = (0.8 * np.sin(2 * np.pi * 300 * t) * 32768).astype(np.int16)
    vad.accept_waveform(tone)
    speech_in_tone = vad.is_speech_detected()
    print(f"  300Hz tone (1s): speech_detected = {speech_in_tone} (expected: True)")

    status = "PASS" if (not speech_in_silence and speech_in_tone) else "PARTIAL"
    print(f"  Result: {status}\n")


def test_kws():
    """Test KWS model loading."""
    from askme.voice.kws import KWSEngine
    from askme.config import get_config

    cfg = get_config()
    kws_cfg = cfg.get("voice", {}).get("kws", {})

    print("=" * 50)
    print("[3] KWS Test (wake word model)")
    print("=" * 50)

    kws = KWSEngine(kws_cfg)
    print(f"  Available: {kws.available}")

    if kws.available:
        stream = kws.create_stream()
        print(f"  Stream created: {stream is not None}")
        keywords = kws_cfg.get("keywords", ["你好", "小智"])
        print(f"  Keywords: {keywords}")
    else:
        print("  SKIP: KWS model not found")

    print()


def test_live_mic():
    """Live microphone → VAD + ASR test. Speak and see real-time transcription."""
    import sounddevice as sd

    from askme.voice.asr import ASREngine
    from askme.voice.vad import VADEngine
    from askme.config import get_config

    cfg = get_config()
    asr_cfg = cfg.get("voice", {}).get("asr", {})
    vad_cfg = cfg.get("voice", {}).get("vad", {})

    print("=" * 50)
    print("[4] Live Mic → VAD + ASR Test")
    print("    Speak into microphone (15 seconds)")
    print("    Say something in Chinese or English")
    print("=" * 50)

    asr = ASREngine(asr_cfg)
    vad = VADEngine(vad_cfg)
    stream = asr.create_stream()

    sr = asr.sample_rate
    chunk_dur = 0.1  # 100ms
    chunk_size = int(sr * chunk_dur)
    total_dur = 15.0  # seconds
    deadline = time.monotonic() + total_dur

    speech_active = False
    results = []

    print(f"\n  Listening ({total_dur:.0f}s)...\n")

    with sd.InputStream(channels=1, dtype="float32", samplerate=sr) as mic:
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            samples, _ = mic.read(chunk_size)
            samples = samples.reshape(-1)

            # VAD
            samples_int16 = (samples * 32768).astype(np.int16)
            vad.accept_waveform(samples_int16)

            is_speech = vad.is_speech_detected()

            if is_speech:
                if not speech_active:
                    speech_active = True
                    print(f"  [{remaining:.0f}s] VAD: speech START", end="", flush=True)

                stream.accept_waveform(sr, samples)
                while asr.is_ready(stream):
                    asr.decode_stream(stream)

                # Show partial result
                partial = asr.get_result(stream).strip()
                if partial:
                    print(f"\r  [{remaining:.0f}s] Partial: {partial}          ", end="", flush=True)
            else:
                if speech_active:
                    speech_active = False
                    stream.accept_waveform(sr, samples)
                    while asr.is_ready(stream):
                        asr.decode_stream(stream)

                    # Check endpoint
                    if asr.is_endpoint(stream):
                        text = asr.get_result(stream).strip()
                        if text:
                            print(f"\n  FINAL: \"{text}\"")
                            results.append(text)
                        asr.reset(stream)
                        stream = asr.create_stream()

    # Flush remaining
    text = asr.get_result(stream).strip()
    if text:
        print(f"\n  FINAL (flush): \"{text}\"")
        results.append(text)

    print(f"\n  Total recognized: {len(results)} utterances")
    for i, r in enumerate(results, 1):
        print(f"    {i}. {r}")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Include live mic test")
    args = parser.parse_args()

    test_asr_offline()
    test_vad()
    test_kws()

    if args.live:
        test_live_mic()
    else:
        print("=" * 50)
        print("[4] Live Mic Test — SKIPPED (use --live to enable)")
        print("=" * 50)

    print("\nAll voice component tests done!")
