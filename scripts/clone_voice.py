#!/usr/bin/env python3
"""MiniMax Voice Clone — upload audio sample and register a custom voice_id.

Usage::

    # Clone from a 10s+ audio file
    python scripts/clone_voice.py samples/my_voice.wav --voice-id thunder-voice-01

    # Clone with custom test text
    python scripts/clone_voice.py samples/my_voice.wav --voice-id thunder-voice-01 \
        --text "你好，我是Thunder巡检机器人"

    # Then set in config.yaml:  minimax_voice_id: thunder-voice-01

Requirements:
    - Audio: mp3/m4a/wav, 10s–5min, <20MB
    - voice_id: 8+ chars, letters+numbers, starts with letter
    - Cloned voice expires in 7 days if not used in any T2A call
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_api_key() -> str:
    """Load MiniMax API key from .env or environment."""
    key = os.environ.get("MINIMAX_API_KEY", "")
    if key:
        return key

    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.isfile(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("MINIMAX_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def upload_audio(api_key: str, base_url: str, audio_path: str) -> str:
    """Upload audio file for voice cloning. Returns file_id."""
    import httpx

    url = f"{base_url}/files/upload"
    headers = {"Authorization": f"Bearer {api_key}"}

    filename = os.path.basename(audio_path)
    ext = os.path.splitext(filename)[1].lower()
    mime_map = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4"}
    mime = mime_map.get(ext, "audio/wav")

    with open(audio_path, "rb") as f:
        files = {"file": (filename, f, mime)}
        data = {"purpose": "voice_clone"}
        print(f"  Uploading {filename} ({os.path.getsize(audio_path) / 1024:.0f} KB)...")
        resp = httpx.post(url, headers=headers, files=files, data=data, timeout=60.0)

    if resp.status_code != 200:
        print(f"  ERROR: Upload failed HTTP {resp.status_code}: {resp.text[:300]}")
        sys.exit(1)

    result = resp.json()
    file_obj = result.get("file", result)
    file_id = file_obj.get("file_id", "")
    if not file_id:
        print(f"  ERROR: No file_id in response: {json.dumps(result, indent=2)}")
        sys.exit(1)

    print(f"  File uploaded: file_id={file_id}")
    return file_id


def clone_voice(
    api_key: str,
    base_url: str,
    file_id: str,
    voice_id: str,
    text: str,
    model: str,
    *,
    noise_reduction: bool = False,
    volume_normalization: bool = False,
) -> dict:
    """Register a cloned voice. Returns the API response."""
    import httpx

    url = f"{base_url}/voice_clone"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "file_id": file_id,
        "voice_id": voice_id,
        "model": model,
        "text": text,
        "need_noise_reduction": noise_reduction,
        # Note: official API uses "volumn" (their typo, not ours)
        "need_volumn_normalization": volume_normalization,
    }

    print(f"  Cloning voice as '{voice_id}' with model '{model}'...")
    resp = httpx.post(url, json=body, headers=headers, timeout=60.0)

    if resp.status_code != 200:
        print(f"  ERROR: Clone failed HTTP {resp.status_code}: {resp.text[:500]}")
        sys.exit(1)

    result = resp.json()
    print(f"  Clone response: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMax Voice Clone")
    parser.add_argument("audio", help="Path to audio file (wav/mp3/m4a, 10s-5min)")
    parser.add_argument("--voice-id", required=True, help="Custom voice ID (8+ chars, starts with letter)")
    parser.add_argument("--text", default="你好，我是Thunder，穹沛科技的巡检机器人。", help="Test text for verification")
    parser.add_argument("--model", default="speech-2.8-turbo", help="TTS model (default: speech-2.8-turbo)")
    parser.add_argument("--noise-reduction", action="store_true", help="Enable noise reduction on source audio")
    parser.add_argument("--volume-norm", action="store_true", help="Enable volume normalization on source audio")
    parser.add_argument("--base-url", default="https://api.minimax.chat/v1", help="MiniMax API base URL")
    args = parser.parse_args()

    # Validate
    if not os.path.isfile(args.audio):
        print(f"ERROR: Audio file not found: {args.audio}")
        sys.exit(1)

    size_mb = os.path.getsize(args.audio) / (1024 * 1024)
    if size_mb > 20:
        print(f"ERROR: File too large ({size_mb:.1f} MB, max 20 MB)")
        sys.exit(1)

    if len(args.voice_id) < 8:
        print(f"ERROR: voice_id must be at least 8 characters, got {len(args.voice_id)}")
        sys.exit(1)

    if not args.voice_id[0].isalpha():
        print("ERROR: voice_id must start with a letter")
        sys.exit(1)

    api_key = load_api_key()
    if not api_key:
        print("ERROR: MINIMAX_API_KEY not set in .env or environment")
        sys.exit(1)

    print(f"\n=== MiniMax Voice Clone ===")
    print(f"  Audio:    {args.audio}")
    print(f"  Voice ID: {args.voice_id}")
    print(f"  Model:    {args.model}")
    print()

    t0 = time.time()

    # Step 1: Upload
    print("[1/2] Uploading audio...")
    file_id = upload_audio(api_key, args.base_url, args.audio)

    # Step 2: Clone
    print("[2/2] Cloning voice...")
    result = clone_voice(
        api_key, args.base_url, file_id, args.voice_id, args.text, args.model,
        noise_reduction=args.noise_reduction,
        volume_normalization=args.volume_norm,
    )

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed:.1f}s ===")
    print(f"\nTo use this voice, set in config.yaml:")
    print(f"  voice.tts.minimax_voice_id: {args.voice_id}")
    print(f"\nNote: Cloned voice expires in 7 days if not used in any T2A call.")


if __name__ == "__main__":
    main()
