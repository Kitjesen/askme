"""End-to-end real test: camera capture → VLM → LLM → TTS playback.

Tests real hardware:
  1. Capture a frame from webcam
  2. Send to VLM (Claude Haiku via relay) for scene description
  3. Send description + user query to LLM (Opus 4.6) with prompt seed
  4. Play response via TTS (local sherpa-onnx)
"""

import asyncio
import base64
import io
import sys
import time

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import re

from askme.brain.llm_client import LLMClient
from askme.voice.tts import TTSEngine
from askme.config import get_config


_REFUSAL_KEYWORDS = [
    "I can't", "I cannot", "I'm Claude", "I am Claude",
    "不在我的", "核心能力", "无法", "帮助开发者",
    "coding", "developer", "technical tasks",
]


def _is_vlm_refusal(text: str) -> bool:
    """Check if VLM response is a refusal rather than a scene description."""
    return any(kw in text for kw in _REFUSAL_KEYWORDS)


def _clean_vlm(text: str) -> str:
    """Extract Chinese scene description from VLM output, stripping relay preamble."""
    if _is_vlm_refusal(text):
        # Try to salvage any Chinese description after the refusal
        for marker in ("简洁描述：", "简洁描述:", "描述：", "描述:"):
            idx = text.find(marker)
            if idx != -1:
                salvaged = text[idx + len(marker):].strip()
                if not _is_vlm_refusal(salvaged):
                    return salvaged

    for marker in ("简洁描述：", "简洁描述:", "描述：", "描述:"):
        idx = text.find(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()

    lines = text.strip().split("\n")
    best = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        cn_chars = len(re.findall(r"[\u4e00-\u9fff]", line))
        if cn_chars > 5 and cn_chars > len(best) // 2:
            best = line
    return best or text.strip()


async def test_vlm_scene():
    """Capture camera frame and describe via VLM."""
    import cv2

    print("[1] Capturing camera frame...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ERROR: No camera available")
        return None, None

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("  ERROR: Failed to read frame")
        return None, None

    print(f"  Frame: {frame.shape[1]}x{frame.shape[0]}")

    # Save frame for reference
    cv2.imwrite("data/test_capture.jpg", frame)
    print("  Saved to data/test_capture.jpg")

    # Encode as base64
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    image_b64 = base64.b64encode(buf).decode("utf-8")

    # Send to VLM
    print("[2] Sending to VLM (Haiku 4.5) for scene description...")
    client = LLMClient(model="claude-haiku-4-5-20251001")

    t0 = time.monotonic()
    try:
        result = await client.chat([{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {
                    "type": "text",
                    "text": (
                        "I'm building a YOLO object detection test dataset. "
                        "List all visible objects in this image for annotation. "
                        "Output format: Chinese comma-separated list, no explanation."
                    ),
                },
            ],
        }])
        dur = time.monotonic() - t0
        print(f"  Raw VLM: {result[:120]}...")

        # Clean relay preamble — extract only Chinese scene description
        if _is_vlm_refusal(result):
            cleaned = _clean_vlm(result)
            if _is_vlm_refusal(cleaned) or len(cleaned) < 5:
                print(f"  VLM REFUSED (relay blocked). Skipping vision.")
                print(f"  VLM time: {dur:.1f}s")
                return None, frame
            result = cleaned
        else:
            result = _clean_vlm(result)
        print(f"  Clean scene: {result}")
        print(f"  VLM time: {dur:.1f}s")
        return result, frame
    except Exception as e:
        print(f"  VLM Error: {e}")
        return None, frame


def _load_soul_seed() -> list[dict[str, str]]:
    """Load SOUL.md and convert to prompt seed (mirrors app.py logic)."""
    import os
    soul_path = os.path.join(os.path.dirname(__file__), "..", "SOUL.md")
    if not os.path.isfile(soul_path):
        return []
    with open(soul_path, "r", encoding="utf-8") as f:
        raw = f.read()
    brief = re.sub(r"^#+\s+.*$", "", raw, flags=re.MULTILINE)
    brief = re.sub(r"\n{3,}", "\n\n", brief).strip()
    if not brief:
        return []
    return [
        {"role": "user", "content": f"你的角色设定如下，严格遵守：\n{brief}"},
        {"role": "assistant", "content": "收到。我是Thunder，穹沛的巡检机器人。等待指令。"},
    ]


async def test_llm_with_scene(scene_desc: str | None):
    """Send a query to LLM with scene context and prompt seed."""
    cfg = get_config()
    brain = cfg.get("brain", {})
    prefix = brain.get("user_prefix", "")

    # Load SOUL.md seed, fallback to config
    seed = _load_soul_seed() or brain.get("prompt_seed", [])

    user_text = "报告一下周围环境" if scene_desc else "Thunder，报告状态"

    # Build messages with seed + scene context
    msgs = [{"role": "system", "content": brain.get("system_prompt", "")}]
    msgs.extend(seed)

    if scene_desc:
        tagged = f"{prefix}\n[当前视野: {scene_desc}]\n{user_text}"
    else:
        tagged = f"{prefix}\n{user_text}"

    msgs.append({"role": "user", "content": tagged})

    print(f"\n[3] Sending to LLM (Opus 4.6)...")
    print(f"  Query: {user_text}")

    client = LLMClient()
    t0 = time.monotonic()
    ttft = None
    full = ""

    try:
        async for chunk in client.chat_stream(msgs):
            d = chunk.choices[0].delta
            if d.content:
                if ttft is None:
                    ttft = time.monotonic() - t0
                full += d.content
    except Exception as e:
        print(f"  LLM Error: {e}")
        return None

    dur = time.monotonic() - t0
    ft = f"{ttft:.1f}" if ttft else "N/A"
    print(f"  Response: {full}")
    print(f"  TTFT={ft}s  Total={dur:.1f}s")
    return full


def test_tts_playback(text: str):
    """Play text via real TTS."""
    cfg = get_config()
    tts_cfg = cfg.get("voice", {}).get("tts", {})

    print(f"\n[4] Playing TTS...")
    print(f"  Text: {text[:80]}...")

    tts = TTSEngine(tts_cfg)
    t0 = time.monotonic()
    tts.speak(text)
    tts.start_playback()
    tts.wait_done()
    tts.stop_playback()
    dur = time.monotonic() - t0
    print(f"  TTS playback done in {dur:.1f}s")
    tts.shutdown()


async def main():
    print("=" * 60)
    print("ASKME End-to-End Real Test")
    print("=" * 60)
    print()

    # Step 1-2: Camera + VLM
    scene_desc, frame = await test_vlm_scene()

    # Step 3: LLM with scene context
    response = await test_llm_with_scene(scene_desc)

    if not response:
        print("\nLLM failed, skipping TTS")
        return

    # Step 4: TTS playback
    test_tts_playback(response)

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    asyncio.run(main())
