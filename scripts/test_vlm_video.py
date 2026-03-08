"""VLM video analysis test.

Downloads and samples frames from an industrial scene video,
sends each frame to VisionBridge VLM pipeline, prints detections.

Usage:
    python scripts/test_vlm_video.py [path/to/video.mp4]
"""

import asyncio
import base64
import logging
import os
import sys
import time

import cv2
import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── config ───────────────────────────────────────────────────────────────────
DEFAULT_VIDEO = os.path.join(os.path.dirname(__file__), "test_data", "industrial.mp4")
SAMPLE_INTERVAL_SEC = 2.0   # one frame every N seconds
MAX_FRAMES = 5               # cap to avoid too many API calls
JPEG_QUALITY = 70


# ── VLM call ──────────────────────────────────────────────────────────────────

_VLM_PROMPT = (
    "I'm building a YOLO object detection test dataset. "
    "List all visible objects in this image for annotation. "
    "Output format: Chinese comma-separated list, no explanation."
)


def call_vlm_sync(vision, frame: np.ndarray) -> str:
    """Encode frame and call VLM synchronously (runs in thread)."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    b64 = base64.b64encode(buf).decode("ascii")

    if vision._vlm_backend == "anthropic":
        resp = vision._vlm_client.messages.create(
            model=vision._vlm_model,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                    },
                    {"type": "text", "text": _VLM_PROMPT},
                ],
            }],
        )
        raw = resp.content[0].text if resp.content else ""
    else:
        resp = vision._vlm_client.chat.completions.create(
            model=vision._vlm_model,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": _VLM_PROMPT},
                ],
            }],
        )
        raw = resp.choices[0].message.content or ""

    from askme.brain.vision_bridge import VisionBridge
    return VisionBridge._clean_vlm_response(raw), raw


async def analyze_video(video_path: str) -> None:
    from askme.brain.vision_bridge import VisionBridge
    from askme.config import get_config

    cfg = get_config()
    brain = cfg.get("brain", {})

    # ── VisionBridge with VLM enabled ─────────────────────────────────────────
    vision = VisionBridge()
    vision._vlm_enabled = True
    vision._vlm_api_key = brain.get("api_key", "")
    vision._vlm_model = "claude-haiku-4-5-20251001"
    vision._vlm_base_url = brain.get("base_url", "https://cursor.scihub.edu.kg/api/v1")

    if not vision._ensure_vlm_client():
        print("[ERROR] VLM client init failed")
        return

    print(f"VLM backend : {vision._vlm_backend}")
    print(f"VLM model   : {vision._vlm_model}")

    # ── open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps

    print(f"Video       : {os.path.basename(video_path)}")
    print(f"Resolution  : {w}x{h}, {fps:.0f}fps, {total}frames, {duration:.1f}s")
    print(f"Sampling    : every {SAMPLE_INTERVAL_SEC}s → up to {MAX_FRAMES} frames")
    print("─" * 60)

    step = max(1, int(fps * SAMPLE_INTERVAL_SEC))
    sample_frames = list(range(0, min(total, step * MAX_FRAMES), step))

    results = []
    for i, frame_idx in enumerate(sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        ts = frame_idx / fps
        print(f"\n[Frame {frame_idx:4d} | t={ts:.1f}s] calling VLM...", flush=True)

        t0 = time.monotonic()
        cleaned, raw = await asyncio.to_thread(call_vlm_sync, vision, frame)
        elapsed = time.monotonic() - t0

        is_refusal = any(m in raw for m in VisionBridge._VLM_REFUSAL_MARKERS)

        if is_refusal:
            print(f"  [REFUSAL] ({elapsed:.1f}s)")
            print(f"  RAW: {raw[:120]}")
        elif cleaned:
            print(f"  [OK] ({elapsed:.1f}s):")
            # pretty-print comma-separated list
            items = [x.strip() for x in cleaned.split(",") if x.strip()]
            for item in items:
                print(f"    - {item}")
        else:
            print(f"  [EMPTY] ({elapsed:.1f}s) | RAW: {raw[:80]}")

        results.append({
            "frame": frame_idx,
            "ts": ts,
            "cleaned": cleaned,
            "refusal": is_refusal,
            "elapsed": elapsed,
        })

    cap.release()

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    ok = sum(1 for r in results if r["cleaned"] and not r["refusal"])
    ref = sum(1 for r in results if r["refusal"])
    avg_t = sum(r["elapsed"] for r in results) / len(results) if results else 0
    print(f"Frames analyzed : {len(results)}")
    print(f"[OK] Detected   : {ok}")
    print(f"[REF] Refused   : {ref}")
    print(f"Avg latency     : {avg_t:.1f}s/frame")
    print("=" * 60)

    # Write full results to UTF-8 file for Chinese inspection
    result_path = os.path.join(os.path.dirname(__file__), "test_data", "vlm_video_results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"[t={r['ts']:.1f}s frame={r['frame']}] {r['cleaned']}\n")
    print(f"Full results (UTF-8): {result_path}")

    # ── save annotated frames ─────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "test_data", "vlm_frames")
    os.makedirs(out_dir, exist_ok=True)

    cap2 = cv2.VideoCapture(video_path)
    for r in results:
        if not r["cleaned"]:
            continue
        cap2.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
        ret, frame = cap2.read()
        if not ret:
            continue

        # Draw detection text on frame
        items = [x.strip() for x in r["cleaned"].split(",") if x.strip()]
        y = 30
        cv2.rectangle(frame, (5, 5), (635, 25 + 22 * len(items)), (0, 0, 0), -1)
        for item in items:
            cv2.putText(frame, item, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 100), 1, cv2.LINE_AA)
            y += 22

        out_path = os.path.join(out_dir, f"frame_{r['frame']:04d}.jpg")
        cv2.imwrite(out_path, frame)

    cap2.release()
    print(f"Annotated frames saved to: {out_dir}")


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    asyncio.run(analyze_video(video_path))
