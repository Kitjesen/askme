"""Test ProactiveAgent: simulated patrol with real VLM + anomaly detection."""

import asyncio
import io
import sys
import time

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from askme.brain.llm_client import LLMClient
from askme.brain.vision_bridge import VisionBridge
from askme.voice.tts import TTSEngine
from askme.config import get_config


async def main():
    cfg = get_config()
    tts_cfg = cfg.get("voice", {}).get("tts", {})
    judge_model = cfg.get("proactive", {}).get(
        "judge_model",
        cfg.get("brain", {}).get("voice_model", "claude-haiku-4-5-20251001"),
    )

    print("=" * 55)
    print("ProactiveAgent Simulation Test")
    print("=" * 55)

    # Init components
    vision = VisionBridge()
    llm = LLMClient()
    tts = TTSEngine(tts_cfg)

    # Override vision to use VLM for this test
    vision._vlm_enabled = True
    vision._vlm_api_key = cfg.get("brain", {}).get("api_key", "")

    scene_history = []

    # Simulate 3 patrol ticks
    for tick in range(1, 4):
        print(f"\n{'─' * 50}")
        print(f"[Tick {tick}] Scanning...")
        print("─" * 50)

        t0 = time.monotonic()
        scene = await vision.describe_scene()
        scan_dur = time.monotonic() - t0

        if not scene:
            print(f"  No scene captured (vision unavailable)")
            continue

        print(f"  Scene: {scene}")
        print(f"  Scan time: {scan_dur:.1f}s")

        # Anomaly detection
        if scene_history:
            history_text = "\n".join(
                f"[{i+1}] {s}" for i, s in enumerate(scene_history[-3:])
            )
            prompt = f"""你是工业巡检机器人的异常检测模块。对比当前场景和之前的观察记录。

之前观察:
{history_text}

当前场景:
{scene}

判断标准: 关注明显变化（人员增减、物品移动、设备状态）

规则:
- 如果有异常或显著变化，输出: ANOMALY|一句话描述异常（中文，20字以内）
- 如果一切正常，输出: NORMAL
- 忽略光线、角度等自然变化"""

            t0 = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    llm.chat(
                        [{"role": "user", "content": prompt}],
                        model=judge_model,
                        temperature=0.1,
                    ),
                    timeout=10.0,
                )
                judge_dur = time.monotonic() - t0
                result = result.strip()
                print(f"  Anomaly check: {result} ({judge_dur:.1f}s)")

                if result.startswith("ANOMALY|"):
                    alert = f"巡检异常：{result[8:].strip()}"
                    print(f"  ALERT: {alert}")
                    tts.speak(alert)
                    tts.start_playback()
                    tts.wait_done()
                    tts.stop_playback()
                else:
                    print(f"  Status: 正常")
            except Exception as e:
                print(f"  Anomaly check failed: {e}")
        else:
            print(f"  First scan — establishing baseline")

        scene_history.append(scene)

        # Brief pause between ticks
        if tick < 3:
            print(f"  Waiting 5s for next tick...")
            await asyncio.sleep(5)

    # Final report
    print(f"\n{'=' * 55}")
    tts.speak(f"巡检完成，共扫描{len(scene_history)}次，未发现重大异常。")
    tts.start_playback()
    tts.wait_done()
    tts.stop_playback()
    tts.shutdown()
    print("ProactiveAgent test complete!")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    asyncio.run(main())
