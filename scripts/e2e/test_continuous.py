"""Continuous test: send multiple queries, play TTS responses.

Runs through a set of test prompts to verify SOUL.md character,
Edge TTS, and VLM integration in sequence.
"""

import asyncio
import io
import re
import sys
import time

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from askme.brain.llm_client import LLMClient
from askme.voice.tts import TTSEngine
from askme.config import get_config

# Test prompts — various scenarios for Thunder
TEST_PROMPTS = [
    "Thunder，报告状态",
    "三号区域温度异常，怎么处理？",
    "今天天气怎么样？",
    "给我讲个笑话",
    "巡检完毕，有什么需要注意的吗？",
]


def _load_soul_seed() -> list[dict[str, str]]:
    """Load SOUL.md and convert to prompt seed."""
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


async def run_continuous():
    cfg = get_config()
    brain = cfg.get("brain", {})
    tts_cfg = cfg.get("voice", {}).get("tts", {})
    prefix = brain.get("user_prefix", "")
    seed = _load_soul_seed() or brain.get("prompt_seed", [])

    client = LLMClient()
    tts = TTSEngine(tts_cfg)

    # Keep conversation history across turns
    history: list[dict] = []

    print("=" * 60)
    print("Thunder Continuous Test")
    print(f"SOUL seed: {len(seed)} turns, Prefix: {prefix[:30]}...")
    print("=" * 60)

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n{'─' * 50}")
        print(f"[{i}/{len(TEST_PROMPTS)}] User: {prompt}")
        print("─" * 50)

        # Build messages
        msgs = [{"role": "system", "content": brain.get("system_prompt", "")}]
        msgs.extend(seed)
        msgs.extend(history)

        tagged = f"{prefix}\n{prompt}"
        msgs.append({"role": "user", "content": tagged})

        # Stream LLM
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
            continue

        dur = time.monotonic() - t0
        ft = f"{ttft:.1f}" if ttft else "N/A"
        print(f"  Thunder: {full}")
        print(f"  TTFT={ft}s  Total={dur:.1f}s  Chars={len(full)}")

        # Update history
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": full})

        # TTS playback
        tts.speak(full)
        tts.start_playback()
        tts.wait_done()
        tts.stop_playback()
        print(f"  TTS done.")

        # Brief pause between turns
        await asyncio.sleep(1.0)

    tts.shutdown()
    print(f"\n{'=' * 60}")
    print("Continuous test complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(run_continuous())
