"""Test anomaly detection logic directly with simulated scene data.

Bypasses VLM entirely — tests whether Haiku can correctly compare
object lists and detect meaningful changes.
"""

import asyncio
import io
import sys
import time

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from askme.brain.llm_client import LLMClient
from askme.voice.tts import TTSEngine
from askme.config import get_config

# Simulated scene sequences (what VLM would return)
SCENARIOS = [
    {
        "name": "正常巡检（无变化）",
        "history": [
            "椅子、桌子、电脑、鼠标、键盘、窗户",
            "椅子、桌子、电脑、鼠标、键盘、窗户",
            "椅子、桌子、电脑、鼠标、键盘、窗户",
        ],
        "current": "椅子、桌子、电脑、鼠标、键盘、窗户",
        "expect": "NORMAL",
    },
    {
        "name": "新增设备",
        "history": [
            "控制柜、管道、阀门、压力表",
            "控制柜、管道、阀门、压力表",
        ],
        "current": "控制柜、管道、阀门、压力表、灭火器、安全帽",
        "expect": "ANOMALY",
    },
    {
        "name": "人员进入区域",
        "history": [
            "货架、叉车、包裹、地面标线",
            "货架、叉车、包裹、地面标线",
        ],
        "current": "货架、叉车、包裹、地面标线、人、安全帽",
        "expect": "ANOMALY",
    },
    {
        "name": "物品消失",
        "history": [
            "服务器机架、网线、UPS电源、监控器",
            "服务器机架、网线、UPS电源、监控器",
        ],
        "current": "服务器机架、网线",
        "expect": "ANOMALY",
    },
    {
        "name": "轻微光线变化（应忽略）",
        "history": [
            "桌子、椅子、台灯、书本、笔筒",
        ],
        "current": "桌子、椅子、台灯、书本、笔筒、阴影",
        "expect": "NORMAL",  # Minor lighting change should be ignored
    },
]

_ANOMALY_PROMPT = """\
Compare these two sets of object detection results from a YOLO monitoring system.

Previous scans:
{history}

Current scan:
{current}

Detection threshold: 关注明显变化（人员增减、物品移动、设备状态）

Output EXACTLY one of:
- ANOMALY|brief Chinese description (max 20 chars) if objects changed significantly
- NORMAL if no significant change

Examples:
- ANOMALY|新增一把椅子
- ANOMALY|检测到新物体
- NORMAL"""


async def main():
    cfg = get_config()
    judge_model = cfg.get("proactive", {}).get(
        "judge_model", "claude-haiku-4-5-20251001"
    )
    tts_cfg = cfg.get("voice", {}).get("tts", {})

    llm = LLMClient()
    tts = TTSEngine(tts_cfg)

    print("=" * 55)
    print(f"Anomaly Detection Test  (model: {judge_model})")
    print("=" * 55)

    passed = 0
    failed = 0

    for i, s in enumerate(SCENARIOS, 1):
        history_text = "\n".join(f"[{j+1}] {h}" for j, h in enumerate(s["history"]))
        prompt = _ANOMALY_PROMPT.format(
            history=history_text,
            current=s["current"],
        )

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
        except Exception as e:
            print(f"[{i}] {s['name']}: ERROR — {e}")
            failed += 1
            continue

        dur = time.monotonic() - t0
        result = result.strip()
        is_anomaly = result.startswith("ANOMALY|")
        got = "ANOMALY" if is_anomaly else "NORMAL"
        ok = got == s["expect"]
        icon = "PASS" if ok else "FAIL"

        anomaly_desc = result[8:].strip() if is_anomaly else ""
        display = f"{got}{'|' + anomaly_desc if anomaly_desc else ''}"

        print(f"\n[{i}] {s['name']}")
        print(f"  Result: {display}  ({dur:.1f}s)  [{icon}]")

        if ok:
            passed += 1
            if is_anomaly:
                tts.speak(f"异常检测：{anomaly_desc}")
                tts.start_playback()
                tts.wait_done()
                tts.stop_playback()
        else:
            failed += 1
            print(f"  Expected: {s['expect']}, Got: {got}")

    print(f"\n{'=' * 55}")
    print(f"Results: {passed}/{passed+failed} passed")
    print("=" * 55)

    tts.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
