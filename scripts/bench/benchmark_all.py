#!/usr/bin/env python3
"""Askme system benchmark — measure every latency that matters.

Run on S100P:
    cd ~/askme && source .venv/bin/activate
    python scripts/bench/benchmark_all.py

Outputs a structured report and saves JSON to data/benchmarks/.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── 1. LLM ───────────────────────────────────────────


async def bench_llm() -> dict:
    """LLM TTFT + total latency (3 runs, streaming)."""
    from askme.llm.client import LLMClient

    client = LLMClient()
    messages = [
        {"role": "system", "content": "你是一个机器人助手。回答要简短。"},
        {"role": "user", "content": "你好"},
    ]

    results = []
    for i in range(3):
        t0 = time.perf_counter()
        first_token_time = None
        tokens = []

        try:
            async for chunk in client.chat_stream(messages):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                text = ""
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    text = getattr(delta, "content", "") or ""
                elif isinstance(chunk, str):
                    text = chunk
                if text:
                    tokens.append(text)

            t1 = time.perf_counter()
            reply = "".join(tokens)
            ttft = (first_token_time - t0) * 1000 if first_token_time else None
            total = (t1 - t0) * 1000
            results.append({"ttft_ms": round(ttft, 1) if ttft else None,
                            "total_ms": round(total, 1), "chars": len(reply)})
            print(f"  LLM [{i+1}]: TTFT={ttft:.0f}ms total={total:.0f}ms chars={len(reply)}")
        except Exception as e:
            results.append({"error": str(e)})
            print(f"  LLM [{i+1}]: ERROR {e}")

    return {"llm": results, "model": getattr(client, "model", "unknown")}


# ── 2. Pulse DDS ─────────────────────────────────────


async def bench_pulse() -> dict:
    """Pulse topic subscription latency."""
    results = {"available": False}
    try:
        from askme.robot.pulse import Pulse, _RCLPY_AVAILABLE
        if not _RCLPY_AVAILABLE:
            print("  Pulse: rclpy not available (skip)")
            return {"pulse": results}

        pulse = Pulse({"enabled": True, "node_name": "askme_bench"})
        await pulse.start()
        if not pulse.connected:
            print("  Pulse: not connected (skip)")
            await pulse.stop()
            return {"pulse": results}

        results["available"] = True
        latencies = []
        received = asyncio.Event()

        def on_det(topic, data):
            ts = data.get("_ts", 0)
            if ts:
                latencies.append((time.time() - ts) * 1000)
            if len(latencies) >= 20:
                received.set()

        pulse.on("/thunder/detections", on_det)
        print("  Pulse: waiting for 20 msgs...")
        try:
            await asyncio.wait_for(received.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            print(f"  Pulse: timeout, got {len(latencies)} msgs")

        await pulse.stop()
        if latencies:
            latencies.sort()
            results.update({
                "msg_count": len(latencies),
                "avg_ms": round(sum(latencies) / len(latencies), 3),
                "min_ms": round(latencies[0], 3),
                "max_ms": round(latencies[-1], 3),
                "p50_ms": round(latencies[len(latencies) // 2], 3),
            })
            print(f"  Pulse: {len(latencies)} msgs avg={results['avg_ms']}ms "
                  f"min={results['min_ms']}ms p50={results['p50_ms']}ms")
    except Exception as e:
        results["error"] = str(e)
        print(f"  Pulse: ERROR {e}")

    return {"pulse": results}


# ── 3. ASR ────────────────────────────────────────────


def bench_asr() -> dict:
    """ASR init time + recognition speed."""
    results = {}
    try:
        from askme.config import get_config
        voice_cfg = get_config().get("voice", {})

        t0 = time.perf_counter()
        from askme.voice.asr import ASREngine
        asr = ASREngine(voice_cfg.get("asr", {}))
        results["init_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        print(f"  ASR init: {results['init_ms']}ms")

        # Feed 1s silence and measure
        samples = [0] * 16000
        t0 = time.perf_counter()
        stream = asr.create_stream()
        stream.accept_waveform(16000, samples)
        while asr.recognizer.is_ready(stream):
            asr.recognizer.decode_stream(stream)
        raw = asr.recognizer.get_result(stream)
        text = raw.text.strip() if hasattr(raw, "text") else str(raw).strip()
        results["recognize_1s_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        results["result"] = text
        print(f"  ASR recognize: {results['recognize_1s_ms']}ms → '{text}'")
    except Exception as e:
        results["error"] = str(e)
        print(f"  ASR: ERROR {e}")

    return {"asr": results}


# ── 4. TTS (end-to-end: text → audio bytes ready) ───


def bench_tts() -> dict:
    """TTS end-to-end: queue text → wait for audio bytes in output queue."""
    results = {}
    try:
        from askme.config import get_config
        voice_cfg = get_config().get("voice", {})
        from askme.voice.tts import TTSEngine

        tts = TTSEngine(voice_cfg.get("tts", {}))
        test_text = "你好，我是巡检机器人。"
        backend = getattr(tts, "_backend", "unknown")

        # Directly call _generate_audio (blocking — does the actual synthesis)
        gen = tts._get_generation()
        t0 = time.perf_counter()
        tts._generate_audio(test_text, gen)
        total = (time.perf_counter() - t0) * 1000

        results["total_ms"] = round(total, 1)
        results["backend"] = backend
        results["text"] = test_text
        print(f"  TTS: {total:.0f}ms (backend={backend})")

        # Second run (warm)
        gen2 = tts._get_generation()
        t0 = time.perf_counter()
        tts._generate_audio(test_text, gen2)
        warm = (time.perf_counter() - t0) * 1000
        results["warm_ms"] = round(warm, 1)
        print(f"  TTS warm: {warm:.0f}ms")

    except Exception as e:
        results["error"] = str(e)
        print(f"  TTS: ERROR {e}")

    return {"tts": results}


# ── 5. Perception ────────────────────────────────────


def bench_perception() -> dict:
    """Perception: file read latency + IoU speed + ChangeDetector cycle."""
    results = {}
    try:
        from askme.constants import DAEMON_DETECTIONS_PATH

        # File read latency
        if os.path.exists(DAEMON_DETECTIONS_PATH):
            times = []
            for _ in range(100):
                t0 = time.perf_counter()
                with open(DAEMON_DETECTIONS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                times.append((time.perf_counter() - t0) * 1000)
            times.sort()
            results["file_read_avg_ms"] = round(sum(times) / len(times), 3)
            results["file_read_p50_ms"] = round(times[50], 3)
            results["file_read_p99_ms"] = round(times[99], 3)
            results["detections"] = len(data.get("detections", data.get("results", [])))
            print(f"  File read: avg={results['file_read_avg_ms']}ms "
                  f"p50={results['file_read_p50_ms']}ms p99={results['file_read_p99_ms']}ms")
        else:
            print(f"  File: {DAEMON_DETECTIONS_PATH} not found (daemon not running)")

        # IoU computation
        from askme.perception.change_detector import compute_iou
        t0 = time.perf_counter()
        for _ in range(10000):
            compute_iou((10, 10, 100, 100), (50, 50, 150, 150))
        iou_us = (time.perf_counter() - t0) / 10000 * 1_000_000
        results["iou_us_per_call"] = round(iou_us, 2)
        print(f"  IoU: {iou_us:.2f}μs/call ({int(1_000_000/iou_us)}/s)")

        # ChangeDetector single cycle
        from askme.perception.change_detector import ChangeDetector
        cd = ChangeDetector({"proactive": {"change_detector": {}}})
        t0 = time.perf_counter()
        cd._read_daemon()
        read_ms = (time.perf_counter() - t0) * 1000
        results["detector_read_ms"] = round(read_ms, 3)
        print(f"  ChangeDetector read: {read_ms:.3f}ms")

    except Exception as e:
        results["error"] = str(e)
        print(f"  Perception: ERROR {e}")

    return {"perception": results}


# ── 6. Memory ────────────────────────────────────────


def bench_memory() -> dict:
    """Memory: conversation add speed + history size."""
    results = {}
    try:
        import platform
        results["python"] = platform.python_version()
        results["arch"] = platform.machine()

        from askme.llm.conversation import ConversationManager

        conv = ConversationManager(max_history=40)
        t0 = time.perf_counter()
        for i in range(40):
            conv.add_user_message(f"message {i}")
            conv.add_assistant_message(f"reply {i}")
        add_ms = (time.perf_counter() - t0) * 1000
        results["add_80_msgs_ms"] = round(add_ms, 2)
        results["history_len"] = len(conv.history)
        print(f"  Memory: 80 msgs in {add_ms:.2f}ms, history={len(conv.history)}")

        # Get messages speed
        t0 = time.perf_counter()
        for _ in range(100):
            msgs = conv.get_messages(system_prompt="test")
        get_ms = (time.perf_counter() - t0) / 100 * 1000
        results["get_messages_ms"] = round(get_ms, 3)
        print(f"  Memory get_messages: {get_ms:.3f}ms (avg 100 calls)")

    except Exception as e:
        results["error"] = str(e)
        print(f"  Memory: ERROR {e}")

    return {"memory": results}


# ── 7. Full voice pipeline estimate ─────────────────


def bench_voice_estimate(all_results: dict) -> dict:
    """Estimate full voice pipeline latency from individual measurements."""
    est = {}
    try:
        # ASR
        asr = all_results.get("asr", {})
        asr_ms = asr.get("recognize_1s_ms", 0)

        # LLM
        llm_runs = all_results.get("llm", [])
        ttfts = [r["ttft_ms"] for r in llm_runs if r.get("ttft_ms")]
        llm_ttft = min(ttfts) if ttfts else 0

        # TTS
        tts = all_results.get("tts", {})
        tts_ttfb = tts.get("ttfb_ms", 0)

        # VAD overhead estimate
        vad_ms = 200  # typical VAD hold time

        total = vad_ms + asr_ms + llm_ttft + (tts_ttfb or 0)
        est = {
            "vad_hold_ms": vad_ms,
            "asr_ms": asr_ms,
            "llm_ttft_ms": round(llm_ttft, 1),
            "tts_ttfb_ms": round(tts_ttfb, 1) if tts_ttfb else "N/A",
            "estimated_total_ms": round(total, 1),
        }
        print(f"\n  Voice pipeline estimate:")
        print(f"    VAD hold:   ~{vad_ms}ms")
        print(f"    ASR:         {asr_ms}ms")
        print(f"    LLM TTFT:    {llm_ttft:.0f}ms  ← bottleneck")
        print(f"    TTS TTFB:    {tts_ttfb or 'N/A'}ms")
        print(f"    ─────────────────────")
        print(f"    Total:      ~{total:.0f}ms to first spoken word")

    except Exception as e:
        est["error"] = str(e)

    return {"voice_pipeline_estimate": est}


# ── Main ─────────────────────────────────────────────


async def main():
    print("=" * 60)
    print("ASKME SYSTEM BENCHMARK")
    print("=" * 60)

    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
    }

    sections = [
        ("1/7", "LLM", lambda: bench_llm()),
        ("2/7", "Pulse DDS", lambda: bench_pulse()),
        ("3/7", "ASR", None),
        ("4/7", "TTS", None),
        ("5/7", "Perception", None),
        ("6/7", "Memory", None),
    ]

    sync_benches = {
        "3/7": bench_asr,
        "4/7": bench_tts,
        "5/7": bench_perception,
        "6/7": bench_memory,
    }

    for label, name, async_fn in sections:
        print(f"\n[{label}] {name}")
        try:
            if async_fn:
                result = await async_fn()
            else:
                result = sync_benches[label]()
            all_results.update(result)
        except Exception as e:
            print(f"  {name}: FATAL {e}")
            all_results[name.lower()] = {"fatal_error": str(e)}

    # Voice pipeline estimate
    print(f"\n[7/7] Voice Pipeline Estimate")
    all_results.update(bench_voice_estimate(all_results))

    # Save
    out_dir = Path("data/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bench_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
