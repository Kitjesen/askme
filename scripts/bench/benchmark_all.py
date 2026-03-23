#!/usr/bin/env python3
"""Askme system benchmark — measure every latency that matters.

Run on S100P:
    cd ~/askme && source .venv/bin/activate
    python scripts/bench/benchmark_all.py

Outputs a structured report to stdout and saves JSON to data/benchmarks/.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


async def bench_llm() -> dict:
    """Benchmark LLM TTFT and total latency."""
    from askme.config import get_config as load_config
    from askme.llm.client import LLMClient

    cfg = load_config()
    client = LLMClient()
    messages = [
        {"role": "system", "content": "你是一个机器人助手。回答要简短。"},
        {"role": "user", "content": "你好"},
    ]

    results = []
    for i in range(3):
        t0 = time.perf_counter()
        first_token_time = None

        async def _stream():
            nonlocal first_token_time
            tokens = []
            async for chunk in client.chat_stream(messages):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                # chunk may be ChatCompletionChunk or str
                text = ""
                if hasattr(chunk, "choices") and chunk.choices:
                    delta = chunk.choices[0].delta
                    text = getattr(delta, "content", "") or ""
                elif isinstance(chunk, str):
                    text = chunk
                if text:
                    tokens.append(text)
            return "".join(tokens)

        try:
            reply = await asyncio.wait_for(_stream(), timeout=15.0)
            t1 = time.perf_counter()
            ttft = (first_token_time - t0) * 1000 if first_token_time else None
            total = (t1 - t0) * 1000
            results.append({"ttft_ms": ttft, "total_ms": total, "chars": len(reply)})
            print(f"  LLM run {i+1}: TTFT={ttft:.0f}ms total={total:.0f}ms chars={len(reply)}")
        except Exception as e:
            results.append({"error": str(e)})
            print(f"  LLM run {i+1}: ERROR {e}")

    return {"llm": results, "model": getattr(client, "model", "unknown")}


async def bench_pulse() -> dict:
    """Benchmark Pulse DDS subscription latency."""
    results = {"available": False}

    try:
        from askme.robot.pulse import Pulse, _RCLPY_AVAILABLE
        if not _RCLPY_AVAILABLE:
            print("  Pulse: rclpy not available (skip)")
            return results

        pulse = Pulse({"enabled": True, "node_name": "askme_bench"})
        await pulse.start()

        if not pulse.connected:
            print("  Pulse: not connected (skip)")
            await pulse.stop()
            return results

        results["available"] = True
        latencies = []
        received = asyncio.Event()

        def on_det(topic, data):
            ts = data.get("_ts", 0)
            if ts:
                lat = (time.time() - ts) * 1000
                latencies.append(lat)
            if len(latencies) >= 20:
                received.set()

        pulse.on("/thunder/detections", on_det)
        print("  Pulse: waiting for 20 detection messages...")

        try:
            await asyncio.wait_for(received.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            print(f"  Pulse: timeout, got {len(latencies)} msgs")

        await pulse.stop()

        if latencies:
            results["msg_count"] = len(latencies)
            results["avg_ms"] = sum(latencies) / len(latencies)
            results["min_ms"] = min(latencies)
            results["max_ms"] = max(latencies)
            results["p50_ms"] = sorted(latencies)[len(latencies) // 2]
            print(f"  Pulse: {len(latencies)} msgs, avg={results['avg_ms']:.2f}ms, "
                  f"min={results['min_ms']:.2f}ms, max={results['max_ms']:.2f}ms")
        else:
            print("  Pulse: no messages received")

    except Exception as e:
        results["error"] = str(e)
        print(f"  Pulse: ERROR {e}")

    return {"pulse": results}


def bench_asr() -> dict:
    """Benchmark ASR initialization and recognition speed."""
    results = {}

    try:
        from askme.config import get_config as load_config
        cfg = load_config()
        voice_cfg = cfg.get("voice", {})

        t0 = time.perf_counter()
        from askme.voice.asr import ASREngine
        asr = ASREngine(voice_cfg.get("asr", {}))
        init_ms = (time.perf_counter() - t0) * 1000
        results["init_ms"] = init_ms
        print(f"  ASR init: {init_ms:.0f}ms")

        # Measure stream creation + recognition
        import numpy as np
        silence = np.zeros(16000, dtype=np.int16).tobytes()

        t0 = time.perf_counter()
        stream = asr.create_stream()
        stream.accept_waveform(16000, silence.cast("h", (16000,)) if hasattr(silence, "cast") else
                               [int.from_bytes(silence[i:i+2], "little", signed=True) for i in range(0, len(silence), 2)])
        text = asr.get_result(stream)
        recog_ms = (time.perf_counter() - t0) * 1000
        results["recognize_1s_silence_ms"] = recog_ms
        results["result"] = text
        print(f"  ASR recognize (1s silence): {recog_ms:.0f}ms → '{text}'")

    except Exception as e:
        results["error"] = str(e)
        print(f"  ASR: ERROR {e}")

    return {"asr": results}


async def bench_tts() -> dict:
    """Benchmark TTS synthesis speed."""
    results = {}

    try:
        from askme.config import get_config as load_config
        cfg = load_config()
        voice_cfg = cfg.get("voice", {})

        from askme.voice.tts import TTSEngine
        tts = TTSEngine(voice_cfg.get("tts", {}))

        test_text = "你好，我是巡检机器人。"

        # Measure _generate_audio (blocking, runs in thread internally)
        t0 = time.perf_counter()
        gen_id = 1
        tts._generate_audio(test_text, gen_id)
        t1 = time.perf_counter()
        total = (t1 - t0) * 1000
        results["total_ms"] = total
        results["text"] = test_text
        results["engine"] = getattr(tts, "_active_engine", "unknown")
        print(f"  TTS: total={total:.0f}ms engine={results['engine']}")

    except Exception as e:
        results["error"] = str(e)
        print(f"  TTS: ERROR {e}")

    return {"tts": results}


def bench_perception() -> dict:
    """Benchmark perception pipeline: file read + change detection."""
    results = {}

    try:
        from askme.constants import DAEMON_DETECTIONS_PATH, DAEMON_HEARTBEAT_PATH

        # Measure file read latency
        if os.path.exists(DAEMON_DETECTIONS_PATH):
            times = []
            for _ in range(100):
                t0 = time.perf_counter()
                with open(DAEMON_DETECTIONS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                times.append((time.perf_counter() - t0) * 1000)
            results["file_read_avg_ms"] = sum(times) / len(times)
            results["file_read_min_ms"] = min(times)
            results["file_read_max_ms"] = max(times)
            results["detection_count"] = len(data.get("detections", []))
            print(f"  Perception file read: avg={results['file_read_avg_ms']:.3f}ms "
                  f"min={results['file_read_min_ms']:.3f}ms max={results['file_read_max_ms']:.3f}ms")
        else:
            results["file_read"] = "not available (no daemon)"
            print(f"  Perception: {DAEMON_DETECTIONS_PATH} not found")

        # Measure IoU computation speed
        from askme.perception.change_detector import compute_iou
        t0 = time.perf_counter()
        for _ in range(10000):
            compute_iou((10, 10, 100, 100), (50, 50, 150, 150))
        iou_us = (time.perf_counter() - t0) / 10000 * 1_000_000
        results["iou_per_call_us"] = iou_us
        print(f"  IoU computation: {iou_us:.2f}μs per call")

    except Exception as e:
        results["error"] = str(e)
        print(f"  Perception: ERROR {e}")

    return {"perception": results}


def bench_memory() -> dict:
    """Benchmark memory system latencies."""
    results = {}

    try:
        import platform
        results["python"] = platform.python_version()
        results["platform"] = platform.machine()

        # Measure import time
        t0 = time.perf_counter()
        from askme.llm.conversation import ConversationManager
        import_ms = (time.perf_counter() - t0) * 1000
        results["import_ms"] = import_ms

        # Measure conversation add/compress
        conv = ConversationManager(max_history=40)
        t0 = time.perf_counter()
        for i in range(40):
            conv.add_user_message(f"message {i}")
            conv.add_assistant_message(f"reply {i}")
        add_ms = (time.perf_counter() - t0) * 1000
        results["add_80_msgs_ms"] = add_ms
        results["history_len"] = len(conv.history)
        print(f"  Memory: add 80 msgs={add_ms:.1f}ms, history={len(conv.history)}")

    except Exception as e:
        results["error"] = str(e)
        print(f"  Memory: ERROR {e}")

    return {"memory": results}


async def main():
    print("=" * 60)
    print("ASKME SYSTEM BENCHMARK")
    print("=" * 60)

    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
    }

    async def _wrap_sync(fn):
        return fn()

    benches = [
        ("1/6", "LLM Latency", bench_llm),
        ("2/6", "Pulse DDS", bench_pulse),
        ("3/6", "ASR", lambda: _wrap_sync(bench_asr)),
        ("4/6", "TTS", bench_tts),
        ("5/6", "Perception", lambda: _wrap_sync(bench_perception)),
        ("6/6", "Memory", lambda: _wrap_sync(bench_memory)),
    ]

    for label, name, fn in benches:
        print(f"\n[{label}] {name}")
        try:
            result = fn()
            if asyncio.iscoroutine(result):
                result = await result
            all_results.update(result)
        except Exception as e:
            print(f"  {name}: FATAL {e}")
            all_results[name.lower().replace(" ", "_")] = {"fatal_error": str(e)}

    # Save results
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
