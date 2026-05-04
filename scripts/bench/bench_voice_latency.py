#!/usr/bin/env python3
"""Measure LLM-to-TTS latency for a single voice turn.

This script is intentionally application-level only: it does not touch kernel
modules, reboot, or change audio devices.  On Sunrise, run with ``--playback``
to include the MCP01 USB direct helper timing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from askme.config import get_config  # noqa: E402
from askme.llm.client import LLMClient  # noqa: E402
from askme.pipeline.stream_processor import _ThinkFilter  # noqa: E402
from askme.voice.stream_splitter import StreamSplitter  # noqa: E402
from askme.voice.tts import TTSEngine  # noqa: E402


def _chunk_content(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk
    if not getattr(chunk, "choices", None):
        return ""
    delta = chunk.choices[0].delta
    return getattr(delta, "content", "") or ""


def _ms_since(started_at: float) -> int:
    return round((time.perf_counter() - started_at) * 1000)


def _build_messages(text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a low-latency voice assistant. Reply in one short "
                "spoken sentence. Do not include markdown."
            ),
        },
        {"role": "user", "content": text},
    ]


async def run_once(args: argparse.Namespace) -> dict[str, Any]:
    cfg = get_config()
    client = LLMClient()
    tts: TTSEngine | None = None

    metrics: dict[str, Any] = {
        "model": args.model or client.model,
        "playback": bool(args.playback),
        "first_chunk_ms": None,
        "first_content_ms": None,
        "first_clean_content_ms": None,
        "first_sentence_ms": None,
        "tts_enqueue_ms": None,
        "first_play_start_ms": None,
        "first_play_done_ms": None,
        "first_speech_play_start_ms": None,
        "first_speech_play_done_ms": None,
        "usb_play_events": [],
        "total_ms": None,
        "sentences": [],
        "reply_preview": "",
        "error": None,
    }

    if args.playback:
        tts = TTSEngine(cfg.get("voice", {}).get("tts", {}))
        original_usb_play = tts._play_chunk_usb_direct
        original_usb_play_speech = tts._play_chunk_usb_direct_speech
        original_usb_play_with_preroll = tts._play_chunk_usb_direct_with_preroll
        original_usb_play_warming = tts._play_chunk_usb_direct_warming

        def make_timed_usb_play(kind, func):
            def timed_usb_play(chunk):
                start_ms = _ms_since(started_at)
                if metrics["first_play_start_ms"] is None:
                    metrics["first_play_start_ms"] = start_ms
                if kind != "prewarm" and metrics["first_speech_play_start_ms"] is None:
                    metrics["first_speech_play_start_ms"] = start_ms

                event = {
                    "kind": kind,
                    "start_ms": start_ms,
                    "samples": int(len(chunk)),
                    "done_ms": None,
                    "ok": None,
                }
                try:
                    ok = func(chunk)
                    event["ok"] = bool(ok)
                    return ok
                finally:
                    done_ms = _ms_since(started_at)
                    event["done_ms"] = done_ms
                    if metrics["first_play_done_ms"] is None:
                        metrics["first_play_done_ms"] = done_ms
                    if kind != "prewarm" and metrics["first_speech_play_done_ms"] is None:
                        metrics["first_speech_play_done_ms"] = done_ms
                    metrics["usb_play_events"].append(event)
            return timed_usb_play

        tts._play_chunk_usb_direct = make_timed_usb_play("plain", original_usb_play)  # type: ignore[method-assign]
        tts._play_chunk_usb_direct_speech = make_timed_usb_play(  # type: ignore[method-assign]
            "speech",
            original_usb_play_speech,
        )
        tts._play_chunk_usb_direct_with_preroll = make_timed_usb_play(  # type: ignore[method-assign]
            "feedback",
            original_usb_play_with_preroll,
        )
        tts._play_chunk_usb_direct_warming = make_timed_usb_play(  # type: ignore[method-assign]
            "prewarm",
            original_usb_play_warming,
        )

    started_at = time.perf_counter()
    splitter = StreamSplitter()
    think_filter = _ThinkFilter()
    reply_parts: list[str] = []
    if tts is not None:
        tts.start_playback()

    try:
        stream = client.chat_stream(
            _build_messages(args.text),
            model=args.model,
            temperature=args.temperature,
            thinking=False,
        )
        async for chunk in stream:
            if metrics["first_chunk_ms"] is None:
                metrics["first_chunk_ms"] = _ms_since(started_at)
            content = _chunk_content(chunk)
            if not content:
                continue
            if metrics["first_content_ms"] is None:
                metrics["first_content_ms"] = _ms_since(started_at)

            clean = think_filter.feed(content)
            if not clean:
                continue
            if metrics["first_clean_content_ms"] is None:
                metrics["first_clean_content_ms"] = _ms_since(started_at)

            reply_parts.append(clean)
            for sentence in splitter.feed(clean):
                if metrics["first_sentence_ms"] is None:
                    metrics["first_sentence_ms"] = _ms_since(started_at)
                metrics["sentences"].append(sentence)
                if tts is not None:
                    tts.speak(sentence)
                    if metrics["tts_enqueue_ms"] is None:
                        metrics["tts_enqueue_ms"] = _ms_since(started_at)
                    tts.start_playback()

        clean_tail = think_filter.flush()
        if clean_tail:
            if metrics["first_clean_content_ms"] is None:
                metrics["first_clean_content_ms"] = _ms_since(started_at)
            reply_parts.append(clean_tail)
            for sentence in splitter.feed(clean_tail):
                if metrics["first_sentence_ms"] is None:
                    metrics["first_sentence_ms"] = _ms_since(started_at)
                metrics["sentences"].append(sentence)
                if tts is not None:
                    tts.speak(sentence)
                    if metrics["tts_enqueue_ms"] is None:
                        metrics["tts_enqueue_ms"] = _ms_since(started_at)
                    tts.start_playback()

        leftover = splitter.flush()
        if leftover:
            if metrics["first_sentence_ms"] is None:
                metrics["first_sentence_ms"] = _ms_since(started_at)
            metrics["sentences"].append(leftover)
            if tts is not None:
                tts.speak(leftover)
                if metrics["tts_enqueue_ms"] is None:
                    metrics["tts_enqueue_ms"] = _ms_since(started_at)
                tts.start_playback()

        if tts is not None:
            await asyncio.to_thread(tts.wait_done, args.timeout)
            tts.stop_playback()
    except Exception as exc:
        metrics["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if tts is not None:
            tts.shutdown()

    metrics["total_ms"] = _ms_since(started_at)
    metrics["reply_preview"] = "".join(reply_parts).strip()[:200]
    return metrics


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="Please reply with exactly: latency test ok.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--playback", action="store_true", help="include TTS and real audio output")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out", default=None, help="optional JSON output path")
    args = parser.parse_args()

    result = await run_once(args)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
