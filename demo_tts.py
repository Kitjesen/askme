"""Demo: test TTS engine with real audio playback.

Usage:
    python demo_tts.py                    # Quick test with local TTS
    python demo_tts.py --text "你好世界"   # Custom text
    python demo_tts.py --backend edge     # Force edge-tts backend
    python demo_tts.py --stream           # Simulate streaming LLM output
"""

from __future__ import annotations

import argparse
import sys
import time

# Fix Windows console encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo_tts")


def demo_single(text: str, backend: str = "local") -> None:
    """Speak a single sentence."""
    from askme.config import get_section

    tts_config = get_section("voice").get("tts", {})
    if backend:
        tts_config["backend"] = backend

    from askme.voice.tts import TTSEngine
    engine = TTSEngine(tts_config)

    logger.info("Backend: %s", engine._backend)
    logger.info("Speaking: %s", text)

    engine.start_playback()
    t0 = time.perf_counter()
    engine.speak(text)
    engine.wait_done()
    elapsed = time.perf_counter() - t0
    logger.info("Done in %.2fs", elapsed)

    engine.shutdown()


def demo_stream(backend: str = "local") -> None:
    """Simulate streaming LLM output with sentence splitting and immediate TTS."""
    from askme.config import get_section
    from askme.voice.stream_splitter import StreamSplitter

    tts_config = get_section("voice").get("tts", {})
    if backend:
        tts_config["backend"] = backend

    from askme.voice.tts import TTSEngine
    engine = TTSEngine(tts_config)
    splitter = StreamSplitter()

    # Simulated LLM streaming tokens (like DeepSeek output)
    tokens = [
        "你好", "！", "我是", "Thunder", "，",
        "你的", "机器狗", "助手", "。",
        "今天", "天气", "不错", "，",
        "适合", "出去", "散步", "。",
        "需要", "我", "帮你", "巡逻", "一下", "吗", "？",
    ]

    logger.info("Backend: %s", engine._backend)
    logger.info("Simulating streaming LLM output (%d tokens)...", len(tokens))

    engine.start_playback()
    t0 = time.perf_counter()

    for i, token in enumerate(tokens):
        # Simulate token arrival delay (50-100ms like real LLM)
        time.sleep(0.06)
        sentences = splitter.feed(token)
        for sentence in sentences:
            t_sentence = time.perf_counter() - t0
            logger.info("[%.2fs] Speaking: %s", t_sentence, sentence)
            engine.speak(sentence)

    # Flush remainder
    remainder = splitter.flush()
    if remainder:
        t_sentence = time.perf_counter() - t0
        logger.info("[%.2fs] Speaking (flush): %s", t_sentence, remainder)
        engine.speak(remainder)

    engine.wait_done()
    elapsed = time.perf_counter() - t0
    logger.info("Total: %.2fs", elapsed)

    engine.shutdown()


def demo_episodic_memory() -> None:
    """Demo the episodic memory system with simulated patrol events."""
    from askme.brain.episodic_memory import EpisodicMemory, score_importance

    logger.info("=== Episodic Memory Demo ===")

    # Create memory (no LLM, just buffer + importance scoring)
    mem = EpisodicMemory()

    # Simulate a patrol
    events = [
        ("system", "启动巡逻模式", {}),
        ("perception", "YOLO: sofa(0.95), tv(0.91)", {"detections": [{"label": "sofa", "conf": 0.95}, {"label": "tv", "conf": 0.91}]}),
        ("action", "导航到走廊", {"target": "hallway"}),
        ("perception", "YOLO: person(0.88)", {"detections": [{"label": "person", "conf": 0.88}]}),
        ("command", "用户说: 你好，过来", {"source": "voice"}),
        ("action", "移动到用户位置", {"target": [1.2, 0.5]}),
        ("outcome", "到达用户位置", {"distance": 0.3}),
        ("perception", "YOLO: cat(0.82)", {"detections": [{"label": "cat", "conf": 0.82}]}),
        ("error", "电机3号关节过流警告", {"joint": 3, "current": 2.5}),
        ("outcome", "巡逻完成", {"duration_s": 120}),
    ]

    logger.info("Logging %d patrol events...", len(events))
    for etype, desc, ctx in events:
        ep = mem.log(etype, desc, ctx)
        logger.info(
            "  [%s] imp=%.2f stab=%.0fs: %s",
            etype, ep.importance, ep.stability, desc[:50],
        )

    logger.info("\nBuffer: %d events, cumulative importance: %.2f",
                mem.buffer_size, mem.cumulative_importance)
    logger.info("Should reflect: %s", mem.should_reflect())

    # Test retrieval
    logger.info("\n--- Retrieval for '猫' ---")
    results = mem.retrieve("猫", top_k=3)
    for i, ep in enumerate(results):
        logger.info(
            "  #%d [score=%.2f] %s (imp=%.2f, R=%.3f)",
            i + 1, ep.retrieval_score({"猫"}), ep.description[:40],
            ep.importance, ep.retrievability(),
        )

    logger.info("\n--- Retrieval for '电机' ---")
    results = mem.retrieve("电机", top_k=3)
    for i, ep in enumerate(results):
        logger.info(
            "  #%d [score=%.2f] %s (imp=%.2f, R=%.3f)",
            i + 1, ep.retrieval_score({"电机"}), ep.description[:40],
            ep.importance, ep.retrievability(),
        )


def main():
    parser = argparse.ArgumentParser(description="Askme TTS & Memory Demo")
    parser.add_argument("--text", default="你好！我是Thunder，你的机器狗助手。很高兴认识你！")
    parser.add_argument("--backend", choices=["local", "edge"], default=None)
    parser.add_argument("--stream", action="store_true", help="Simulate streaming LLM output")
    parser.add_argument("--memory", action="store_true", help="Demo episodic memory")
    args = parser.parse_args()

    if args.memory:
        demo_episodic_memory()
    elif args.stream:
        demo_stream(backend=args.backend)
    else:
        demo_single(args.text, backend=args.backend)


if __name__ == "__main__":
    main()
