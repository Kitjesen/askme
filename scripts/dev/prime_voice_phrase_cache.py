"""Pre-synthesize deterministic voice replies into the persistent PCM cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from askme.config import get_config
from askme.robot_interaction.routing_policy import DEFAULT_QUICK_REPLIES
from askme.voice.interaction import default_cached_phrases
from askme.voice.output.tts import TTSEngine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.board.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    os.environ["ASKME_CONFIG_PATH"] = str(config_path)
    config = get_config(reload=True)
    tts = TTSEngine(config.get("voice", {}).get("tts", {}))
    results: list[dict[str, object]] = []
    try:
        for cache_key, text in default_cached_phrases(DEFAULT_QUICK_REPLIES).items():
            storage_key = tts._phrase_cache_storage_key(text, cache_key)
            if args.force:
                tts._phrase_cache._memory.pop(storage_key, None)
                path = tts._phrase_cache._path_for(storage_key)
                if path.exists():
                    path.unlink()
            result = tts.prime_cached_phrase(text, cache_key=cache_key)
            result["text"] = text
            results.append(result)
    finally:
        tts.shutdown()

    failed = [item for item in results if not item.get("cached")]
    payload = {
        "status": "passed" if not failed else "failed",
        "config": str(config_path),
        "phrases": results,
        "failed": len(failed),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for item in results:
            print(
                f"{item.get('cache_key')}: "
                f"{'ok' if item.get('cached') else item.get('reason', 'failed')}"
            )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
