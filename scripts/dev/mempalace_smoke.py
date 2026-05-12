"""Smoke-test askme's optional MemPalace memory backend.

This script is intentionally product-facing: it prints one JSON object that
answers whether local MemPalace memory is actually installed, writable,
retrievable, and used without silent fallback.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from askme.config import project_root
from askme.memory.bridge import MemoryBridge


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "app": {"data_dir": str(args.data_dir)},
        "brain": {},
        "memory": {
            "enabled": True,
            "backend": "mempalace",
            "mempalace_fallback_backend": args.fallback_backend,
            "mempalace_palace_path": str(args.palace),
            "mempalace_wing": args.wing,
            "mempalace_room": args.room,
            "mempalace_n_results": args.n_results,
            "mempalace_min_similarity": args.min_similarity,
            "retrieve_timeout": args.timeout,
            "rag_enforce_expiry": True,
        },
    }


async def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if importlib.util.find_spec("mempalace") is None:
        return {
            "ok": False,
            "code": "mempalace_not_installed",
            "message": "Install with: pip install -e \".[mempalace]\"",
            "installed": False,
            "backend": "mempalace",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    bridge = MemoryBridge(config=build_config(args), data_dir=args.data_dir)
    metadata = {
        "record_id": args.record_id,
        "source": args.source,
        "category": args.category,
        "approval_status": "published",
        "expires_at": args.expires_at,
        "evidence_version": 1,
        "entity_key": args.entity_key,
        "fact_key": args.fact_key,
        "value": args.value or args.fact,
    }

    await bridge.save_fact(args.fact, metadata)
    context = await bridge.retrieve(args.query)
    health = bridge.health()
    evidence = health.get("last_evidence") or []
    ok = bool(evidence) and health.get("last_backend") == "mempalace"
    return {
        "ok": ok,
        "code": "passed" if ok else "no_mempalace_evidence",
        "installed": True,
        "backend": health.get("last_backend"),
        "configured_backend": "mempalace",
        "fallback_reason": health.get("last_fallback_reason", ""),
        "mempalace_ready": health.get("mempalace_ready"),
        "mempalace_path": health.get("mempalace_path"),
        "context": context,
        "evidence": evidence,
        "dropped_evidence": health.get("last_dropped_evidence") or [],
        "answer_policy": health.get("last_answer_policy") or {},
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="Smoke-test askme MemPalace memory.")
    parser.add_argument("--fact", default="A区卫生间在东侧")
    parser.add_argument("--query", default="卫生间在哪")
    parser.add_argument("--record-id", default="smoke_restroom_a")
    parser.add_argument("--source", default="smoke-test")
    parser.add_argument("--category", default="location")
    parser.add_argument("--entity-key", default="site.restroom.a")
    parser.add_argument("--fact-key", default="location")
    parser.add_argument("--value", default="")
    parser.add_argument("--expires-at", default="2099-01-01T00:00:00+00:00")
    parser.add_argument("--palace", type=Path, default=root / ".tmp" / "mempalace-smoke")
    parser.add_argument("--data-dir", type=Path, default=root / ".tmp" / "mempalace-smoke-data")
    parser.add_argument("--wing", default="askme")
    parser.add_argument("--room", default="robot")
    parser.add_argument("--n-results", type=int, default=5)
    parser.add_argument("--min-similarity", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--fallback-backend", choices=["vector", "mem0"], default="vector")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = asyncio.run(run_smoke(args))
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
