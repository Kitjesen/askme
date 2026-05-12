"""Forward device JSON/JSONL events into Askme field operations.

Example:
    python scripts/runtime/bridges/field_ingest_bridge.py camera-events.jsonl --watch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from askme.pipeline.field_ingest_bridge import (  # noqa: E402
    run_field_ingest_bridge_once,
    watch_field_ingest_bridge,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bridge camera/sensor/robot JSON or JSONL events to /api/field/ingest.",
    )
    parser.add_argument("source", help="JSON object/array or append-only JSONL/NDJSON file")
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8765",
        help="Askme runtime base URL (default: http://127.0.0.1:8765)",
    )
    parser.add_argument("--state-path", default="", help="Offset/fingerprint state file")
    parser.add_argument("--watch", action="store_true", help="Keep polling for new events")
    parser.add_argument("--interval", type=float, default=1.0, help="Watch polling interval seconds")
    parser.add_argument("--dry-run", action="store_true", help="Normalize only; do not POST")
    parser.add_argument("--limit", type=int, default=0, help="Maximum events per pass; 0 means all")
    parser.add_argument("--timeout", type=float, default=8.0, help="HTTP timeout seconds")
    parser.add_argument("--json", action="store_true", help="Print JSON for one-shot mode")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state_path = args.state_path or None
    if args.watch:
        watch_field_ingest_bridge(
            source=args.source,
            server=args.server,
            state_path=state_path,
            interval_s=args.interval,
            dry_run=args.dry_run,
            limit=args.limit,
            timeout_s=args.timeout,
        )
        return 0

    payload = run_field_ingest_bridge_once(
        source=args.source,
        server=args.server,
        state_path=state_path,
        dry_run=args.dry_run,
        limit=args.limit,
        timeout_s=args.timeout,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            "field-ingest-bridge: "
            f"{payload['status']} count={payload['count']} "
            f"failed={payload['failed']} dry_run={payload['dry_run']}"
        )
    return 1 if payload.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
