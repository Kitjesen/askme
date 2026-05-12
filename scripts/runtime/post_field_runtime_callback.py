"""Post a signed field runtime-delivery callback to askme.

This is intended for shadow/lab/robot runtime processes and manual smoke tests.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from askme.runtime.field_callbacks import (
    FIELD_RUNTIME_DELIVERY_STATUSES,
    build_field_runtime_callback_payload,
    build_field_runtime_callback_sequence,
    field_event_id_from_runtime_result,
    post_field_runtime_callback,
    post_field_runtime_callback_sequence,
)


def _parse_extra(values: list[str]) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--extra must be KEY=VALUE, got {value!r}")
        key, raw = value.split("=", 1)
        extra[key.strip()] = raw
    return extra


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("ASKME_BASE_URL", "http://127.0.0.1:8765"))
    parser.add_argument("--event-id", default="")
    parser.add_argument("--status", choices=sorted(FIELD_RUNTIME_DELIVERY_STATUSES))
    parser.add_argument("--result-json", default="", help="RuntimeHandoffService submit_plan_payload JSON")
    parser.add_argument("--secret", default=os.getenv("ASKME_FIELD_RUNTIME_CALLBACK_HMAC_SECRET", ""))
    parser.add_argument("--runtime-callback-id", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--handoff-id", default="")
    parser.add_argument("--dispatch-mode", default="task_handoff")
    parser.add_argument("--robot-motion-policy", default="")
    parser.add_argument("--reason", default="")
    parser.add_argument("--hardware-dispatch", action="store_true")
    parser.add_argument("--extra", action="append", default=[], help="Extra callback field as KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true", help="Print signed JSON without posting")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.result_json:
        with open(args.result_json, encoding="utf-8-sig") as handle:
            result = json.load(handle)
        event_id = args.event_id or field_event_id_from_runtime_result(result)
        payloads = build_field_runtime_callback_sequence(
            result,
            secret=args.secret,
            event_id=event_id,
            reason=args.reason,
        )
        if args.dry_run:
            print(json.dumps(payloads, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        responses = post_field_runtime_callback_sequence(
            base_url=args.base_url,
            event_id=event_id,
            payloads=payloads,
            timeout_s=args.timeout_s,
        )
        print(json.dumps(responses, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if all(item.get("recorded") for item in responses) else 2

    if not args.event_id or not args.status:
        raise SystemExit("--event-id and --status are required unless --result-json is used")
    payload = build_field_runtime_callback_payload(
        status=args.status,
        secret=args.secret,
        runtime_callback_id=args.runtime_callback_id or None,
        run_id=args.run_id,
        handoff_id=args.handoff_id,
        dispatch_mode=args.dispatch_mode,
        robot_motion_policy=args.robot_motion_policy,
        hardware_dispatch=args.hardware_dispatch,
        reason=args.reason,
        extra=_parse_extra(args.extra),
    )
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    result = post_field_runtime_callback(
        base_url=args.base_url,
        event_id=args.event_id,
        payload=payload,
        timeout_s=args.timeout_s,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("recorded") else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
