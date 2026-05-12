"""Validate a field-operations site profile before customer deployment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from askme.pipeline.field_site_profile import build_site_profile_report

DEFAULT_PROFILE = Path("deploy/site-profiles/park-demo.yaml")
DEFAULT_OUTPUT = Path("artifacts/field_operations/demo/site-profile-readiness.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Warn when referenced environment variables are not set.",
    )
    args = parser.parse_args(argv)

    report = build_site_profile_report(args.profile, check_env=args.check_env)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report, "report_path": str(args.output)}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
