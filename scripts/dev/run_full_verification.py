#!/usr/bin/env python3
"""Run full verification for Askme: lint, core tests, dialogue smoke, voice health.

Usage:
    python scripts/dev/run_full_verification.py
    python scripts/dev/run_full_verification.py --skip-lint
    python scripts/dev/run_full_verification.py --skip-tests
    python scripts/dev/run_full_verification.py --skip-smoke
    python scripts/dev/run_full_verification.py --skip-voice
    python scripts/dev/run_full_verification.py --report report.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "artifacts" / "verification"


def _timestamp() -> str:
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _run(
    description: str,
    args: list[str],
    *,
    timeout_s: int = 300,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    print(f"\n{'=' * 70}")
    print(f"[{_timestamp()}] {description}")
    print(f"  $ {' '.join(args)}")
    print(f"{'=' * 70}")

    start = time.time()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    try:
        result = subprocess.run(
            args,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=merged_env,
        )
        elapsed = time.time() - start
        passed = result.returncode == 0

        # Print stdout/stderr (truncate to last 200 lines for readability)
        output_lines = result.stdout.splitlines()
        if len(output_lines) > 200:
            print(f"  (stdout truncated: {len(output_lines)} lines, showing last 200)")
            output_lines = output_lines[-200:]
        if output_lines:
            print("\n".join(output_lines))
        if result.stderr:
            err_lines = result.stderr.splitlines()
            if len(err_lines) > 100:
                print(f"  (stderr truncated: {len(err_lines)} lines, showing last 100)")
                err_lines = err_lines[-100:]
            print("\n".join(err_lines))

        status = "PASSED" if passed else "FAILED"
        print(f"\n  Result: {status} (exit={result.returncode}, elapsed={elapsed:.1f}s)")

        return {
            "description": description,
            "command": " ".join(args),
            "passed": passed,
            "returncode": result.returncode,
            "elapsed_s": round(elapsed, 1),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"\n  Result: TIMEOUT (>{timeout_s}s)")
        return {
            "description": description,
            "command": " ".join(args),
            "passed": False,
            "returncode": -1,
            "elapsed_s": round(elapsed, 1),
            "error": f"timed out after {timeout_s}s",
            "stdout": "",
            "stderr": "",
        }
    except FileNotFoundError as exc:
        print(f"\n  Result: SKIPPED (command not found: {exc})")
        return {
            "description": description,
            "command": " ".join(args),
            "passed": False,
            "returncode": -2,
            "elapsed_s": 0.0,
            "error": f"command not found: {exc}",
            "stdout": "",
            "stderr": "",
        }


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])
    total = len(results)
    all_passed = failed == 0

    print(f"\n{'=' * 70}")
    print(f"  VERIFICATION {'PASSED' if all_passed else 'FAILED'}")
    print(f"  {passed}/{total} checks passed, {failed} failed")
    print(f"{'=' * 70}")

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        elapsed = r.get("elapsed_s", 0)
        print(f"  [{status}] ({elapsed}s) {r['description']}")

    return {
        "all_passed": all_passed,
        "total": total,
        "passed": passed,
        "failed": failed,
        "timestamp": _timestamp(),
        "checks": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-lint", action="store_true", help="Skip ruff check")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest core tests")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip dialogue smoke")
    parser.add_argument("--skip-voice", action="store_true", help="Skip voice health check")
    parser.add_argument("--report", default=None, help="Write JSON report to this path")
    args = parser.parse_args()

    results: list[dict[str, Any]] = []

    # Step 1: ruff check
    if not args.skip_lint:
        results.append(
            _run(
                "Ruff lint check",
                [sys.executable, "-m", "ruff", "check", "."],
                timeout_s=120,
            )
        )
    else:
        print("\n[Skipped] Ruff lint check (--skip-lint)")

    # Step 2: pytest core tests
    if not args.skip_tests:
        test_env = {"ASKME_LAB_UNSAFE_TOOLS": "true"}
        results.append(
            _run(
                "Pytest core tests (default fast shard)",
                [
                    sys.executable, "-m", "pytest", "tests/",
                    "-q", "--tb=short", "-x", "--maxfail=5",
                    "-W", "ignore::DeprecationWarning",
                ],
                timeout_s=600,
                env=test_env,
            )
        )
    else:
        print("\n[Skipped] Pytest core tests (--skip-tests)")

    # Step 3: dialogue smoke
    if not args.skip_smoke:
        results.append(
            _run(
                "Dialogue smoke (--fake-llm)",
                [
                    sys.executable, "-m", "askme.cli",
                    "runtime", "dialogue-smoke",
                    "--fake-llm",
                ],
                timeout_s=300,
            )
        )
    else:
        print("\n[Skipped] Dialogue smoke (--skip-smoke)")

    # Step 4: voice health
    if not args.skip_voice:
        results.append(
            _run(
                "Voice health check",
                [
                    sys.executable, "-m", "askme.cli",
                    "runtime", "voice-health",
                ],
                timeout_s=120,
            )
        )
    else:
        print("\n[Skipped] Voice health check (--skip-voice)")

    # Summary
    summary = _summary(results)

    # Write report
    report_path = args.report
    if not report_path:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = str(
            REPORT_DIR / f"verification-{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nReport written to: {report_path}")

    sys.exit(0 if summary["all_passed"] else 1)


if __name__ == "__main__":
    main()
