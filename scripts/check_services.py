#!/usr/bin/env python3
"""Pre-flight service connectivity check for askme.

Usage::

    python scripts/check_services.py          # check all services
    python scripts/check_services.py --json   # machine-readable output

Exit codes:
    0  All configured services reachable
    1  One or more configured services unreachable
    2  No services configured (likely missing .env)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


# ── Load .env if present ────────────────────────────────────────────────────

def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — no external deps required."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# ── HTTP probe ───────────────────────────────────────────────────────────────

def _probe(url: str, timeout: float = 3.0) -> tuple[bool, str]:
    """Return (reachable, detail_message)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status < 500:
                return True, f"HTTP {resp.status}"
            return False, f"HTTP {resp.status}"
    except urllib.error.HTTPError as exc:
        if exc.code < 500:
            return True, f"HTTP {exc.code}"
        return False, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return False, str(exc.reason)
    except OSError as exc:
        return False, str(exc)


# ── Service definitions ──────────────────────────────────────────────────────

SERVICES = [
    {
        "name": "nav-gateway",
        "env": "NAV_GATEWAY_URL",
        "health": "/api/v1/health",
        "skills": ["navigate", "mapping", "follow_person", "nav_cancel", "nav_query"],
        "required": False,
    },
    {
        "name": "dog-control",
        "env": "DOG_CONTROL_SERVICE_URL",
        "health": "/api/v1/health",
        "skills": ["dog_control"],
        "required": False,
    },
    {
        "name": "dog-safety",
        "env": "DOG_SAFETY_SERVICE_URL",
        "health": "/api/v1/health",
        "skills": ["E-STOP propagation"],
        "required": False,
    },
    {
        "name": "llm-relay",
        "env": "LLM_BASE_URL",
        "health": "/models",
        "skills": ["all LLM calls"],
        "required": True,
    },
]


# ── Check runner ─────────────────────────────────────────────────────────────

def check_all() -> list[dict]:
    results = []
    for svc in SERVICES:
        url = os.environ.get(svc["env"], "").rstrip("/")
        if not url:
            results.append({
                "name": svc["name"],
                "env": svc["env"],
                "configured": False,
                "reachable": False,
                "url": "",
                "detail": "env var not set",
                "skills": svc["skills"],
                "required": svc["required"],
            })
            continue
        health_url = url + svc["health"]
        reachable, detail = _probe(health_url)
        results.append({
            "name": svc["name"],
            "env": svc["env"],
            "configured": True,
            "reachable": reachable,
            "url": url,
            "detail": detail,
            "skills": svc["skills"],
            "required": svc["required"],
        })
    return results


# ── Output formatters ────────────────────────────────────────────────────────

_GREEN = "\033[32m"
_RED   = "\033[31m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"
_BOLD  = "\033[1m"

def _c(text: str, code: str) -> str:
    if sys.stdout.isatty():
        return f"{code}{text}{_RESET}"
    return text


def print_table(results: list[dict]) -> None:
    print()
    print(_c("  askme — Pre-flight Service Check", _BOLD))
    print("  " + "─" * 56)
    print(f"  {'Service':<20} {'Status':<14} {'URL / Note'}")
    print("  " + "─" * 56)

    for r in results:
        if not r["configured"]:
            status = _c("NOT CONFIGURED", _YELLOW)
            note = f"set {r['env']} in .env — skills: {', '.join(r['skills'][:2])}"
            if r["required"]:
                status = _c("MISSING (required)", _RED)
        elif r["reachable"]:
            status = _c("OK", _GREEN) + f"  ({r['detail']})"
            note = r["url"]
        else:
            status = _c("UNREACHABLE", _RED) + f"  {r['detail']}"
            note = r["url"]

        print(f"  {r['name']:<20} {status:<24} {note}")

    print("  " + "─" * 56)


def summarise(results: list[dict]) -> tuple[int, str]:
    configured = [r for r in results if r["configured"]]
    unreachable = [r for r in results if r["configured"] and not r["reachable"]]
    missing_required = [r for r in results if r["required"] and not r["configured"]]

    if missing_required:
        names = ", ".join(r["name"] for r in missing_required)
        return 1, _c(f"  FAIL — required service(s) not configured: {names}", _RED)

    if not configured:
        return 2, _c("  WARN — no services configured (copy .env.example → .env)", _YELLOW)

    if unreachable:
        names = ", ".join(r["name"] for r in unreachable)
        return 1, _c(f"  FAIL — unreachable: {names}", _RED)

    return 0, _c(f"  OK — {len(configured)} service(s) reachable", _GREEN)


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="askme pre-flight service check")
    parser.add_argument("--json", action="store_true", help="output JSON")
    parser.add_argument(
        "--env",
        default=str(Path(__file__).parent.parent / ".env"),
        help="path to .env file (default: project root .env)",
    )
    args = parser.parse_args()

    _load_dotenv(Path(args.env))
    results = check_all()

    if args.json:
        exit_code, _ = summarise(results)
        print(json.dumps({"exit_code": exit_code, "services": results}, indent=2))
        sys.exit(exit_code)

    print_table(results)
    exit_code, summary = summarise(results)
    print(summary)
    print()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
