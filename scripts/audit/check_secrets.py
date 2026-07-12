#!/usr/bin/env python3
"""Scan for accidentally committed secrets, API keys, and passwords.

Checks performed:
  1. Git staged files           — look for patterns matching known secrets
  2. Config files               — detect hardcoded (non-${ENV_VAR}) secrets
  3. Python source files        — detect ``api_key``, ``password``, etc.
     assigned to literal strings

Returns exit code 0 (pass) or 1 (fail), plus a JSON report on stdout.

Usage:
    python scripts/audit/check_secrets.py                   # scan staged + tracked
    python scripts/audit/check_secrets.py --path src/       # scan specific dir
    python scripts/audit/check_secrets.py --all-files       # scan all tracked files
    python scripts/audit/check_secrets.py --json            # JSON output only
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------
# Patterns — keep these in sync with git-secrets / truffleHog rules
# ------------------------------------------------------------------

# High-confidence patterns (almost certainly a leaked credential)
HIGH_CONFIDENCE_PATTERNS: list[re.Pattern] = [
    # AWS Access Key (AKIA...)
    re.compile(r"(?<![A-Za-z0-9/+\-=])(AKIA[0-9A-Z]{16})(?![A-Za-z0-9/+\-=])"),
    # Generic "-----BEGIN (RSA|EC|OPENSSH|DSA|PGP) PRIVATE KEY-----"
    re.compile(r"-----BEGIN\s+(?:RSA|EC|OPENSSH|DSA|PGP)\s+PRIVATE\s+KEY-----"),
    # GitHub / GitLab tokens
    re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}"),
    re.compile(r"glpat-[A-Za-z0-9\-_]{20,}"),
    # Slack tokens
    re.compile(r"xox[baprs]-[0-9a-zA-Z\-]{10,}"),
    # Generic JWT (eyJ...)
    re.compile(r"eyJ[a-zA-Z0-9\-_]{10,}\.[a-zA-Z0-9\-_]{10,}\.[a-zA-Z0-9\-_]{10,}"),
]

# Medium-confidence patterns (flag for manual review)
MEDIUM_PATTERNS: list[re.Pattern] = [
    # Hardcoded password-like assignments (Python)
    re.compile(r'(?:password|passwd|pwd|secret|api_key|apikey)\s*[=:]\s*["\'][^"\']+["\']'),  # noqa: E501
    # Basic auth in URLs
    re.compile(r"https?://[^:/\s]+:[^@/\s]+@"),
    # Hash-like hex strings (40+ chars) that look like secrets
    re.compile(r"[0-9a-f]{40,}"),
]

# Header/footer lines to skip (copyright, SPDX, license text)
_SKIP_HEADERS: list[re.Pattern] = [
    re.compile(r"^#\s*Copyright"),
    re.compile(r"^#\s*SPDX"),
    re.compile(r"^#\s*License"),
]


# ------------------------------------------------------------------
# Git helpers
# ------------------------------------------------------------------

def _git_staged_files(root: Path) -> list[Path]:
    """Return paths of files staged for commit."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            capture_output=True, text=True, check=False, cwd=root,
        )
        if result.returncode != 0:
            print(f"Warning: git diff failed: {result.stderr.strip()}", file=sys.stderr)
            return []
        return [root / p for p in result.stdout.strip().splitlines() if p]
    except FileNotFoundError:
        print("Warning: git not found, skipping staged-file scan.", file=sys.stderr)
        return []


def _git_tracked_files(root: Path) -> list[Path]:
    """Return paths of all files tracked by git."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True, text=True, check=False, cwd=root,
        )
        if result.returncode != 0:
            return []
        return [root / p for p in result.stdout.strip().splitlines() if p]
    except FileNotFoundError:
        return []


# ------------------------------------------------------------------
# Scanning
# ------------------------------------------------------------------

def _should_skip(path: Path) -> bool:
    """Return True if *path* should not be scanned."""
    # Only text-like files
    ext = path.suffix.lower()
    text_exts = {".py", ".yaml", ".yml", ".toml", ".json", ".md", ".txt",
                 ".cfg", ".conf", ".ini", ".env", ".sh", ".bat", ".ps1",
                 ".html", ".js", ".ts", ".css", ".proto", ".dockerfile",
                 ".service", ".example"}
    if ext not in text_exts:
        return True
    # Skip vendored / generated / binary dirs
    skip_parts = {"node_modules", ".git", "__pycache__", ".venv",
                  ".venv-scientist", ".mypy_cache", ".pytest_cache",
                  ".ruff_cache", ".codex"}
    parts = path.parts
    return any(sp in parts for sp in skip_parts)


def _is_literal_secret(line: str) -> bool:
    """Heuristic: flag non-template assignments of keys to string literals.

    E.g. ``api_key: "sk-abc123"`` is flagged, while ``api_key: ${VAR}``
    and ``api_key = os.environ.get("VAR")`` are not.
    """
    stripped = line.strip()
    # Skip comments and empty
    if not stripped or stripped.startswith("#"):
        return False
    # If a value references ${VAR} or os.environ / os.getenv, it's acceptable
    if "${" in stripped:
        return False
    if "os.environ" in stripped or "os.getenv" in stripped:
        return False
    # Check YAML / Python assignment to security-sensitive keys
    for key in ("api_key", "apikey", "password", "secret", "token", "jwt_secret"):
        if re.search(rf'\b{re.escape(key)}\b\s*[=:]\s*["\']', stripped, re.IGNORECASE):
            return True
    return False


def _scan_file(path: Path, high: list[dict[str, Any]], medium: list[dict[str, Any]]) -> None:
    """Scan a single file for secret patterns."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError) as exc:
        medium.append({
            "path": str(path),
            "line": 0,
            "pattern": "unreadable",
            "detail": str(exc),
        })
        return

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        # Skip copyright / license header lines to reduce noise
        if any(p.match(raw_line) for p in _SKIP_HEADERS):
            continue

        # High-confidence
        for pattern in HIGH_CONFIDENCE_PATTERNS:
            match = pattern.search(raw_line)
            if match:
                high.append({
                    "path": str(path),
                    "line": lineno,
                    "pattern": pattern.pattern[:60],
                    "match": match.group(0)[:40] + ("..." if len(match.group(0)) > 40 else ""),
                })
                # Stop scanning this line once we flag a high-confidence match
                break

        # Medium-confidence
        for pattern in MEDIUM_PATTERNS:
            if pattern.search(raw_line):
                # Avoid double-flagging if already flagged as high
                if any(h["path"] == str(path) and h["line"] == lineno for h in high):
                    break
                medium.append({
                    "path": str(path),
                    "line": lineno,
                    "pattern": pattern.pattern[:60],
                    "match": raw_line.strip()[:80],
                })
                break

    # Additional check: literal string assignments to secret keys
    if path.suffix in {".yaml", ".yml", ".toml", ".env", ".py", ".json"}:
        for lineno, raw_line in enumerate(text.splitlines(), start=1):
            if _is_literal_secret(raw_line):
                key_name = raw_line.split(":")[0].split("=")[0].strip()
                medium.append({
                    "path": str(path),
                    "line": lineno,
                    "pattern": "literal-secret-assignment",
                    "match": f"{key_name} assigned to a literal string (not env var)",
                })


# ------------------------------------------------------------------
# Report
# ------------------------------------------------------------------

def _print_report(
    high: list[dict[str, Any]],
    medium: list[dict[str, Any]],
    scanned: int,
    json_output: bool,
) -> int:
    """Print report and return exit code (0 = pass, 1 = fail)."""
    if json_output:
        report = {
            "scanned_files": scanned,
            "high_confidence": len(high),
            "medium_confidence": len(medium),
            "findings": {
                "high": high,
                "medium": medium,
            },
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"\n{'=' * 60}")
        print("  Secret Scan Report")
        print(f"  Files scanned: {scanned}")
        print(f"{'=' * 60}")

        if high:
            print(f"\n  HIGH CONFIDENCE FINDINGS ({len(high)}):")
            print(f"  {'-' * 56}")
            for f in high:
                print(f"    {f['path']}:{f['line']}  {f['match']}")
        else:
            print("\n  HIGH CONFIDENCE: None")

        if medium:
            print(f"\n  MEDIUM CONFIDENCE FINDINGS ({len(medium)}):")
            print(f"  {'-' * 56}")
            for f in medium:
                print(f"    {f['path']}:{f['line']}  {f['pattern']}")
                print(f"      {f.get('match', f.get('detail', 'No details available'))}")
        else:
            print("\n  MEDIUM CONFIDENCE: None")

        print()

    return 1 if high else 0


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan for secrets and hardcoded credentials in the codebase.",
    )
    parser.add_argument(
        "--path", "-p",
        type=str,
        default=None,
        help="Scan a specific directory instead of the repository root.",
    )
    parser.add_argument(
        "--all-files", "-a",
        action="store_true",
        help="Scan all git-tracked files (default: staged only).",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output JSON report instead of human-readable text.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    root = Path(args.path).resolve() if args.path else Path(__file__).resolve().parents[2]

    # Collect files to scan
    if args.all_files:
        files = _git_tracked_files(root)
        if not files:
            print("No git-tracked files found; falling back to recursive walk.", file=sys.stderr)
            files = [p for p in root.rglob("*") if p.is_file() and _should_skip(p) is False]
    else:
        files = _git_staged_files(root)

    if not files:
        print("No files to scan.")
        return 0

    # Filter out skippable files
    files = [f for f in files if not _should_skip(f)]

    high: list[dict[str, Any]] = []
    medium: list[dict[str, Any]] = []

    for fpath in sorted(files):
        _scan_file(fpath, high, medium)

    return _print_report(high, medium, len(files), json_output=args.json)


if __name__ == "__main__":
    sys.exit(main())
