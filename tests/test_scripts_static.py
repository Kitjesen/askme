from __future__ import annotations

import ast
import re
from pathlib import Path


SCRIPTS_ROOT = Path("scripts")

SCRIPT_SOURCE_SUFFIXES = {
    ".bat",
    ".ps1",
    ".py",
    ".service",
    ".sh",
}

DANGEROUS_PATTERNS = {
    "rm -rf": re.compile(r"\brm\s+-rf\b"),
    "sudo": re.compile(r"\bsudo\b"),
    "systemctl": re.compile(r"\bsystemctl\b"),
    "rsync --delete": re.compile(r"\brsync\b[^\n]*\s--delete\b"),
    "shell=True": re.compile(r"shell\s*=\s*True"),
    "curl pipe shell": re.compile(r"\bcurl\b[^\n|]*\|[^\n]*(?:sh|bash)\b"),
    "powershell web request": re.compile(r"\b(?:Invoke-WebRequest|iwr)\b"),
    "start process": re.compile(r"\bStart-Process\b"),
}

MANUAL_OPERATION_ALLOWLIST = {
    ("scripts/dev/deploy_agentic_shell.sh", "sudo"),
    ("scripts/dev/deploy_agentic_shell.sh", "systemctl"),
    ("scripts/dev/e2e_test.sh", "systemctl"),
    ("scripts/demo/foxglove_bridge.sh", "sudo"),
    ("scripts/runtime/bridges/frame_daemon.py", "sudo"),
    ("scripts/runtime/bridges/frame_daemon.py", "systemctl"),
}


def _script_source_files() -> list[Path]:
    return sorted(
        path
        for path in SCRIPTS_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in SCRIPT_SOURCE_SUFFIXES
    )


def test_python_scripts_parse_without_execution() -> None:
    failures: list[str] = []
    for path in sorted(SCRIPTS_ROOT.rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            failures.append(f"{path}: {exc}")

    assert failures == []


def test_dangerous_operator_commands_are_explicitly_allowlisted() -> None:
    violations: list[str] = []
    for path in _script_source_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        relpath = path.as_posix()
        for label, pattern in DANGEROUS_PATTERNS.items():
            if not pattern.search(text):
                continue
            if (relpath, label) in MANUAL_OPERATION_ALLOWLIST:
                continue
            violations.append(f"{relpath} contains {label}")

    assert violations == []


def test_artifacts_bucket_does_not_contain_executable_scripts() -> None:
    artifact_scripts = [
        path
        for path in (SCRIPTS_ROOT / "artifacts").rglob("*")
        if path.is_file() and path.suffix.lower() in SCRIPT_SOURCE_SUFFIXES
    ]

    assert artifact_scripts == []
