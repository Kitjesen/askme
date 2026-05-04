#!/usr/bin/env python3
"""Scan source text files for UTF-8 decode errors and mojibake markers."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_EXTENSIONS = {
    ".bat",
    ".cfg",
    ".cmd",
    ".env",
    ".example",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

DEFAULT_EXCLUDED_PARTS = {
    ".claude",
    ".git",
    ".mypy_cache",
    ".omc",
    ".omx",
    ".pytest_cache",
    ".ruff_cache",
    ".tmp",
    ".venv",
    ".venv-scientist",
    "__pycache__",
    "build",
    "data",
    "data/pytest-tmp",
    "dist",
    "htmlcov",
    "models",
    "node_modules",
    "tests/tmp",
}

MOJIBAKE_PATTERNS = {
    "replacement-character": "\ufffd",
    "box-drawing-gbk-mojibake": "\u9239\u20ac",
    "smart-punctuation-gbk-mojibake": "\u9225",
    "arrow-gbk-mojibake": "\u922b",
    "latin1-utf8-mojibake": "\u00c3",
    "nbsp-latin1-mojibake": "\u00c2",
    "cp1252-punctuation-mojibake": "\u00e2\u20ac",
    "replacement-latin1-mojibake": "\u00ef\u00bf\u00bd",
}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    column: int
    kind: str
    snippet: str


def scan_paths(
    paths: list[Path],
    *,
    extensions: set[str] | None = None,
    excluded_parts: set[str] | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    suffixes = extensions or DEFAULT_EXTENSIONS
    excludes = excluded_parts or DEFAULT_EXCLUDED_PARTS
    for path in _iter_text_files(paths, suffixes=suffixes, excluded_parts=excludes):
        findings.extend(scan_file(path))
    return findings


def scan_file(path: Path) -> list[Finding]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return [
            Finding(
                path=str(path),
                line=0,
                column=0,
                kind="read-error",
                snippet=str(exc),
            )
        ]

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [
            Finding(
                path=str(path),
                line=exc.start + 1,
                column=0,
                kind="utf8-decode-error",
                snippet=str(exc),
            )
        ]

    findings: list[Finding] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        for kind, marker in MOJIBAKE_PATTERNS.items():
            column = line.find(marker)
            if column >= 0:
                findings.append(
                    Finding(
                        path=str(path),
                        line=line_number,
                        column=column + 1,
                        kind=kind,
                        snippet=_ascii_snippet(line),
                    )
                )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan source text files for UTF-8 decode errors and mojibake markers"
    )
    parser.add_argument("paths", nargs="*", default=["."], help="Files or directories to scan")
    parser.add_argument("--json", action="store_true", help="Print machine-readable findings")
    parser.add_argument(
        "--extension",
        action="append",
        default=[],
        help="Additional file extension to scan, for example .csv",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional path part or relative path to exclude",
    )
    args = parser.parse_args(argv)

    extensions = set(DEFAULT_EXTENSIONS)
    extensions.update(_normalise_extension(ext) for ext in args.extension)
    excluded_parts = set(DEFAULT_EXCLUDED_PARTS)
    excluded_parts.update(arg.replace("\\", "/") for arg in args.exclude)

    findings = scan_paths(
        [Path(path) for path in args.paths],
        extensions=extensions,
        excluded_parts=excluded_parts,
    )
    if args.json:
        print(json.dumps([asdict(finding) for finding in findings], indent=2))  # noqa: T201
    elif findings:
        for finding in findings:
            print(  # noqa: T201
                f"{_ascii_path(finding.path)}:{finding.line}:{finding.column}: "
                f"{finding.kind}: {finding.snippet}"
            )
    else:
        print("OK: no UTF-8 decode errors or mojibake markers found")  # noqa: T201
    return 1 if findings else 0


def _iter_text_files(
    paths: list[Path],
    *,
    suffixes: set[str],
    excluded_parts: set[str],
) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            if _is_text_candidate(path, suffixes):
                files.append(path)
            continue
        if _is_excluded(path, excluded_parts):
            continue
        if path.is_dir():
            for child in path.rglob("*"):
                if _is_excluded(child, excluded_parts):
                    continue
                if child.is_file() and _is_text_candidate(child, suffixes):
                    files.append(child)
    return sorted(files)


def _is_text_candidate(path: Path, suffixes: set[str]) -> bool:
    if path.name.startswith(".env"):
        return True
    return path.suffix.lower() in suffixes


def _is_excluded(path: Path, excluded_parts: set[str]) -> bool:
    normalised = path.as_posix()
    parts = set(path.parts)
    for excluded in excluded_parts:
        if excluded in parts or normalised == excluded or normalised.startswith(f"{excluded}/"):
            return True
    return False


def _normalise_extension(extension: str) -> str:
    extension = extension.strip()
    if not extension:
        return extension
    return extension if extension.startswith(".") else f".{extension}"


def _ascii_snippet(line: str, *, limit: int = 140) -> str:
    escaped = line.encode("unicode_escape").decode("ascii")
    if len(escaped) <= limit:
        return escaped
    return f"{escaped[: limit - 3]}..."


def _ascii_path(path: str) -> str:
    return path.encode("unicode_escape").decode("ascii")


if __name__ == "__main__":
    raise SystemExit(main())
