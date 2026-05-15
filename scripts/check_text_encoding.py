"""Lightweight repository text encoding checker."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
_REPLACEMENT_CHARACTER = "\ufffd"
_BOX_DRAWING_GBK_MOJIBAKE = "\u9239\u20ac"


@dataclass(frozen=True)
class Finding:
    path: Path
    kind: str
    line: int
    column: int
    message: str


def scan_file(path: str | Path) -> list[Finding]:
    """Return encoding findings for one file."""
    file_path = Path(path)
    raw = file_path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [
            Finding(
                path=file_path,
                kind="utf8-decode-error",
                line=raw[: exc.start].count(b"\n") + 1,
                column=exc.start - raw.rfind(b"\n", 0, exc.start),
                message=str(exc),
            )
        ]

    findings: list[Finding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        replacement_column = line.find(_REPLACEMENT_CHARACTER)
        if replacement_column >= 0:
            findings.append(
                Finding(
                    path=file_path,
                    kind="replacement-character",
                    line=line_number,
                    column=replacement_column + 1,
                    message="replacement character found",
                )
            )

        mojibake_column = line.find(_BOX_DRAWING_GBK_MOJIBAKE)
        if mojibake_column >= 0:
            findings.append(
                Finding(
                    path=file_path,
                    kind="box-drawing-gbk-mojibake",
                    line=line_number,
                    column=mojibake_column + 1,
                    message="probable GBK-decoded box drawing mojibake found",
                )
            )
    return findings


def iter_text_files(paths: Iterable[str | Path]) -> Iterable[Path]:
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            yield path
            continue
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and child.suffix.lower() in _TEXT_SUFFIXES:
                    yield child


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Files or directories to scan")
    args = parser.parse_args(argv)

    findings: list[Finding] = []
    for path in iter_text_files(args.paths):
        findings.extend(scan_file(path))

    for finding in findings:
        print(
            f"{finding.path}:{finding.line}:{finding.column}: "
            f"{finding.kind}: {finding.message}"
        )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
