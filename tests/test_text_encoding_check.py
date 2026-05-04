"""Tests for the repository text encoding checker."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_checker():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_text_encoding.py"
    spec = importlib.util.spec_from_file_location("check_text_encoding", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scan_file_accepts_valid_utf8_chinese(tmp_path: Path) -> None:
    checker = _load_checker()
    path = tmp_path / "ok.md"
    path.write_text("\u8bed\u97f3 health check\n", encoding="utf-8")

    assert checker.scan_file(path) == []


def test_scan_file_flags_replacement_character(tmp_path: Path) -> None:
    checker = _load_checker()
    path = tmp_path / "bad.md"
    path.write_text("broken \ufffd text\n", encoding="utf-8")

    findings = checker.scan_file(path)

    assert len(findings) == 1
    assert findings[0].kind == "replacement-character"
    assert findings[0].line == 1


def test_scan_file_flags_gbk_box_drawing_mojibake(tmp_path: Path) -> None:
    checker = _load_checker()
    path = tmp_path / "bad.md"
    path.write_text("comment \u9239\u20ac\u9239\u20ac\n", encoding="utf-8")

    findings = checker.scan_file(path)

    assert len(findings) == 1
    assert findings[0].kind == "box-drawing-gbk-mojibake"


def test_scan_file_flags_non_utf8_bytes(tmp_path: Path) -> None:
    checker = _load_checker()
    path = tmp_path / "bad.md"
    path.write_bytes(b"hello\xffworld\n")

    findings = checker.scan_file(path)

    assert len(findings) == 1
    assert findings[0].kind == "utf8-decode-error"


def test_main_returns_nonzero_for_findings(tmp_path: Path, capsys) -> None:
    checker = _load_checker()
    path = tmp_path / "bad.md"
    path.write_text("broken \ufffd text\n", encoding="utf-8")

    exit_code = checker.main([str(path)])

    assert exit_code == 1
    assert "replacement-character" in capsys.readouterr().out
