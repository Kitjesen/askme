"""Tests for EditFileTool — surgical string replacement."""

from __future__ import annotations

from pathlib import Path

import pytest

from askme.tools.builtin_tools import EditFileTool


@pytest.fixture()
def edit_tool(tmp_path: Path) -> EditFileTool:
    tool = EditFileTool()
    tool._WORKSPACE = tmp_path / "workspace"
    tool._WORKSPACE.mkdir()
    return tool


@pytest.fixture()
def sample_file(edit_tool: EditFileTool) -> Path:
    p = edit_tool._WORKSPACE / "hello.py"
    p.write_text("def greet():\n    return 'hello'\n", encoding="utf-8")
    return p


# ── Happy path ────────────────────────────────────────────────────────────────

class TestEditFileHappyPath:
    def test_simple_replacement(self, edit_tool: EditFileTool, sample_file: Path) -> None:
        result = edit_tool.execute(
            path=str(sample_file),
            old_string="return 'hello'",
            new_string="return 'world'",
        )
        assert "1 处" in result
        assert sample_file.read_text(encoding="utf-8") == "def greet():\n    return 'world'\n"

    def test_relative_path_resolves_to_workspace(self, edit_tool: EditFileTool) -> None:
        (edit_tool._WORKSPACE / "notes.txt").write_text("foo bar baz", encoding="utf-8")
        result = edit_tool.execute(path="notes.txt", old_string="bar", new_string="BAR")
        assert "1 处" in result
        assert (edit_tool._WORKSPACE / "notes.txt").read_text() == "foo BAR baz"

    def test_multiline_replacement(self, edit_tool: EditFileTool, sample_file: Path) -> None:
        result = edit_tool.execute(
            path=str(sample_file),
            old_string="def greet():\n    return 'hello'\n",
            new_string="def greet(name='world'):\n    return f'hello {name}'\n",
        )
        assert "1 处" in result
        assert "def greet(name=" in sample_file.read_text(encoding="utf-8")

    def test_result_reports_char_counts(self, edit_tool: EditFileTool, sample_file: Path) -> None:
        result = edit_tool.execute(
            path=str(sample_file),
            old_string="'hello'",
            new_string="'hi'",
        )
        assert "→" in result  # shows old_len → new_len


# ── Error cases ───────────────────────────────────────────────────────────────

class TestEditFileErrors:
    def test_empty_path_returns_error(self, edit_tool: EditFileTool) -> None:
        result = edit_tool.execute(path="", old_string="x", new_string="y")
        assert result.startswith("[Error]")

    def test_empty_old_string_returns_error(self, edit_tool: EditFileTool, sample_file: Path) -> None:
        result = edit_tool.execute(path=str(sample_file), old_string="", new_string="y")
        assert result.startswith("[Error]")

    def test_nonexistent_file_returns_error(self, edit_tool: EditFileTool) -> None:
        result = edit_tool.execute(
            path=str(edit_tool._WORKSPACE / "nope.txt"),
            old_string="x",
            new_string="y",
        )
        assert "[Error]" in result
        assert "不存在" in result

    def test_old_string_not_found(self, edit_tool: EditFileTool, sample_file: Path) -> None:
        result = edit_tool.execute(
            path=str(sample_file),
            old_string="DOES_NOT_EXIST",
            new_string="whatever",
        )
        assert "[Error]" in result
        assert "未找到" in result

    def test_non_unique_match_returns_error(self, edit_tool: EditFileTool) -> None:
        p = edit_tool._WORKSPACE / "dup.txt"
        p.write_text("abc abc abc", encoding="utf-8")
        result = edit_tool.execute(path=str(p), old_string="abc", new_string="XYZ")
        assert "[Error]" in result
        assert "3 次" in result

    def test_file_unchanged_on_not_found(self, edit_tool: EditFileTool, sample_file: Path) -> None:
        original = sample_file.read_text(encoding="utf-8")
        edit_tool.execute(path=str(sample_file), old_string="NO_MATCH", new_string="x")
        assert sample_file.read_text(encoding="utf-8") == original

    def test_directory_path_returns_error(self, edit_tool: EditFileTool) -> None:
        result = edit_tool.execute(
            path=str(edit_tool._WORKSPACE),
            old_string="x",
            new_string="y",
        )
        assert "[Error]" in result


# ── Registration ──────────────────────────────────────────────────────────────

class TestEditFileRegistration:
    def test_registered_in_builtin_tools(self) -> None:
        from askme.tools.builtin_tools import _BUILTIN_TOOLS
        names = [t.name for t in _BUILTIN_TOOLS]
        assert "edit_file" in names

    def test_in_agent_allowed_tools(self) -> None:
        from askme.tools.builtin_tools import EditFileTool
        assert EditFileTool.agent_allowed is True
