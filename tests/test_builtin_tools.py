"""Tests for builtin tools — path guards, URL allowlist, and core tool execution."""

from __future__ import annotations

from askme.config import project_root
from askme.tools.builtin_tools import (
    GetTimeTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
    _is_path_allowed,
    _is_url_allowed,
)

# ── _is_path_allowed ──────────────────────────────────────────────────────────

class TestIsPathAllowed:
    def test_data_dir_allowed(self):
        path = str(project_root() / "data" / "skills_settings.json")
        assert _is_path_allowed(path) is True

    def test_logs_dir_allowed(self):
        path = str(project_root() / "logs" / "askme.log")
        assert _is_path_allowed(path) is True

    def test_skills_dir_allowed(self):
        path = str(project_root() / "askme" / "skills" / "builtin" / "patrol" / "SKILL.md")
        assert _is_path_allowed(path) is True

    def test_source_code_blocked(self):
        path = str(project_root() / "askme" / "config.py")
        assert _is_path_allowed(path) is False

    def test_config_yaml_blocked(self):
        path = str(project_root() / "config.yaml")
        assert _is_path_allowed(path) is False

    def test_dot_env_blocked(self):
        path = str(project_root() / ".env")
        assert _is_path_allowed(path) is False

    def test_path_traversal_blocked(self):
        # Tries to escape data/ via ../..
        path = str(project_root() / "data" / ".." / ".." / "etc" / "passwd")
        assert _is_path_allowed(path) is False

    def test_invalid_path_returns_false(self):
        # Null byte in path
        assert _is_path_allowed("invalid\x00path") is False


# ── _is_url_allowed ───────────────────────────────────────────────────────────

class TestIsUrlAllowed:
    def test_empty_allowlist_allows_localhost(self):
        assert _is_url_allowed("http://localhost:8080/api", []) is True

    def test_empty_allowlist_allows_127_0_0_1(self):
        assert _is_url_allowed("http://127.0.0.1:5000/api", []) is True

    def test_empty_allowlist_blocks_external(self):
        assert _is_url_allowed("https://example.com/api", []) is False

    def test_allowlist_prefix_match(self):
        assert _is_url_allowed("http://myservice:8080/endpoint", ["http://myservice:8080"]) is True

    def test_allowlist_prefix_not_matching(self):
        assert _is_url_allowed("http://other:8080/api", ["http://myservice:8080"]) is False

    def test_multiple_allowlist_entries(self):
        allowlist = ["http://service-a:5000", "http://service-b:6000"]
        assert _is_url_allowed("http://service-b:6000/endpoint", allowlist) is True

    def test_url_must_start_with_prefix(self):
        # Substring match in middle should not count
        assert _is_url_allowed("http://evil.com/http://service:5000", ["http://service:5000"]) is False


# ── GetTimeTool ───────────────────────────────────────────────────────────────

class TestGetTimeTool:
    def test_returns_string(self):
        tool = GetTimeTool()
        result = tool.execute()
        assert isinstance(result, str)

    def test_format_looks_like_datetime(self):
        tool = GetTimeTool()
        result = tool.execute()
        # Should be YYYY-MM-DD HH:MM:SS
        parts = result.split(" ")
        assert len(parts) == 2
        date_part, time_part = parts
        assert len(date_part) == 10 and date_part[4] == "-"
        assert len(time_part) == 8 and time_part[2] == ":"


# ── ReadFileTool ──────────────────────────────────────────────────────────────

class TestReadFileTool:
    def test_reads_allowed_file(self, tmp_path, monkeypatch):
        # Patch the allowed roots to include tmp_path
        import askme.tools.builtin_tools as bt
        monkeypatch.setattr(bt, "_ALLOWED_READ_ROOTS", (tmp_path,))
        file = tmp_path / "test.txt"
        file.write_text("hello world", encoding="utf-8")
        tool = ReadFileTool()
        result = tool.execute(path=str(file))
        assert result == "hello world"

    def test_blocked_path_returns_error(self):
        tool = ReadFileTool()
        result = tool.execute(path="/etc/passwd")
        assert "[Error]" in result
        assert "允许" in result or "范围" in result

    def test_empty_path_returns_error(self):
        tool = ReadFileTool()
        result = tool.execute(path="")
        assert "[Error]" in result

    def test_nonexistent_allowed_file_returns_error(self, tmp_path, monkeypatch):
        import askme.tools.builtin_tools as bt
        monkeypatch.setattr(bt, "_ALLOWED_READ_ROOTS", (tmp_path,))
        tool = ReadFileTool()
        result = tool.execute(path=str(tmp_path / "nonexistent.txt"))
        assert "[Error]" in result

    def test_truncates_at_3000_chars(self, tmp_path, monkeypatch):
        import askme.tools.builtin_tools as bt
        monkeypatch.setattr(bt, "_ALLOWED_READ_ROOTS", (tmp_path,))
        file = tmp_path / "big.txt"
        file.write_text("x" * 5000, encoding="utf-8")
        tool = ReadFileTool()
        result = tool.execute(path=str(file))
        assert len(result) == 3000


# ── ListDirectoryTool ─────────────────────────────────────────────────────────

class TestListDirectoryTool:
    def test_lists_allowed_directory(self, tmp_path, monkeypatch):
        import askme.tools.builtin_tools as bt
        monkeypatch.setattr(bt, "_ALLOWED_READ_ROOTS", (tmp_path,))
        (tmp_path / "file_a.txt").write_text("a")
        (tmp_path / "file_b.txt").write_text("b")
        tool = ListDirectoryTool()
        result = tool.execute(path=str(tmp_path))
        assert "file_a.txt" in result
        assert "file_b.txt" in result

    def test_blocked_directory_returns_error(self):
        tool = ListDirectoryTool()
        result = tool.execute(path="/etc")
        assert "[Error]" in result

    def test_nonexistent_directory_returns_error(self, tmp_path, monkeypatch):
        import askme.tools.builtin_tools as bt
        monkeypatch.setattr(bt, "_ALLOWED_READ_ROOTS", (tmp_path,))
        tool = ListDirectoryTool()
        result = tool.execute(path=str(tmp_path / "nonexistent"))
        assert "[Error]" in result

    def test_at_most_50_entries(self, tmp_path, monkeypatch):
        import askme.tools.builtin_tools as bt
        monkeypatch.setattr(bt, "_ALLOWED_READ_ROOTS", (tmp_path,))
        for i in range(60):
            (tmp_path / f"file_{i:03d}.txt").write_text(str(i))
        tool = ListDirectoryTool()
        result = tool.execute(path=str(tmp_path))
        # At most 50 lines (files)
        assert len(result.splitlines()) <= 50


# ── WriteFileTool ─────────────────────────────────────────────────────────────

class TestWriteFileTool:
    def _patched_tool(self, tmp_path) -> WriteFileTool:
        tool = WriteFileTool()
        # Override workspace to use tmp_path
        tool._ALLOWED_ROOT = tmp_path / "workspace"
        return tool

    def test_writes_file_successfully(self, tmp_path):
        tool = self._patched_tool(tmp_path)
        result = tool.execute(path="output.txt", content="hello world")
        assert "[Error]" not in result
        written = (tmp_path / "workspace" / "output.txt").read_text()
        assert written == "hello world"

    def test_empty_path_returns_error(self, tmp_path):
        tool = self._patched_tool(tmp_path)
        result = tool.execute(path="", content="data")
        assert "[Error]" in result

    def test_path_traversal_blocked(self, tmp_path):
        tool = self._patched_tool(tmp_path)
        result = tool.execute(path="../../etc/passwd", content="bad")
        assert "[Error]" in result
        # File should NOT be written outside workspace
        assert not (tmp_path / "etc" / "passwd").exists()

    def test_nested_subdir_created(self, tmp_path):
        tool = self._patched_tool(tmp_path)
        result = tool.execute(path="subdir/nested/file.txt", content="data")
        assert "[Error]" not in result
        assert (tmp_path / "workspace" / "subdir" / "nested" / "file.txt").exists()

    def test_result_contains_char_count(self, tmp_path):
        tool = self._patched_tool(tmp_path)
        content = "hello world"
        result = tool.execute(path="test.txt", content=content)
        assert str(len(content)) in result

    def test_overwrites_existing_file(self, tmp_path):
        tool = self._patched_tool(tmp_path)
        tool.execute(path="file.txt", content="original")
        tool.execute(path="file.txt", content="updated")
        written = (tmp_path / "workspace" / "file.txt").read_text()
        assert written == "updated"


# ── RunCommandTool ────────────────────────────────────────────────────────────

class TestRunCommandTool:
    def test_empty_command_returns_error(self):
        from askme.tools.builtin_tools import RunCommandTool
        tool = RunCommandTool()
        assert "[Error]" in tool.execute(command="")

    def test_simple_echo_command(self):
        from askme.tools.builtin_tools import RunCommandTool
        tool = RunCommandTool()
        result = tool.execute(command="echo hello")
        assert "hello" in result

    def test_invalid_syntax_returns_error(self):
        from askme.tools.builtin_tools import RunCommandTool
        tool = RunCommandTool()
        # Unmatched quote = shlex.split error
        result = tool.execute(command="echo 'unclosed")
        assert "[Error]" in result

    def test_output_truncated_at_2000(self):
        from askme.tools.builtin_tools import RunCommandTool
        tool = RunCommandTool()
        # Generate output > 2000 chars via python
        result = tool.execute(command="python3 -c \"print('x' * 3000)\"")
        assert len(result) <= 2000
