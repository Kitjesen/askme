"""Tests for SandboxedBashTool and WriteFileTool — sandbox escape and normal execution."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from askme.tools.builtin_tools import SandboxedBashTool, WriteFileTool


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def bash_tool(tmp_path: Path) -> SandboxedBashTool:
    tool = SandboxedBashTool()
    tool._WORKSPACE = tmp_path / "workspace"
    tool._WORKSPACE.mkdir()
    return tool


@pytest.fixture()
def write_tool(tmp_path: Path) -> WriteFileTool:
    tool = WriteFileTool()
    tool._ALLOWED_ROOT = tmp_path / "workspace"
    tool._ALLOWED_ROOT.mkdir()
    return tool


# ── SandboxedBashTool: sandbox escape prevention ─────────────────────────────


def test_bash_blocks_dotdot_escape(bash_tool: SandboxedBashTool) -> None:
    """../.. cwd escape must be rejected."""
    result = bash_tool.execute(command="ls", cwd="../../etc")
    assert "[Error]" in result
    assert "逃逸" in result or "escape" in result.lower() or "不在允许" in result


def test_bash_blocks_absolute_path_escape(bash_tool: SandboxedBashTool) -> None:
    """Absolute cwd outside workspace must be rejected."""
    result = bash_tool.execute(command="ls", cwd="/etc")
    assert "[Error]" in result


def test_bash_blocks_deep_escape(bash_tool: SandboxedBashTool) -> None:
    """Deeply nested ../ escape must be rejected."""
    result = bash_tool.execute(command="echo hi", cwd="../../../../../etc/passwd")
    assert "[Error]" in result


# ── SandboxedBashTool: normal execution ──────────────────────────────────────


def test_bash_executes_echo(bash_tool: SandboxedBashTool) -> None:
    """Simple echo command returns output."""
    result = bash_tool.execute(command="echo hello_thunder")
    assert "hello_thunder" in result


def test_bash_executes_python(bash_tool: SandboxedBashTool) -> None:
    """Python -c command works inside workspace."""
    result = bash_tool.execute(command=f"{sys.executable} -c \"print('agentic_ok')\"")
    assert "agentic_ok" in result


def test_bash_creates_file_in_workspace(bash_tool: SandboxedBashTool, tmp_path: Path) -> None:
    """Bash can write files in the workspace."""
    result = bash_tool.execute(command="echo robot > output.txt")
    assert "[Error]" not in result and "[Timeout]" not in result
    output_file = bash_tool._WORKSPACE / "output.txt"
    assert output_file.exists()


def test_bash_empty_command_returns_error(bash_tool: SandboxedBashTool) -> None:
    result = bash_tool.execute(command="")
    assert "[Error]" in result


def test_bash_timeout(bash_tool: SandboxedBashTool) -> None:
    """Commands exceeding timeout return timeout message."""
    bash_tool._TIMEOUT = 1  # 1 second for test speed
    result = bash_tool.execute(command="sleep 5")
    assert "[Timeout]" in result


def test_bash_stderr_captured(bash_tool: SandboxedBashTool) -> None:
    """stderr output is included in result."""
    result = bash_tool.execute(
        command=f"{sys.executable} -c \"import sys; sys.stderr.write('error_output')\""
    )
    assert "error_output" in result


def test_bash_cwd_defaults_to_workspace(bash_tool: SandboxedBashTool) -> None:
    """No cwd argument → runs in workspace root."""
    result = bash_tool.execute(command="pwd" if os.name != "nt" else "cd")
    assert "[Error]" not in result


def test_bash_valid_subdir_cwd(bash_tool: SandboxedBashTool) -> None:
    """Valid subdirectory inside workspace is allowed."""
    subdir = bash_tool._WORKSPACE / "subdir"
    subdir.mkdir()
    result = bash_tool.execute(command="echo ok", cwd="subdir")
    assert "ok" in result


# ── WriteFileTool: sandbox escape prevention ─────────────────────────────────


def test_write_blocks_dotdot_escape(write_tool: WriteFileTool) -> None:
    result = write_tool.execute(path="../../evil.txt", content="bad")
    assert "[Error]" in result


def test_write_blocks_absolute_path_escape(write_tool: WriteFileTool) -> None:
    result = write_tool.execute(path="/tmp/evil.txt", content="bad")
    assert "[Error]" in result


# ── WriteFileTool: normal operation ──────────────────────────────────────────


def test_write_creates_file(write_tool: WriteFileTool) -> None:
    result = write_tool.execute(path="hello.txt", content="Hello Thunder!")
    assert "hello.txt" in result or "已写入" in result
    assert (write_tool._ALLOWED_ROOT / "hello.txt").read_text(encoding="utf-8") == "Hello Thunder!"


def test_write_creates_nested_directories(write_tool: WriteFileTool) -> None:
    result = write_tool.execute(path="scripts/run.py", content="print('ok')")
    assert "[Error]" not in result
    assert (write_tool._ALLOWED_ROOT / "scripts" / "run.py").exists()


def test_write_reports_char_count(write_tool: WriteFileTool) -> None:
    content = "Hello " * 10
    result = write_tool.execute(path="count.txt", content=content)
    assert str(len(content)) in result


def test_write_empty_path_returns_error(write_tool: WriteFileTool) -> None:
    result = write_tool.execute(path="", content="data")
    assert "[Error]" in result


def test_write_overwrites_existing_file(write_tool: WriteFileTool) -> None:
    write_tool.execute(path="over.txt", content="first")
    write_tool.execute(path="over.txt", content="second")
    content = (write_tool._ALLOWED_ROOT / "over.txt").read_text(encoding="utf-8")
    assert content == "second"


# ── _http_allowlist: runtime ports auto-included ─────────────────────────────


def test_http_allowlist_includes_runtime_ports() -> None:
    """Auto-allowlist includes all 7 runtime service ports."""
    from askme.tools.builtin_tools import _http_allowlist

    with patch("askme.tools.builtin_tools.get_section", return_value={}):
        allowlist = _http_allowlist()

    for port in (5050, 5060, 5070, 5080, 5090, 5100, 5110):
        assert f"http://localhost:{port}" in allowlist, f"Port {port} missing from allowlist"


def test_http_allowlist_includes_config_entries() -> None:
    """Config-defined entries are preserved in allowlist."""
    from askme.tools.builtin_tools import _http_allowlist

    with patch(
        "askme.tools.builtin_tools.get_section",
        return_value={"http_allowlist": ["http://my-service:9090"]},
    ):
        allowlist = _http_allowlist()

    assert "http://my-service:9090" in allowlist


def test_http_allowlist_includes_runtime_base_urls() -> None:
    """Runtime service base_url values are auto-added to allowlist."""
    from askme.tools.builtin_tools import _http_allowlist

    def _mock_get_section(section: str) -> dict:
        if section == "runtime":
            return {
                "dog_safety": {"base_url": "http://192.168.1.10:5070"},
                "dog_control": {"base_url": "http://192.168.1.10:5080"},
            }
        return {}

    with patch("askme.tools.builtin_tools.get_section", side_effect=_mock_get_section):
        allowlist = _http_allowlist()

    assert "http://192.168.1.10:5070" in allowlist
    assert "http://192.168.1.10:5080" in allowlist
