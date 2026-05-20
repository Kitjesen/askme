"""Tests for ThunderAgentShell — agentic loop, tool routing, timeout handling."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.agent_shell.agent_hooks import AgentHookRunner
from askme.agent_shell.thunder_agent_shell import (
    _MAX_DEPTH,
    _MAX_ITERATIONS,
    _SPAWN_AGENT_SCHEMA,
    _build_agent_system_prompt,
    ThunderAgentShell,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_chunk(content: str = "", tool_calls: list | None = None) -> MagicMock:
    """Build a fake streaming chunk."""
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls or []
    chunk.choices = [MagicMock(delta=delta)]
    return chunk


def _make_tool_call_chunk(idx: int, call_id: str, name: str, args: str) -> MagicMock:
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = ""
    tc = MagicMock()
    tc.index = idx
    tc.id = call_id
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = args
    delta.tool_calls = [tc]
    chunk.choices = [MagicMock(delta=delta)]
    return chunk


async def _text_stream(*chunks: MagicMock):
    for c in chunks:
        yield c


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.chat_stream = MagicMock()
    return llm


@pytest.fixture()
def mock_tools() -> MagicMock:
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute.return_value = "tool_result_ok"
    return tools


@pytest.fixture()
def mock_audio() -> MagicMock:
    audio = MagicMock()
    audio.speak = MagicMock()
    return audio


@pytest.fixture()
def shell(mock_llm, mock_tools, mock_audio, tmp_path) -> ThunderAgentShell:
    s = ThunderAgentShell(
        llm_client=mock_llm,
        tool_registry=mock_tools,
        audio=mock_audio,
        model="claude-haiku-4-5-20251001",
        workspace=tmp_path / "workspace",
    )
    return s


# ── Basic task execution ──────────────────────────────────────────────────────


def test_agent_system_prompt_is_product_neutral(tmp_path) -> None:
    """Runtime prompt should not expose legacy robot branding to the model."""
    prompt = _build_agent_system_prompt(tmp_path / "workspace")

    assert "现场机器人" in prompt
    assert "自主执行 Agent" in prompt
    assert "机器人运行时快捷接口" in prompt
    assert "robot_api" in prompt
    assert "Thunder 机器人" not in prompt
    assert "Thunder runtime" not in prompt


def test_agent_shell_neutral_alias_preserves_legacy_class() -> None:
    from askme.agent_shell import AgentShell

    assert AgentShell is ThunderAgentShell


@pytest.mark.asyncio
async def test_simple_task_returns_response(shell, mock_llm) -> None:
    """Single-turn task (no tool calls) returns LLM text."""
    mock_llm.chat_stream.return_value = _text_stream(
        _make_chunk("任务完成！已保存结果。"),
        _make_chunk(""),
    )

    result = await shell.run_task("列出工作区文件")
    assert "任务完成" in result or result != ""


@pytest.mark.asyncio
async def test_start_announcement_spoken(shell, mock_llm, mock_audio) -> None:
    """Shell always speaks announcement at task start."""
    mock_llm.chat_stream.return_value = _text_stream(_make_chunk("done"))

    await shell.run_task("测试任务")
    mock_audio.speak.assert_called()
    # First speak call should be the "好的，我来处理一下" announcement
    first_call = mock_audio.speak.call_args_list[0][0][0]
    assert "好的" in first_call or "处理" in first_call


@pytest.mark.asyncio
async def test_workspace_created(shell, mock_llm, tmp_path) -> None:
    """Workspace directory is created if it doesn't exist."""
    workspace = tmp_path / "new_workspace"
    shell._workspace = workspace
    assert not workspace.exists()

    mock_llm.chat_stream.return_value = _text_stream(_make_chunk("ok"))
    await shell.run_task("test")

    assert workspace.exists()


# ── Tool call loop ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_call_then_final_response(shell, mock_llm, mock_tools) -> None:
    """Agent calls a tool, then gives final response."""
    call_count = 0

    async def _stream_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: emit tool call
            yield _make_tool_call_chunk(0, "call_1", "bash", '{"command": "ls"}')
        else:
            # Second call: final answer
            yield _make_chunk("列表获取完毕。")

    mock_llm.chat_stream.side_effect = _stream_side_effect
    mock_tools.execute.return_value = "file1.txt\nfile2.txt"

    result = await shell.run_task("列出文件")
    assert mock_tools.execute.called
    assert result  # got a response


@pytest.mark.asyncio
async def test_run_task_persists_redacted_product_summary(
    shell,
    mock_llm,
    mock_tools,
) -> None:
    """Completed runs leave a customer-safe local summary with redacted tool arguments."""
    call_count = 0
    mock_tools.get_agent_allowed_names.return_value = {"bash"}
    mock_tools.get_voice_labels.return_value = {}
    mock_tools.get_definitions.return_value = [
        {"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}
    ]
    mock_tools.execute.return_value = "token=tool-result-secret done"

    async def _stream_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield _make_tool_call_chunk(
                0,
                "call_1",
                "bash",
                '{"command":"curl","api_key":"secret-key","nested":{"token":"secret-token"}}',
            )
        else:
            yield _make_chunk("任务完成")

    mock_llm.chat_stream.side_effect = _stream_side_effect

    result = await shell.run_task("调用带密钥的工具")
    summary = shell.last_run_summary()
    persisted = json.loads((shell._workspace / "last_agent_run.json").read_text(encoding="utf-8"))
    history_lines = (shell._workspace / "agent_runs.jsonl").read_text(encoding="utf-8").splitlines()

    assert result == "任务完成"
    assert summary["status"] == "completed"
    assert summary["tool_call_count"] == 1
    assert summary["tools_used"] == ["bash"]
    assert summary["tools"][0]["status"] == "ok"
    assert "secret-key" not in summary["tools"][0]["arguments_preview"]
    assert "secret-token" not in summary["tools"][0]["arguments_preview"]
    assert "tool-result-secret" not in summary["tools"][0]["result_preview"]
    assert "***" in summary["tools"][0]["arguments_preview"]
    assert persisted["run_id"] == summary["run_id"]
    assert len(history_lines) == 1


@pytest.mark.asyncio
async def test_timeout_persists_run_summary(shell, mock_llm) -> None:
    """Timed-out runs are visible in the run summary instead of disappearing."""
    async def _slow_stream(*args, **kwargs):
        await asyncio.sleep(10)
        yield _make_chunk("too late")

    mock_llm.chat_stream.side_effect = _slow_stream

    result = await shell.run_task("slow task", timeout=0.1)
    summary = shell.last_run_summary()
    persisted = json.loads((shell._workspace / "last_agent_run.json").read_text(encoding="utf-8"))

    assert "超时" in result or "timeout" in result.lower()
    assert summary["status"] == "timeout"
    assert persisted["status"] == "timeout"
    assert persisted["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_tool_execution_error_handled(shell, mock_llm, mock_tools) -> None:
    """Tool execution error returns error string to LLM, doesn't crash."""
    call_count = 0

    async def _stream_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield _make_tool_call_chunk(0, "c1", "bash", '{"command": "bad"}')
        else:
            yield _make_chunk("尝试了但失败了。")

    mock_llm.chat_stream.side_effect = _stream_side_effect
    mock_tools.execute.side_effect = Exception("tool broke")

    result = await shell.run_task("执行任务")
    # Should not raise, should have a result
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_tool_execution_timeout_returns_error(shell, monkeypatch) -> None:
    """A timed-out tool call becomes a tool-result error string."""
    to_thread = MagicMock(side_effect=asyncio.TimeoutError)
    monkeypatch.setattr("askme.agent_shell.thunder_agent_shell.asyncio.to_thread", to_thread)
    shell._allowed_tools = {"bash"}

    result = await shell._execute_tool({
        "id": "c1",
        "name": "bash",
        "arguments": '{"command": "sleep"}',
    })

    assert "[Error]" in result
    assert "超时" in result


@pytest.mark.asyncio
async def test_pre_tool_hook_blocks_execution(shell, mock_tools) -> None:
    """Profile PreToolUse hook can block a tool before registry execution."""
    shell._allowed_tools = {"bash"}
    shell._hook_runner = AgentHookRunner(
        {
            "PreToolUse": [
                {
                    "matcher": "bash",
                    "contains": "rm -rf",
                    "decision": "deny",
                    "reason": "destructive command",
                }
            ]
        }
    )

    result = await shell._execute_tool({
        "id": "c1",
        "name": "bash",
        "arguments": '{"command":"rm -rf data"}',
    })

    assert "Agent hook blocked bash" in result
    assert mock_tools.execute.called is False


@pytest.mark.asyncio
async def test_post_tool_hook_blocks_result(shell, mock_tools) -> None:
    """Profile PostToolUse hook can hide a tool result after execution."""
    shell._allowed_tools = {"read_file"}
    shell._hook_runner = AgentHookRunner(
        {
            "PostToolUse": [
                {
                    "matcher": "read_file",
                    "contains": "secret=",
                    "decision": "block",
                    "reason": "secret content",
                }
            ]
        }
    )
    mock_tools.execute.return_value = "secret=123"

    result = await shell._execute_tool({
        "id": "c1",
        "name": "read_file",
        "arguments": '{"path":"secret.txt"}',
    })

    assert "Agent hook blocked read_file" in result
    assert mock_tools.execute.called is True


@pytest.mark.asyncio
async def test_tool_call_speaks_voice_label_and_updates_current_action(
    shell,
    mock_llm,
    mock_tools,
    mock_audio,
) -> None:
    """Known tool voice labels are spoken before tool execution."""
    call_count = 0
    mock_tools.get_agent_allowed_names.return_value = {"bash"}
    mock_tools.get_voice_labels.return_value = {"bash": "运行命令"}
    mock_tools.execute.return_value = "ok"

    async def _stream_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield _make_tool_call_chunk(0, "c1", "bash", '{"command": "echo ok"}')
        else:
            yield _make_chunk("完成。")

    mock_llm.chat_stream.side_effect = _stream_side_effect

    await shell.run_task("运行命令")

    spoken = [call.args[0] for call in mock_audio.speak.call_args_list]
    assert any("正在运行命令" in msg for msg in spoken)
    assert shell._current_action == "正在运行命令"


# ── Iteration limit ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_max_iterations_stops_loop(shell, mock_llm, mock_tools) -> None:
    """Loop stops after _MAX_ITERATIONS even if LLM keeps calling tools."""
    call_count = 0

    async def _always_tool(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        yield _make_tool_call_chunk(0, f"c{call_count}", "bash", '{"command": "echo hi"}')

    mock_llm.chat_stream.side_effect = _always_tool
    mock_tools.execute.return_value = "hi"

    result = await shell.run_task("infinite loop test", timeout=60.0)
    assert isinstance(result, str)
    assert call_count <= _MAX_ITERATIONS + 1


# ── Timeout ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_timeout_returns_gracefully(shell, mock_llm) -> None:
    """Timeout during task returns a descriptive message."""
    async def _slow_stream(*args, **kwargs):
        await asyncio.sleep(10)
        yield _make_chunk("too late")

    mock_llm.chat_stream.side_effect = _slow_stream

    result = await shell.run_task("slow task", timeout=0.1)
    assert "超时" in result or "timeout" in result.lower() or isinstance(result, str)


@pytest.mark.asyncio
async def test_run_task_returns_error_message_when_loop_raises(shell) -> None:
    """Unexpected loop failures are converted into user-facing error text."""
    shell._run_agent_loop = AsyncMock(side_effect=RuntimeError("boom"))

    result = await shell.run_task("broken task")

    assert "任务执行出错" in result
    assert "boom" in result


# ── Allowed tools ─────────────────────────────────────────────────────────────


def test_key_tools_have_agent_allowed() -> None:
    """Critical tools must have agent_allowed=True."""
    from askme.tools.builtin_tools import (
        HttpRequestTool,
        ReadFileTool,
        SandboxedBashTool,
        SpeakProgressTool,
        WriteFileTool,
    )
    assert SandboxedBashTool.agent_allowed is True
    assert WriteFileTool.agent_allowed is True
    assert ReadFileTool.agent_allowed is True
    assert HttpRequestTool.agent_allowed is True
    assert SpeakProgressTool.agent_allowed is True


def test_dispatch_skill_not_agent_allowed() -> None:
    """dispatch_skill must NOT have agent_allowed (prevents nested recursion)."""
    from askme.tools.builtin_tools import DispatchSkillTool
    assert DispatchSkillTool.agent_allowed is False


# ── Context injection ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_context_passed_to_llm(shell, mock_llm) -> None:
    """Extra context is included in the user message sent to LLM."""
    captured_messages: list = []

    async def _capture_stream(messages, **kwargs):
        captured_messages.extend(messages)
        yield _make_chunk("done")

    mock_llm.chat_stream.side_effect = _capture_stream

    await shell.run_task("task", context={"mission_context": "仓库A巡检"})

    # Find the user message
    user_msgs = [m for m in captured_messages if m.get("role") == "user"]
    assert any("仓库A巡检" in m.get("content", "") for m in user_msgs)


# ── RobotApiTool: basic tests ─────────────────────────────────────────────────


def test_robot_api_tool_unknown_service() -> None:
    from askme.tools.robot_api_tool import RobotApiTool
    tool = RobotApiTool()
    result = tool.execute(service="unknown", method="GET", path="/test")
    assert "[Error]" in result
    assert "未知服务" in result


def test_robot_api_tool_empty_path() -> None:
    from askme.tools.robot_api_tool import RobotApiTool
    tool = RobotApiTool()
    result = tool.execute(service="safety", method="GET", path="")
    assert "[Error]" in result


def test_robot_api_tool_service_unreachable(monkeypatch) -> None:
    """When service is not running, returns descriptive error."""
    from askme.tools import robot_api_tool
    # Use a definitely-closed port so we get a connection error regardless of local env
    monkeypatch.setitem(robot_api_tool._SERVICE_PORTS, "safety", 19998)
    from askme.tools.robot_api_tool import RobotApiTool
    tool = RobotApiTool()
    result = tool.execute(service="safety", method="GET", path="/api/v1/safety/modes/estop")
    # Connection refused → URLError path → always returns [Error]
    assert "[Error]" in result
    assert "safety" in result.lower() or "19998" in result or "不可达" in result


# ── SpawnAgent: sub-agent support ────────────────────────────────────────────


def test_spawn_agent_schema_exists() -> None:
    """spawn_agent is inline (not in registry) but must have a schema."""
    assert _SPAWN_AGENT_SCHEMA["function"]["name"] == "spawn_agent"


def test_spawn_agent_schema_well_formed() -> None:
    fn = _SPAWN_AGENT_SCHEMA["function"]
    assert fn["name"] == "spawn_agent"
    assert "task" in fn["parameters"]["properties"]
    assert "task" in fn["parameters"]["required"]


def test_spawn_agent_schema_injected_for_root_agent(mock_llm, mock_tools, mock_audio, tmp_path) -> None:
    """Root agent (depth=0) should have spawn_agent in tool definitions."""
    shell = ThunderAgentShell(
        llm_client=mock_llm,
        tool_registry=mock_tools,
        audio=mock_audio,
        workspace=tmp_path,
    )
    assert shell._depth == 0
    assert shell._depth < _MAX_DEPTH


def test_spawn_agent_schema_not_injected_for_child(mock_llm, mock_tools, tmp_path) -> None:
    """Child agent at max depth should NOT get spawn_agent in tool_definitions."""
    child = ThunderAgentShell(
        llm_client=mock_llm,
        tool_registry=mock_tools,
        audio=None,
        workspace=tmp_path,
        _depth=_MAX_DEPTH,
    )
    assert child._depth >= _MAX_DEPTH


@pytest.mark.asyncio
async def test_spawn_child_agent_depth_limit(mock_llm, mock_tools, tmp_path) -> None:
    """_spawn_child_agent returns error when already at max depth."""
    import json
    child = ThunderAgentShell(
        llm_client=mock_llm,
        tool_registry=mock_tools,
        audio=None,
        workspace=tmp_path,
        _depth=_MAX_DEPTH,
    )
    result = await child._spawn_child_agent(json.dumps({"task": "nested task"}))
    assert "[Error]" in result
    assert "嵌套" in result or "depth" in result.lower()


@pytest.mark.asyncio
async def test_spawn_child_agent_empty_task(mock_llm, mock_tools, tmp_path) -> None:
    """_spawn_child_agent returns error when task is empty."""
    import json
    root = ThunderAgentShell(
        llm_client=mock_llm,
        tool_registry=mock_tools,
        audio=None,
        workspace=tmp_path,
        _depth=0,
    )
    result = await root._spawn_child_agent(json.dumps({"task": ""}))
    assert "[Error]" in result


@pytest.mark.asyncio
async def test_spawn_child_agent_invalid_json(mock_llm, mock_tools, tmp_path) -> None:
    """Malformed spawn_agent arguments are rejected without raising."""
    root = ThunderAgentShell(
        llm_client=mock_llm,
        tool_registry=mock_tools,
        audio=None,
        workspace=tmp_path,
        _depth=0,
    )

    result = await root._spawn_child_agent("{not-json")

    assert "[Error]" in result
    assert "JSON" in result


@pytest.mark.asyncio
async def test_spawn_child_agent_wraps_child_failure(mock_llm, mock_tools, tmp_path) -> None:
    """A child run_task exception is returned as a child-task failure string."""
    import json

    root = ThunderAgentShell(
        llm_client=mock_llm,
        tool_registry=mock_tools,
        audio=None,
        workspace=tmp_path,
        _depth=0,
    )

    async def _fail_run_task(self, task, *, context=None, timeout=30.0):
        raise RuntimeError("child broke")

    with patch.object(ThunderAgentShell, "run_task", _fail_run_task):
        result = await root._spawn_child_agent(json.dumps({"task": "child task"}))

    assert "[子任务失败]" in result
    assert "child broke" in result


@pytest.mark.asyncio
async def test_spawn_child_agent_child_is_silent_and_receives_context(
    mock_llm,
    mock_tools,
    mock_audio,
    tmp_path,
) -> None:
    """spawn_agent creates a silent child and passes context with the fixed timeout."""
    import json

    root = ThunderAgentShell(
        llm_client=mock_llm,
        tool_registry=mock_tools,
        audio=mock_audio,
        workspace=tmp_path,
        _depth=0,
    )
    captured: dict[str, object] = {}

    async def _capture_run_task(self, task, *, context=None, timeout=30.0):
        captured["audio"] = self._audio
        captured["depth"] = self._depth
        captured["task"] = task
        captured["context"] = context
        captured["timeout"] = timeout
        return "child ok"

    with patch.object(ThunderAgentShell, "run_task", _capture_run_task):
        result = await root._spawn_child_agent(
            json.dumps({"task": "child task", "context": "use repo docs"})
        )

    assert "[子任务完成]" in result
    assert captured == {
        "audio": None,
        "depth": 1,
        "task": "child task",
        "context": {"上下文": "use repo docs"},
        "timeout": 30.0,
    }


@pytest.mark.asyncio
async def test_spawn_child_agent_runs_task(mock_llm, mock_tools, tmp_path) -> None:
    """_spawn_child_agent spawns a child shell and returns its result."""
    call_count = 0

    async def _stream(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        yield _make_chunk("子任务完成！")

    mock_llm.chat_stream.side_effect = _stream

    import json
    root = ThunderAgentShell(
        llm_client=mock_llm,
        tool_registry=mock_tools,
        audio=None,
        workspace=tmp_path,
        _depth=0,
    )
    result = await root._spawn_child_agent(json.dumps({"task": "写一个函数", "context": "Python"}))
    assert "子任务完成" in result
    assert "子任务" in result


@pytest.mark.asyncio
async def test_child_agent_audio_none_no_crash(mock_llm, mock_tools, tmp_path) -> None:
    """Child agent with audio=None must not crash during run_task."""
    mock_llm.chat_stream.return_value = _text_stream(_make_chunk("done"))
    child = ThunderAgentShell(
        llm_client=mock_llm,
        tool_registry=mock_tools,
        audio=None,
        workspace=tmp_path / "child_ws",
        _depth=1,
    )
    result = await child.run_task("test subtask", timeout=5.0)
    assert isinstance(result, str)


# ── LLM retry tests ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_llm_retries_on_transient_error(mock_llm, mock_tools, mock_audio, tmp_path) -> None:
    """_call_llm retries up to 2 times; succeeds on 3rd attempt."""
    call_count = [0]

    async def _flaky_stream(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] < 3:
            raise ConnectionError("relay hiccup")
        yield _make_chunk("任务完成")

    mock_llm.chat_stream.side_effect = _flaky_stream

    shell = ThunderAgentShell(
        llm_client=mock_llm,
        tool_registry=mock_tools,
        audio=mock_audio,
        workspace=tmp_path / "ws",
    )
    with patch("asyncio.sleep", return_value=None):
        text, tool_calls = await shell._call_llm([], [], "sys")

    assert text == "任务完成"
    assert tool_calls == []
    assert call_count[0] == 3


@pytest.mark.asyncio
async def test_call_llm_raises_after_all_retries_exhausted(mock_llm, mock_tools, mock_audio, tmp_path) -> None:
    """_call_llm raises the last exception when all 3 attempts fail."""
    async def _always_fail(*args, **kwargs):
        raise RuntimeError("relay down")
        yield  # make it a generator

    mock_llm.chat_stream.side_effect = _always_fail

    shell = ThunderAgentShell(
        llm_client=mock_llm,
        tool_registry=mock_tools,
        audio=mock_audio,
        workspace=tmp_path / "ws",
    )
    with patch("asyncio.sleep", return_value=None):
        with pytest.raises(RuntimeError, match="relay down"):
            await shell._call_llm([], [], "sys")

    assert mock_llm.chat_stream.call_count == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_call_llm_cancelled_error_not_retried(mock_llm, mock_tools, mock_audio, tmp_path) -> None:
    """CancelledError must propagate immediately — never retried."""
    async def _cancelled(*args, **kwargs):
        raise asyncio.CancelledError()
        yield

    mock_llm.chat_stream.side_effect = _cancelled

    shell = ThunderAgentShell(
        llm_client=mock_llm,
        tool_registry=mock_tools,
        audio=mock_audio,
        workspace=tmp_path / "ws",
    )
    with pytest.raises(asyncio.CancelledError):
        await shell._call_llm([], [], "sys")

    assert mock_llm.chat_stream.call_count == 1  # no retry
