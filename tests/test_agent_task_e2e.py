"""
End-to-end integration test for the agent_task flow.

Tests the full chain:
  user says "帮我写一个脚本"
    → IntentRouter routes to "agent_task" skill
      → SkillDispatcher.dispatch()
        → BrainPipeline.execute_skill("agent_task")
          → ThunderAgentShell.run_task()
            → Mock LLM: bash tool call → write_file → final answer
              → result spoken + returned

No real LLM, TTS, filesystem (except tmp workspace) or network needed.
"""

from __future__ import annotations

import asyncio
import json
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.agent_shell.thunder_agent_shell import ThunderAgentShell, _AGENT_ALLOWED_TOOLS
from askme.tools.tool_registry import ToolRegistry
from askme.tools.builtin_tools import SandboxedBashTool, WriteFileTool, GetTimeTool


# ── Mock LLM helpers ──────────────────────────────────────────────────────────


def _tc_chunk(idx: int, call_id: str, name: str, args: str) -> Any:
    tc = types.SimpleNamespace(
        index=idx,
        id=call_id,
        function=types.SimpleNamespace(name=name, arguments=args),
    )
    delta = types.SimpleNamespace(tool_calls=[tc], content="")
    return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])


def _text_chunk(text: str) -> Any:
    delta = types.SimpleNamespace(tool_calls=[], content=text)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])


async def _one_tool_then_answer(write_args: dict[str, str], answer: str):
    """Simulate: LLM calls write_file once, then gives final answer."""
    call_count = 0

    async def _stream(messages, tools=None, tool_choice=None, model=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: emit a write_file tool call
            yield _tc_chunk(0, "c1", "write_file", json.dumps(write_args))
        else:
            # Second call: final answer after tool result
            yield _text_chunk(answer)

    return _stream


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "agent_workspace"
    ws.mkdir()
    return ws


@pytest.fixture()
def tool_registry(workspace: Path) -> ToolRegistry:
    """Registry with real bash, write_file, get_current_time tools (sandbox to tmp workspace)."""
    registry = ToolRegistry(config={
        "default_timeout": 10.0,
        "dangerous_timeout": 35.0,
        "require_confirmation_levels": [],  # no confirmation needed in tests
        "approval_timeout_seconds": 0,
    })

    bash = SandboxedBashTool()
    bash._WORKSPACE = workspace
    registry.register(bash)

    wf = WriteFileTool()
    wf._ALLOWED_ROOT = workspace
    registry.register(wf)

    registry.register(GetTimeTool())
    return registry


@pytest.fixture()
def mock_audio() -> MagicMock:
    audio = MagicMock()
    audio.speak = MagicMock()
    return audio


@pytest.fixture()
def mock_llm() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def shell(mock_llm, tool_registry, mock_audio, workspace) -> ThunderAgentShell:
    return ThunderAgentShell(
        llm_client=mock_llm,
        tool_registry=tool_registry,
        audio=mock_audio,
        model="claude-haiku-4-5-20251001",
        workspace=workspace,
    )


# ── Core agent loop tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_writes_file_and_returns_answer(shell, mock_llm, workspace) -> None:
    """Agent calls write_file tool then returns a final answer."""
    write_args = {"path": "hello.py", "content": "print('Hello Thunder!')"}
    mock_llm.chat_stream.side_effect = await _one_tool_then_answer(
        write_args, "脚本已写好，保存在 hello.py。"
    )

    result = await shell.run_task("帮我写一个 hello world 脚本")

    assert "脚本" in result or "hello" in result.lower()
    assert (workspace / "hello.py").exists()
    assert "Hello Thunder!" in (workspace / "hello.py").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_agent_executes_bash_command(shell, mock_llm, workspace) -> None:
    """Agent calls bash to list workspace, then returns answer."""
    call_count = 0

    async def _stream(messages, tools=None, tool_choice=None, model=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield _tc_chunk(0, "c1", "bash", json.dumps({"command": "echo thunder_ok"}))
        else:
            yield _text_chunk("命令执行完毕，输出是 thunder_ok。")

    mock_llm.chat_stream.side_effect = _stream

    result = await shell.run_task("在工作区执行一条命令")
    assert isinstance(result, str)
    assert result  # has some response


@pytest.mark.asyncio
async def test_agent_speaks_progress_during_task(shell, mock_llm, mock_audio) -> None:
    """Agent speaks initial announcement before any tool calls."""
    mock_llm.chat_stream.side_effect = lambda *a, **kw: (
        x for x in [_text_chunk("完成")]
    )

    # Can't really test async generator return from lambda easily — use proper async def
    async def _fast_stream(messages, tools=None, tool_choice=None, model=None):
        yield _text_chunk("完成")

    mock_llm.chat_stream.side_effect = _fast_stream

    await shell.run_task("测试任务")

    speak_calls = [c[0][0] for c in mock_audio.speak.call_args_list]
    # First call should be the start announcement
    assert any("好的" in s or "处理" in s for s in speak_calls)


@pytest.mark.asyncio
async def test_agent_tool_result_passed_to_llm(shell, mock_llm, workspace) -> None:
    """Tool result is included in the follow-up LLM messages."""
    captured_messages: list[list] = []

    call_count = 0

    async def _stream(messages, tools=None, tool_choice=None, model=None):
        nonlocal call_count
        call_count += 1
        captured_messages.append(list(messages))
        if call_count == 1:
            yield _tc_chunk(0, "c1", "get_current_time", "{}")
        else:
            yield _text_chunk("时间获取成功。")

    mock_llm.chat_stream.side_effect = _stream

    await shell.run_task("现在几点了")

    # Second LLM call should include tool result
    assert len(captured_messages) == 2
    second_call_msgs = captured_messages[1]
    roles = [m.get("role") for m in second_call_msgs]
    assert "tool" in roles


@pytest.mark.asyncio
async def test_agent_handles_tool_error_gracefully(shell, mock_llm, workspace) -> None:
    """If a tool raises an exception, result is error string — loop continues."""
    call_count = 0

    async def _stream(messages, tools=None, tool_choice=None, model=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Try to write to an illegal path — will get error string back
            yield _tc_chunk(0, "c1", "write_file", json.dumps({
                "path": "../../escape.txt", "content": "bad"
            }))
        else:
            yield _text_chunk("好的，改用工作区路径重试。")

    mock_llm.chat_stream.side_effect = _stream

    result = await shell.run_task("写文件")
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_agent_timeout_returns_message(shell, mock_llm) -> None:
    """Timeout produces a message, doesn't crash."""
    async def _slow_stream(messages, tools=None, tool_choice=None, model=None):
        await asyncio.sleep(10)
        yield _text_chunk("too late")

    mock_llm.chat_stream.side_effect = _slow_stream

    result = await shell.run_task("slow task", timeout=0.1)
    assert isinstance(result, str)
    assert len(result) > 0


# ── Routing: agent_task skill exists + voice trigger ─────────────────────────


def test_agent_task_skill_loaded() -> None:
    """SkillManager loads agent_task with correct metadata."""
    from askme.skills.skill_manager import SkillManager
    sm = SkillManager()
    sm.load()
    skill = sm.get("agent_task")
    assert skill is not None, "agent_task skill not found"
    assert skill.timeout == 120
    assert skill.safety_level == "dangerous"
    assert skill.voice_trigger  # has voice triggers


def test_agent_task_voice_triggers_registered() -> None:
    """Voice trigger phrases route to agent_task."""
    from askme.skills.skill_manager import SkillManager
    sm = SkillManager()
    sm.load()
    triggers = sm.get_voice_triggers()
    agent_triggers = [k for k, v in triggers.items() if v == "agent_task"]
    assert len(agent_triggers) > 0, "No voice triggers registered for agent_task"


def test_agent_task_not_shadowing_web_search() -> None:
    """agent_task voice triggers don't conflict with web_search triggers."""
    from askme.skills.skill_manager import SkillManager
    sm = SkillManager()
    sm.load()
    triggers = sm.get_voice_triggers()
    # "帮我搜索" should route to web_search, not agent_task
    search_trigger = triggers.get("帮我搜索")
    if search_trigger is not None:
        assert search_trigger == "web_search", (
            f"'帮我搜索' should route to web_search, got {search_trigger!r}"
        )


# ── ThunderAgentShell wiring in BrainPipeline ────────────────────────────────


@pytest.mark.asyncio
async def test_brain_pipeline_routes_agent_task(tmp_path: Path) -> None:
    """BrainPipeline.execute_skill('agent_task') routes to ThunderAgentShell.run_task()."""
    from askme.pipeline.brain_pipeline import BrainPipeline

    # Minimal mock deps
    audio = MagicMock()
    audio.speak = MagicMock()
    audio.drain_buffers = MagicMock()
    audio.start_playback = MagicMock()
    audio.stop_playback = MagicMock()
    audio.wait_speaking_done = MagicMock()

    mock_skill = MagicMock()
    mock_skill.depends = []
    skill_mgr = MagicMock()
    skill_mgr.get.return_value = mock_skill
    skill_mgr.get_enabled.return_value = []
    skill_mgr.get_skill_catalog.return_value = "agent_task"

    agent_shell = AsyncMock()
    agent_shell.run_task.return_value = "任务完成，脚本已创建。"

    pipeline = BrainPipeline(
        llm=MagicMock(),
        conversation=MagicMock(),
        memory=MagicMock(),
        tools=MagicMock(),
        skill_manager=skill_mgr,
        skill_executor=MagicMock(),
        audio=audio,
        splitter=MagicMock(),
        agent_shell=agent_shell,
    )

    result = await pipeline.execute_skill("agent_task", "帮我写一个脚本")

    agent_shell.run_task.assert_called_once()
    call_kwargs = agent_shell.run_task.call_args
    assert call_kwargs[0][0] == "帮我写一个脚本" or "帮我写一个脚本" in str(call_kwargs)
    assert "任务完成" in result


# ── _prepare_agent_result truncation ─────────────────────────────────────────


def _make_pipeline_with_agent_shell(tmp_path):
    """Build a minimal BrainPipeline with a mock agent_shell."""
    from askme.pipeline.brain_pipeline import BrainPipeline
    from askme.agent_shell.thunder_agent_shell import ThunderAgentShell

    audio = MagicMock()
    audio.speak = MagicMock()
    audio.drain_buffers = MagicMock()
    audio.start_playback = MagicMock()
    audio.stop_playback = MagicMock()
    audio.wait_speaking_done = MagicMock()

    agent_shell = MagicMock(spec=ThunderAgentShell)
    agent_shell._workspace = tmp_path / "workspace"
    agent_shell._workspace.mkdir()
    agent_shell._default_timeout = 120.0

    pipeline = BrainPipeline(
        llm=MagicMock(),
        conversation=MagicMock(),
        memory=MagicMock(),
        tools=MagicMock(),
        skill_manager=MagicMock(),
        skill_executor=MagicMock(),
        audio=audio,
        splitter=MagicMock(),
        agent_shell=agent_shell,
        max_response_chars=50,  # low limit to trigger truncation
    )
    return pipeline, agent_shell


def test_prepare_agent_result_short_returns_unchanged(tmp_path) -> None:
    """Short result is returned as-is without truncation."""
    pipeline, agent_shell = _make_pipeline_with_agent_shell(tmp_path)
    short = "任务完成。"
    spoken, stored = pipeline._prepare_agent_result(short)
    assert spoken == short
    assert stored == short


def test_prepare_agent_result_long_truncates_for_tts(tmp_path) -> None:
    """Long result is truncated for TTS and saved to workspace."""
    pipeline, agent_shell = _make_pipeline_with_agent_shell(tmp_path)
    long_result = "第一句话。" * 20  # 100+ chars
    spoken, stored = pipeline._prepare_agent_result(long_result)

    assert len(spoken) < len(long_result)
    assert "完整结果已保存" in spoken
    assert stored == long_result


def test_prepare_agent_result_saves_file_to_workspace(tmp_path) -> None:
    """Full result is written to workspace/last_result.txt when truncated."""
    pipeline, agent_shell = _make_pipeline_with_agent_shell(tmp_path)
    long_result = "详细分析结果。" * 30

    pipeline._prepare_agent_result(long_result)

    saved = agent_shell._workspace / "last_result.txt"
    assert saved.exists()
    assert saved.read_text(encoding="utf-8") == long_result


def test_prepare_agent_result_truncates_at_sentence_boundary(tmp_path) -> None:
    """Truncation prefers sentence boundary over hard character cut."""
    pipeline, agent_shell = _make_pipeline_with_agent_shell(tmp_path)
    # 30 chars up to 。, then more content
    result = "第一句完整的话。" + "额外内容" * 20
    spoken, _ = pipeline._prepare_agent_result(result)
    # Should cut at the 。 not mid-word
    assert spoken.startswith("第一句完整的话")


# ── ReadFileTool relative path resolution ──────────────────────────────────────


def test_read_file_relative_path_resolved_to_workspace(monkeypatch, tmp_path) -> None:
    """Relative paths in read_file are resolved against data/agent_workspace/."""
    import askme.tools.builtin_tools as bt

    workspace = tmp_path / "data" / "agent_workspace"
    workspace.mkdir(parents=True)
    (workspace / "result.txt").write_text("agent result here", encoding="utf-8")

    monkeypatch.setattr(bt, "project_root", lambda: tmp_path)
    monkeypatch.setattr(bt, "_ALLOWED_READ_ROOTS", (tmp_path / "data",))

    tool = bt.ReadFileTool()
    result = tool.execute(path="result.txt")
    assert "agent result here" in result


def test_read_file_relative_subpath_resolved_to_workspace(monkeypatch, tmp_path) -> None:
    """Nested relative paths (scripts/hello.py) resolve inside agent_workspace."""
    import askme.tools.builtin_tools as bt

    workspace = tmp_path / "data" / "agent_workspace"
    (workspace / "scripts").mkdir(parents=True)
    (workspace / "scripts" / "hello.py").write_text("print('hello')", encoding="utf-8")

    monkeypatch.setattr(bt, "project_root", lambda: tmp_path)
    monkeypatch.setattr(bt, "_ALLOWED_READ_ROOTS", (tmp_path / "data",))

    tool = bt.ReadFileTool()
    result = tool.execute(path="scripts/hello.py")
    assert "print('hello')" in result


def test_read_file_relative_escape_blocked(monkeypatch, tmp_path) -> None:
    """Relative path with ../ cannot escape workspace (security check still applies)."""
    import askme.tools.builtin_tools as bt

    workspace = tmp_path / "data" / "agent_workspace"
    workspace.mkdir(parents=True)
    # Secret file outside allowed root
    (tmp_path / "secret.txt").write_text("credentials", encoding="utf-8")

    monkeypatch.setattr(bt, "project_root", lambda: tmp_path)
    monkeypatch.setattr(bt, "_ALLOWED_READ_ROOTS", (tmp_path / "data",))

    tool = bt.ReadFileTool()
    result = tool.execute(path="../../secret.txt")
    assert "[Error]" in result
    assert "credentials" not in result


# ── Step counter in ThunderAgentShell announcements ───────────────────────────


def test_step_counter_announced_from_iteration_2(workspace, tool_registry) -> None:
    """Iteration 2+ announces step number: '第2步，正在搜索网络...'"""
    import types
    import asyncio

    mock_audio = MagicMock()
    mock_audio.speak = MagicMock()

    call_count = [0]

    async def _two_tool_stream(messages, tools=None, tool_choice=None, model=None):
        call_count[0] += 1
        if call_count[0] == 1:
            # First LLM call: tool call iteration 1
            tc = types.SimpleNamespace(
                index=0, id="c1",
                function=types.SimpleNamespace(name="get_current_time", arguments="{}"),
            )
            yield types.SimpleNamespace(choices=[types.SimpleNamespace(
                delta=types.SimpleNamespace(tool_calls=[tc], content="")
            )])
        elif call_count[0] == 2:
            # Second LLM call: another tool (iteration 2)
            tc = types.SimpleNamespace(
                index=0, id="c2",
                function=types.SimpleNamespace(name="get_current_time", arguments="{}"),
            )
            yield types.SimpleNamespace(choices=[types.SimpleNamespace(
                delta=types.SimpleNamespace(tool_calls=[tc], content="")
            )])
        else:
            # Third call: final answer
            yield types.SimpleNamespace(choices=[types.SimpleNamespace(
                delta=types.SimpleNamespace(tool_calls=[], content="完成")
            )])

    llm = MagicMock()
    llm.chat_stream = _two_tool_stream

    shell = ThunderAgentShell(
        llm_client=llm,
        tool_registry=tool_registry,
        audio=mock_audio,
        workspace=workspace,
    )

    asyncio.run(shell.run_task("获取时间两次"))

    # Collect all speak() calls
    spoken_texts = [call.args[0] for call in mock_audio.speak.call_args_list]
    # At least one call should have a step prefix (iteration 2)
    step_calls = [t for t in spoken_texts if "第" in t and "步" in t]
    assert len(step_calls) >= 1, f"Expected step announcement, got: {spoken_texts}"
