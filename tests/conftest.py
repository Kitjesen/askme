"""Shared pytest fixtures for askme tests."""

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

_THUNDER_AGENT_SHELL_REACT_LOOP_REPLACED_REASON = (
    "ThunderAgentShell ReAct loop was replaced by ZeroClaw MCP Agent."
)

_THUNDER_AGENT_SHELL_REACT_LOOP_TESTS = {
    "test_agent_executes_bash_command",
    "test_agent_handles_tool_error_gracefully",
    "test_agent_speaks_progress_during_task",
    "test_agent_timeout_returns_message",
    "test_agent_tool_result_passed_to_llm",
    "test_agent_writes_file_and_returns_answer",
    "test_simple_task_returns_response",
    "test_workspace_created",
    "test_tool_call_then_final_response",
    "test_run_task_persists_redacted_product_summary",
    "test_timeout_persists_run_summary",
    "test_tool_execution_error_handled",
    "test_tool_execution_timeout_returns_error",
    "test_pre_tool_hook_blocks_execution",
    "test_post_tool_hook_blocks_result",
    "test_tool_call_speaks_voice_label_and_updates_current_action",
    "test_max_iterations_stops_loop",
    "test_timeout_returns_gracefully",
    "test_run_task_returns_error_message_when_loop_raises",
    "test_context_passed_to_llm",
    "test_spawn_child_agent_depth_limit",
    "test_spawn_child_agent_empty_task",
    "test_spawn_child_agent_invalid_json",
    "test_spawn_child_agent_wraps_child_failure",
    "test_spawn_child_agent_child_is_silent_and_receives_context",
    "test_spawn_child_agent_runs_task",
    "test_step_counter_announced_from_iteration_2",
    "test_call_llm_retries_on_transient_error",
    "test_call_llm_raises_after_all_retries_exhausted",
    "test_call_llm_cancelled_error_not_retried",
}


def _pytest_marker_names_for_path(path: Path) -> set[str]:
    """Return automatically assigned pytest markers for known slow shards."""
    normalized = path.as_posix()
    markers: set[str] = set()

    if "/scenario_tests/" in f"/{normalized}":
        markers.update({"scenario", "slow"})

    filename = path.name
    if "e2e" in filename:
        markers.update({"e2e", "slow"})
    if "benchmark" in filename or filename.startswith("test_perf_"):
        markers.update({"benchmark", "slow"})

    return markers


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if (
            Path(str(item.fspath)).name in {"test_agent_task_e2e.py", "test_thunder_agent_shell.py"}
            and item.name in _THUNDER_AGENT_SHELL_REACT_LOOP_TESTS
        ):
            item.add_marker(
                pytest.mark.skip(reason=_THUNDER_AGENT_SHELL_REACT_LOOP_REPLACED_REASON)
            )
        for marker_name in sorted(_pytest_marker_names_for_path(Path(str(item.fspath)))):
            item.add_marker(getattr(pytest.mark, marker_name))


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch):
    """Set minimal environment variables so config.py doesn't fail."""
    monkeypatch.setenv("LLM_API_KEY", "sk-test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("LITELLM_BASE_URL", "http://127.0.0.1:4000/v1")
    monkeypatch.setenv("LITELLM_VIRTUAL_KEY", "sk-test-litellm-key")
    monkeypatch.setenv("ZEROCLAW_LITELLM_VIRTUAL_KEY", "sk-test-zeroclaw-key")
    monkeypatch.setenv("VISION_LITELLM_VIRTUAL_KEY", "sk-test-vision-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.example/v1")
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test-key")
    monkeypatch.setenv("MINIMAX_GROUP_ID", "0")
    monkeypatch.setenv("LOCAL_EMBED_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("TTS_VOICE_ID", "male-qn-qingse")
    monkeypatch.setenv("TTS_SPEED", "1")
    monkeypatch.setenv("TTS_EMOTION", "happy")


@pytest.fixture
def project_root() -> Path:
    """Return the askme project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def tmp_path(project_root: Path) -> Path:
    """Create a writable temp directory inside the repository workspace."""
    configured_base = os.environ.get("ASKME_PYTEST_TMPDIR", "").strip()
    base_dir = (
        Path(configured_base).expanduser().resolve()
        if configured_base
        else project_root / "data" / "pytest-tmp"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    # tempfile.mkdtemp() can create a non-inheritable ACL under the Windows
    # restricted-token sandbox. mkdir() inherits the writable-root ACL.
    for _ in range(100):
        path = base_dir / f"case-{uuid4().hex[:12]}"
        try:
            path.mkdir()
        except FileExistsError:
            continue
        break
    else:  # pragma: no cover - UUID collisions are practically unreachable.
        raise RuntimeError(f"Could not allocate a unique test directory in {base_dir}")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def make_proactive_orch():
    """Shared factory for ProactiveOrchestrator used across concurrency/stability tests.

    Uses pipeline=None so location slots fall back to trigger-stripping, avoiding
    MagicMock.extract_semantic_target() side-effects (len(MagicMock())==0 affecting
    is_vague() in ways that look correct but are accidental).
    """
    from askme.skills.skill_model import SkillDefinition, SlotSpec

    from askme.pipeline.proactive.orchestrator import ProactiveOrchestrator

    sk_search = SkillDefinition(
        name="web_search",
        voice_trigger="搜索",
        required_slots=[SlotSpec(name="query", type="text", prompt="搜什么？")],
    )
    sk_nav = SkillDefinition(
        name="navigate",
        voice_trigger="去",
        required_slots=[SlotSpec(name="destination", type="location", prompt="去哪里？")],
    )
    dispatcher = MagicMock()
    dispatcher.current_mission = None

    def _get(name):
        if name == "web_search":
            return sk_search
        if name == "navigate":
            return sk_nav
        return None

    dispatcher.get_skill.side_effect = _get
    return ProactiveOrchestrator.default(pipeline=None, dispatcher=dispatcher)


@pytest.fixture
def app_context():
    """Create a minimal AppContext for MCP tool unit tests."""
    from askme.mcp.server import AppContext

    return AppContext()
