"""
Call-chain integration tests for voice CLI capabilities.

Tests three things end-to-end (no running server needed, LLM mocked):
  1. Voice trigger routing: "帮我查一下文件" → agent_task
  2. create_skill: agent creates a new skill → hot-reload → new trigger active in router
  3. agent_task dispatches through SkillManager to list_directory skill
"""
from __future__ import annotations

from pathlib import Path

import pytest

from askme.llm.intent_router import IntentRouter, IntentType
from askme.skills.skill_manager import SkillManager
from askme.tools.skill_tools import CreateSkillTool

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def skill_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SkillManager:
    """SkillManager with real builtin skills loaded.

    generated_skills_dir is redirected to tmp_path/skills so tests
    don't pollute the real data/skills/ directory and don't pick up
    artifacts left by previous runs.
    """
    import askme.skills.skill_manager as _sm

    # Patch module-level _DATA_DIR so generated_skills_dir property returns tmp_path/skills
    monkeypatch.setattr(_sm, "_DATA_DIR", tmp_path)
    generated_dir = tmp_path / "skills"
    generated_dir.mkdir(exist_ok=True)
    mgr = SkillManager(project_dir=tmp_path)
    mgr.load()
    return mgr


@pytest.fixture()
def router(skill_manager: SkillManager) -> IntentRouter:
    """IntentRouter wired with all loaded voice triggers."""
    return IntentRouter(voice_triggers=skill_manager.get_voice_triggers())


# ── 1. Voice trigger routing ──────────────────────────────────────────────────

class TestVoiceTriggerRouting:
    """Verify IntentRouter dispatches file/search phrases to the right skills."""

    AGENT_TASK_PHRASES = [
        # Use phrases that are exact triggers in agent_task SKILL.md and do NOT
        # overlap with web_search triggers ("帮我搜索" is shared — avoid it).
        "帮我查一下",
        "研究一下",
        "帮我写一段代码",
        "做个自动化",
        "帮我分析",
        "写个脚本",
        "帮我跑",
    ]

    def test_agent_task_triggers_load(self, skill_manager: SkillManager) -> None:
        """agent_task skill must be discoverable with a non-empty trigger set."""
        triggers = skill_manager.get_voice_triggers()
        agent_task_triggers = [p for p, s in triggers.items() if s == "agent_task"]
        assert len(agent_task_triggers) >= 10, (
            f"Expected ≥10 agent_task triggers, got {len(agent_task_triggers)}: "
            f"{agent_task_triggers}"
        )

    def test_list_directory_trigger_loads(self, skill_manager: SkillManager) -> None:
        """list_directory skill must have at least one voice trigger."""
        triggers = skill_manager.get_voice_triggers()
        ld_triggers = [p for p, s in triggers.items() if s == "list_directory"]
        assert len(ld_triggers) >= 1, (
            f"Expected ≥1 list_directory trigger, got {ld_triggers}"
        )

    @pytest.mark.parametrize("phrase", AGENT_TASK_PHRASES)
    def test_agent_task_routing(self, router: IntentRouter, phrase: str) -> None:
        """File/search voice phrases must route to agent_task, not GENERAL."""
        intent = router.route(phrase)
        assert intent.type == IntentType.VOICE_TRIGGER, (
            f"'{phrase}' routed to {intent.type} instead of VOICE_TRIGGER"
        )
        assert intent.skill_name == "agent_task", (
            f"'{phrase}' routed to skill '{intent.skill_name}' instead of 'agent_task'"
        )

    def test_list_directory_routing(self, router: IntentRouter) -> None:
        """'看看文件' must route to list_directory."""
        intent = router.route("看看文件")
        assert intent.type == IntentType.VOICE_TRIGGER
        assert intent.skill_name == "list_directory"

    def test_estop_handled_safely(self, router: IntentRouter) -> None:
        """'紧急停止' must be handled by a safety path — either ESTOP intent
        (when safety_checker present) or the robot_estop skill (voice trigger).
        Either route is acceptable; the key is it must NOT fall to GENERAL.
        """
        intent = router.route("紧急停止")
        assert intent.type in (IntentType.ESTOP, IntentType.VOICE_TRIGGER), (
            f"'紧急停止' fell through to {intent.type} — safety path broken"
        )
        # If routed as voice trigger, it must land on the estop skill
        if intent.type == IntentType.VOICE_TRIGGER:
            assert intent.skill_name == "robot_estop", (
                f"Expected robot_estop skill, got '{intent.skill_name}'"
            )

    def test_general_fallback_for_unknown(self, router: IntentRouter) -> None:
        """Unknown phrases must fall through to GENERAL (LLM) intent."""
        intent = router.route("今天天气怎么样")
        assert intent.type == IntentType.GENERAL


# ── 2. create_skill → hot-reload → new trigger active ────────────────────────

class TestCreateSkillCallChain:
    """Full create_skill call chain: tool writes SKILL.md → hot-reload → router updated."""

    def test_create_skill_writes_file_and_hot_reloads(
        self, skill_manager: SkillManager, router: IntentRouter, tmp_path: Path
    ) -> None:
        """CreateSkillTool creates a skill file and updates router triggers."""
        # Wire tool with real SkillManager + router
        tool = CreateSkillTool()
        tool.set_context(skill_manager, router)

        # Use a trigger phrase unique enough to not clash with any built-in skill
        _UNIQUE_TRIGGER = "查一下测试电量情况xyz"

        # Before: new trigger doesn't exist
        triggers_before = set(router._voice_triggers.keys())
        assert _UNIQUE_TRIGGER not in triggers_before

        # Execute create_skill
        result = tool.execute(
            name="check_battery",
            description="查看Thunder机器人当前电量和健康状态",
            voice_trigger=_UNIQUE_TRIGGER,
            prompt=(
                "用户想查看机器人电量。调用 robot_api 工具，"
                "service=telemetry, method=GET, path=/api/v1/health，"
                "解析 battery_percent 字段并口语化告知用户。"
            ),
            tools_section="robot_api",
            tags="robot,sensor",
        )

        # Tool should succeed
        assert "[Error]" not in result, f"create_skill failed: {result}"
        assert "check_battery" in result
        assert "热加载" in result or "已创建" in result

        # Skill file must exist on disk
        skill_file = skill_manager.generated_skills_dir / "check_battery" / "SKILL.md"
        assert skill_file.exists(), f"SKILL.md not found at {skill_file}"
        content = skill_file.read_text(encoding="utf-8")
        assert "check_battery" in content
        assert _UNIQUE_TRIGGER in content

        # Router must now recognise the new trigger
        assert _UNIQUE_TRIGGER in router._voice_triggers, (
            "Router not updated after hot-reload. "
            f"Current triggers: {list(router._voice_triggers.keys())[:10]}"
        )
        assert router._voice_triggers[_UNIQUE_TRIGGER] == "check_battery"

    def test_new_skill_is_routable_immediately(
        self, skill_manager: SkillManager, router: IntentRouter, tmp_path: Path
    ) -> None:
        """After create_skill, the router can dispatch the new trigger without restart."""
        tool = CreateSkillTool()
        tool.set_context(skill_manager, router)
        tool.execute(
            name="greet_visitor",
            description="向访客打招呼",
            voice_trigger="打个招呼,你好访客,欢迎光临",
            prompt="向用户友好地打招呼，用简短的中文口语。",
        )

        # All new triggers must route to the new skill
        for phrase in ["打个招呼", "你好访客", "欢迎光临"]:
            intent = router.route(phrase)
            assert intent.type == IntentType.VOICE_TRIGGER, (
                f"'{phrase}' should match new skill but got {intent.type}"
            )
            assert intent.skill_name == "greet_visitor"

    def test_create_skill_invalid_name_returns_error(
        self, skill_manager: SkillManager, router: IntentRouter
    ) -> None:
        """Blank skill name must return an error string, not raise."""
        tool = CreateSkillTool()
        tool.set_context(skill_manager, router)
        result = tool.execute(name="", description="test", prompt="test")
        assert "[Error]" in result

    def test_create_skill_without_context_returns_error(self) -> None:
        """Calling create_skill without set_context must return an error string."""
        tool = CreateSkillTool()
        result = tool.execute(name="foo", description="bar", prompt="baz")
        assert "[Error]" in result


# ── 3. SkillManager hot_reload propagates to router ──────────────────────────

class TestHotReload:
    """Verify hot_reload correctly rebuilds trigger index after adding a skill."""

    def test_hot_reload_increments_skill_count(
        self, skill_manager: SkillManager, router: IntentRouter, tmp_path: Path
    ) -> None:
        initial_count = len(skill_manager.get_all())
        assert initial_count >= 1  # built-in skills exist

        # Write a minimal SKILL.md to generated_skills_dir
        new_dir = skill_manager.generated_skills_dir / "dummy_skill"
        new_dir.mkdir(parents=True, exist_ok=True)
        (new_dir / "SKILL.md").write_text(
            "---\n"
            "name: dummy_skill\n"
            "description: dummy\n"
            "version: 1.0.0\n"
            "trigger: voice\n"
            "model: \"\"\n"
            "timeout: 30\n"
            "tags: [test]\n"
            "depends: []\n"
            "conflicts: []\n"
            "safety_level: normal\n"
            "voice_trigger: 测试一下,跑个测试\n"
            "---\n\n## Prompt\n\n测试技能。\n",
            encoding="utf-8",
        )

        n = skill_manager.hot_reload(router)
        assert n > initial_count, (
            f"Expected skill count to increase from {initial_count}, got {n}"
        )

        # Router must know the new triggers
        assert "测试一下" in router._voice_triggers
        assert router._voice_triggers["测试一下"] == "dummy_skill"
