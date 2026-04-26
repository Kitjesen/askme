"""Tests for PromptBuilder — system prompt assembly and message preparation."""

from __future__ import annotations

from unittest.mock import MagicMock

from askme.pipeline.prompt_builder import PromptBuilder

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_builder(
    *,
    base_prompt: str = "You are a robot assistant.",
    prompt_seed: list[dict] | None = None,
    user_prefix: str = "",
    tools=None,
    skill_manager=None,
    dog_safety=None,
    episodic=None,
    session_memory=None,
    vision=None,
    qp_memory=None,
    memory_system=None,
    general_tool_max_safety_level: str = "normal",
) -> PromptBuilder:
    if tools is None:
        tools = MagicMock()
        tools.get_definitions.return_value = []
    if skill_manager is None:
        skill_manager = MagicMock()
        skill_manager.get_skill_catalog.return_value = "none"
    return PromptBuilder(
        base_prompt=base_prompt,
        prompt_seed=prompt_seed or [],
        user_prefix=user_prefix,
        tools=tools,
        skill_manager=skill_manager,
        general_tool_max_safety_level=general_tool_max_safety_level,
        dog_safety=dog_safety,
        episodic=episodic,
        session_memory=session_memory,
        vision=vision,
        qp_memory=qp_memory,
        memory_system=memory_system,
    )


# ── TestBuildL0RuntimeBlock ───────────────────────────────────────────────────

class TestBuildL0RuntimeBlock:
    def test_no_dog_safety_returns_empty(self):
        pb = _make_builder(dog_safety=None)
        assert pb.build_l0_runtime_block() == ""

    def test_unconfigured_dog_safety_returns_empty(self):
        dog_safety = MagicMock()
        dog_safety.is_configured.return_value = False
        pb = _make_builder(dog_safety=dog_safety)
        assert pb.build_l0_runtime_block() == ""

    def test_configured_safety_returns_block(self):
        dog_safety = MagicMock()
        dog_safety.is_configured.return_value = True
        dog_safety.is_estop_active.return_value = False
        pb = _make_builder(dog_safety=dog_safety)
        block = pb.build_l0_runtime_block()
        assert block != ""
        assert "运行时状态" in block

    def test_estop_active_shows_warning(self):
        dog_safety = MagicMock()
        dog_safety.is_configured.return_value = True
        dog_safety.is_estop_active.return_value = True
        pb = _make_builder(dog_safety=dog_safety)
        block = pb.build_l0_runtime_block()
        assert "已激活" in block or "⚠" in block

    def test_estop_inactive_shows_normal(self):
        dog_safety = MagicMock()
        dog_safety.is_configured.return_value = True
        dog_safety.is_estop_active.return_value = False
        pb = _make_builder(dog_safety=dog_safety)
        block = pb.build_l0_runtime_block()
        assert "正常" in block


# ── TestBuildSystemPrompt ─────────────────────────────────────────────────────

class TestBuildSystemPrompt:
    def test_contains_base_prompt(self):
        pb = _make_builder(base_prompt="Base instructions here.")
        result = pb.build_system_prompt(None)
        assert "Base instructions here." in result

    def test_none_context_str_not_in_prompt(self):
        pb = _make_builder()
        result = pb.build_system_prompt(None)
        assert "Relevant memory" not in result

    def test_context_str_appended(self):
        pb = _make_builder()
        result = pb.build_system_prompt("Some memory context")
        assert "Some memory context" in result
        assert "Relevant memory" in result

    def test_episodic_knowledge_included(self):
        episodic = MagicMock()
        episodic.get_knowledge_context.return_value = "知识上下文"
        episodic.get_recent_digest.return_value = ""
        episodic.get_relevant_context.return_value = ""
        pb = _make_builder(episodic=episodic)
        result = pb.build_system_prompt(None)
        assert "知识上下文" in result

    def test_episodic_recent_digest_included(self):
        episodic = MagicMock()
        episodic.get_knowledge_context.return_value = ""
        episodic.get_recent_digest.return_value = "最近事件摘要"
        episodic.get_relevant_context.return_value = ""
        pb = _make_builder(episodic=episodic)
        result = pb.build_system_prompt(None)
        assert "最近事件摘要" in result

    def test_episodic_relevant_context_included(self):
        episodic = MagicMock()
        episodic.get_knowledge_context.return_value = ""
        episodic.get_recent_digest.return_value = ""
        episodic.get_relevant_context.return_value = "相关记忆"
        pb = _make_builder(episodic=episodic)
        result = pb.build_system_prompt(None, user_text="查询")
        assert "相关记忆" in result

    def test_session_memory_summaries_included(self):
        session_memory = MagicMock()
        session_memory.get_recent_summaries.return_value = "Session context here"
        pb = _make_builder(session_memory=session_memory)
        result = pb.build_system_prompt(None)
        assert "Session context here" in result

    def test_empty_session_memory_not_added(self):
        session_memory = MagicMock()
        session_memory.get_recent_summaries.return_value = ""
        pb = _make_builder(base_prompt="BASE", session_memory=session_memory)
        result = pb.build_system_prompt(None)
        # Should only contain base prompt + nothing extra from session
        assert "BASE" in result

    def test_memory_system_trends_included(self):
        memory_system = MagicMock()
        memory_system.get_trends.return_value = "Trend data"
        pb = _make_builder(memory_system=memory_system)
        result = pb.build_system_prompt(None)
        assert "Trend data" in result
        assert "趋势" in result

    def test_memory_system_trend_error_silenced(self):
        memory_system = MagicMock()
        memory_system.get_trends.side_effect = RuntimeError("broken")
        pb = _make_builder(memory_system=memory_system)
        result = pb.build_system_prompt(None)  # should not raise
        assert isinstance(result, str)

    def test_qp_memory_context_included(self):
        qp_memory = MagicMock()
        qp_memory.get_context_smart.return_value = "站点信息"
        pb = _make_builder(qp_memory=qp_memory)
        result = pb.build_system_prompt(None, user_text="查询地图")
        assert "站点信息" in result
        assert "站点记忆" in result

    def test_qp_memory_not_injected_when_already_present(self):
        qp_memory = MagicMock()
        qp_memory.get_context_smart.return_value = "已有站点记忆"
        pb = _make_builder(qp_memory=qp_memory)
        # user_text already contains [站点记忆] — should not inject again
        result = pb.build_system_prompt(None, user_text="查询 [站点记忆] 信息")
        qp_memory.get_context_smart.assert_not_called()

    def test_qp_memory_error_silenced(self):
        qp_memory = MagicMock()
        qp_memory.get_context_smart.side_effect = RuntimeError("fail")
        pb = _make_builder(qp_memory=qp_memory)
        result = pb.build_system_prompt(None)  # should not raise
        assert isinstance(result, str)

    def test_vision_available_adds_vision_line(self):
        vision = MagicMock()
        vision.available = True
        pb = _make_builder(vision=vision)
        result = pb.build_system_prompt(None)
        assert "视觉能力" in result

    def test_vision_scene_desc_included(self):
        vision = MagicMock()
        vision.available = True
        pb = _make_builder(vision=vision)
        result = pb.build_system_prompt(None, scene_desc="走廊一片空旷")
        assert "走廊一片空旷" in result

    def test_vision_unavailable_not_added(self):
        vision = MagicMock()
        vision.available = False
        pb = _make_builder(base_prompt="BASE", vision=vision)
        result = pb.build_system_prompt(None)
        assert "视觉能力" not in result

    def test_tool_names_listed_in_prompt(self):
        tools = MagicMock()
        tools.get_definitions.return_value = [
            {"function": {"name": "get_time"}},
            {"function": {"name": "list_files"}},
        ]
        pb = _make_builder(tools=tools)
        result = pb.build_system_prompt(None)
        assert "get_time" in result
        assert "list_files" in result

    def test_no_tools_no_tool_line(self):
        tools = MagicMock()
        tools.get_definitions.return_value = []
        pb = _make_builder(base_prompt="BASE", tools=tools)
        result = pb.build_system_prompt(None)
        assert "工具" not in result

    def test_skill_catalog_included_when_not_none(self):
        skill_manager = MagicMock()
        skill_manager.get_skill_catalog.return_value = "patrol, navigate"
        pb = _make_builder(skill_manager=skill_manager)
        result = pb.build_system_prompt(None)
        assert "patrol" in result
        assert "可用技能" in result

    def test_skill_catalog_none_not_included(self):
        skill_manager = MagicMock()
        skill_manager.get_skill_catalog.return_value = "none"
        pb = _make_builder(base_prompt="BASE", skill_manager=skill_manager)
        result = pb.build_system_prompt(None)
        assert "可用技能" not in result

    def test_l0_block_prepended_to_base_prompt(self):
        dog_safety = MagicMock()
        dog_safety.is_configured.return_value = True
        dog_safety.is_estop_active.return_value = False
        pb = _make_builder(base_prompt="BASE_PROMPT", dog_safety=dog_safety)
        result = pb.build_system_prompt(None)
        l0_pos = result.find("运行时状态")
        base_pos = result.find("BASE_PROMPT")
        assert l0_pos < base_pos  # L0 block comes before base prompt


# ── TestPrepareMessages ───────────────────────────────────────────────────────

class TestPrepareMessages:
    def _msgs(self, user_content: str = "hello") -> list[dict]:
        return [
            {"role": "system", "content": "System message."},
            {"role": "user", "content": user_content},
        ]

    def test_no_seed_no_prefix_returns_unchanged(self):
        pb = _make_builder(prompt_seed=[], user_prefix="")
        msgs = self._msgs()
        result = pb.prepare_messages(msgs)
        assert result == msgs

    def test_user_prefix_prepended_to_last_user_message(self):
        pb = _make_builder(prompt_seed=[], user_prefix="[TTS]")
        msgs = self._msgs("ask me something")
        result = pb.prepare_messages(msgs)
        last_user = next(m for m in reversed(result) if m["role"] == "user")
        assert last_user["content"].startswith("[TTS]\n")
        assert "ask me something" in last_user["content"]

    def test_original_messages_not_mutated(self):
        pb = _make_builder(prompt_seed=[], user_prefix="[PREFIX]")
        msgs = self._msgs("original")
        original_user = msgs[1]["content"]
        pb.prepare_messages(msgs)
        assert msgs[1]["content"] == original_user  # original unchanged

    def test_with_seed_system_message_dropped(self):
        seed = [{"role": "user", "content": "You are a robot."}, {"role": "assistant", "content": "Understood."}]
        pb = _make_builder(prompt_seed=seed)
        msgs = self._msgs("hello")
        result = pb.prepare_messages(msgs)
        roles = [m["role"] for m in result]
        assert "system" not in roles

    def test_with_seed_seed_messages_prepended(self):
        seed = [{"role": "user", "content": "SEED_USER"}, {"role": "assistant", "content": "SEED_ASST"}]
        pb = _make_builder(prompt_seed=seed)
        msgs = self._msgs("hello")
        result = pb.prepare_messages(msgs)
        assert result[0]["content"] == "SEED_USER"
        assert result[1]["content"] == "SEED_ASST"

    def test_with_seed_and_tools_injects_tool_exchange(self):
        seed = [{"role": "user", "content": "SEED"}, {"role": "assistant", "content": "OK"}]
        tools = MagicMock()
        tools.get_definitions.return_value = [{"function": {"name": "get_time"}}]
        pb = _make_builder(prompt_seed=seed, tools=tools)
        msgs = self._msgs("hello")
        result = pb.prepare_messages(msgs)
        contents = " ".join(m.get("content", "") for m in result)
        assert "get_time" in contents

    def test_with_seed_no_tools_no_extra_exchange(self):
        seed = [{"role": "user", "content": "SEED"}, {"role": "assistant", "content": "OK"}]
        tools = MagicMock()
        tools.get_definitions.return_value = []
        pb = _make_builder(prompt_seed=seed, tools=tools)
        msgs = self._msgs("hello")
        result = pb.prepare_messages(msgs)
        # Seed + original (minus system)
        assert len(result) == len(seed) + 1  # seed + 1 user msg

    def test_empty_messages_with_seed(self):
        seed = [{"role": "user", "content": "SEED"}, {"role": "assistant", "content": "OK"}]
        pb = _make_builder(prompt_seed=seed)
        result = pb.prepare_messages([])
        # Should not raise; returns seed messages (plus possibly tool exchange)
        assert isinstance(result, list)

    def test_user_prefix_applied_only_to_last_user(self):
        pb = _make_builder(prompt_seed=[], user_prefix="[TAG]")
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first user"},
            {"role": "assistant", "content": "asst response"},
            {"role": "user", "content": "second user"},
        ]
        result = pb.prepare_messages(msgs)
        # Only last user message should have prefix
        user_msgs = [m for m in result if m["role"] == "user"]
        assert user_msgs[0]["content"] == "first user"
        assert user_msgs[1]["content"].startswith("[TAG]\n")
