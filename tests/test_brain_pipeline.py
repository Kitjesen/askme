"""Tests for BrainPipeline — vision integration, system prompt assembly."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def _make_pipeline(
    tmp_path,
    monkeypatch,
    *,
    vision_desc: str = "",
    vision_available: bool = False,
    turn_ledger=None,
):
    """Build a BrainPipeline with mocked dependencies."""
    monkeypatch.setattr("askme.memory.episodic_memory.project_root", lambda: tmp_path)
    monkeypatch.setattr(
        "askme.memory.episodic_memory.get_config",
        lambda: {"app": {"data_dir": str(tmp_path / "data")}},
    )

    from askme.memory.episodic_memory import EpisodicMemory
    from askme.pipeline.brain_pipeline import BrainPipeline

    # Mocks
    llm = AsyncMock()
    conversation = MagicMock()
    conversation.history = []
    conversation.get_messages.return_value = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "hello"},
    ]
    memory = AsyncMock()
    memory.retrieve = AsyncMock(return_value="")
    memory.health = MagicMock(return_value={})
    tools = MagicMock()
    tools.get_definitions.return_value = []
    skill_manager = MagicMock()
    skill_manager.get_skill_catalog.return_value = "none"
    skill_executor = MagicMock()
    audio = MagicMock()
    splitter = MagicMock()
    splitter.reset.return_value = None
    splitter.feed.return_value = []
    splitter.flush.return_value = None

    # Vision mock
    vision = MagicMock()
    vision.available = vision_available
    vision.describe_scene = AsyncMock(return_value=vision_desc)

    episodic = EpisodicMemory()

    # Mock LLM streaming: return a simple response
    async def fake_stream(messages, **kwargs):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = "回复内容"
        chunk.choices[0].delta.tool_calls = None
        yield chunk

    llm.chat_stream = fake_stream

    pipeline = BrainPipeline(
        llm=llm,
        conversation=conversation,
        memory=memory,
        tools=tools,
        skill_manager=skill_manager,
        skill_executor=skill_executor,
        audio=audio,
        splitter=splitter,
        vision=vision,
        episodic_memory=episodic,
        turn_ledger=turn_ledger,
    )

    return pipeline, episodic, vision


async def test_execute_skill_commits_one_canonical_turn_without_reprojecting_legacy_history(
    tmp_path,
    monkeypatch,
):
    from askme.conversation import TurnStatus

    ledger = MagicMock()
    thread = MagicMock(
        thread_id="thread-direct-skill",
        channel="voice",
        person_id=None,
        operator_id=None,
        robot_id=None,
        site_id=None,
    )
    turn = MagicMock(
        turn_id="turn-direct-skill",
        thread_id="thread-direct-skill",
        source="text",
        user_text="巡检 A 区",
        metadata={},
        status=TurnStatus.STARTED,
    )
    ledger.resolve_thread.return_value = thread
    ledger.start_turn.return_value = turn
    ledger.get_turn.return_value = turn
    ledger.commit_turn.return_value = MagicMock(status=TurnStatus.COMMITTED)
    monkeypatch.setattr("askme.memory.episodic_memory.EpisodicMemory", MagicMock)
    pipeline, _, _ = _make_pipeline(
        tmp_path,
        monkeypatch,
        turn_ledger=ledger,
    )
    conversation = pipeline._conversation

    async def execute_skill(
        skill_name,
        user_text,
        extra_context="",
        source="voice",
        conversation_session_id=None,
        voice_turn_id=None,
        turn_cancel_token=None,
    ):
        del skill_name, extra_context, source, voice_turn_id, turn_cancel_token
        conversation.add_user_message(
            user_text,
            conversation_session_id=conversation_session_id,
        )
        conversation.add_assistant_message(
            "巡检完成",
            conversation_session_id=conversation_session_id,
        )
        return "巡检完成"

    skill_gate = MagicMock()
    skill_gate.execute_skill = AsyncMock(side_effect=execute_skill)
    pipeline.set_skill_gate(skill_gate)

    result = await pipeline.execute_skill(
        "patrol",
        "巡检 A 区",
        source="text",
        conversation_session_id="thread-direct-skill",
        voice_turn_id="turn-direct-skill",
    )

    assert result == "巡检完成"
    assert ledger.resolve_thread.call_args.kwargs["channel"] == "text"
    ledger.start_turn.assert_called_once()
    ledger.commit_turn.assert_called_once_with(
        "turn-direct-skill",
        user_text=None,
        assistant_text="巡检完成",
        heard_text="巡检完成",
        played_ms=None,
        metadata={"skill_name": "patrol"},
    )
    conversation.add_user_message.assert_called_once()
    conversation.add_assistant_message.assert_called_once()


async def test_execute_skill_fails_canonical_turn_for_internal_error_result(
    tmp_path,
    monkeypatch,
):
    from askme.conversation import TurnStatus

    monkeypatch.setattr("askme.memory.episodic_memory.EpisodicMemory", MagicMock)
    pipeline, _, _ = _make_pipeline(tmp_path, monkeypatch)
    interaction = MagicMock()
    manager = MagicMock()
    manager.open.return_value = interaction
    pipeline._interaction_turns = manager
    skill_gate = MagicMock()
    skill_gate.execute_skill = AsyncMock(return_value="[Skill Error] boom")
    pipeline.set_skill_gate(skill_gate)

    result = await pipeline.execute_skill(
        "patrol",
        "巡检 A 区",
        source="text",
        conversation_session_id="thread-failed-skill",
        voice_turn_id="turn-failed-skill",
    )

    assert result == "[Skill Error] boom"
    manager.open.assert_called_once()
    manager.settle.assert_called_once()
    settled_interaction, outcome = manager.settle.call_args.args
    assert settled_interaction is interaction
    assert outcome.status is TurnStatus.FAILED
    assert outcome.reason == "internal_error_result"
    assert outcome.metadata == {"skill_name": "patrol"}


async def test_execute_skill_legacy_gate_keeps_bracketed_customer_result(
    tmp_path,
    monkeypatch,
):
    from askme.conversation import TurnStatus

    monkeypatch.setattr("askme.memory.episodic_memory.EpisodicMemory", MagicMock)
    pipeline, _, _ = _make_pipeline(tmp_path, monkeypatch)
    interaction = MagicMock()
    manager = MagicMock()
    manager.open.return_value = interaction
    pipeline._interaction_turns = manager
    skill_gate = MagicMock()
    skill_gate.execute_skill = AsyncMock(return_value="[1] 巡检完成")
    pipeline.set_skill_gate(skill_gate)

    result = await pipeline.execute_skill(
        "patrol",
        "巡检 A 区",
        source="text",
        conversation_session_id="thread-bracket-result",
        voice_turn_id="turn-bracket-result",
    )

    assert result == "[1] 巡检完成"
    settled_interaction, outcome = manager.settle.call_args.args
    assert settled_interaction is interaction
    assert outcome.status is TurnStatus.COMMITTED


async def test_execute_skill_uses_injected_gate_disposition_for_turn_settlement(
    tmp_path,
    monkeypatch,
):
    from askme.conversation import TurnStatus
    from askme.pipeline.core.protocols import SkillExecutionDisposition

    monkeypatch.setattr("askme.memory.episodic_memory.EpisodicMemory", MagicMock)
    pipeline, _, _ = _make_pipeline(tmp_path, monkeypatch)
    interaction = MagicMock()
    manager = MagicMock()
    manager.open.return_value = interaction
    pipeline._interaction_turns = manager
    skill_gate = MagicMock()
    skill_gate.execute_skill = AsyncMock(return_value="execution stopped")
    skill_gate.classify_execution_result = MagicMock(
        return_value=SkillExecutionDisposition(
            status="cancelled",
            code="operator_cancelled",
        )
    )
    pipeline.set_skill_gate(skill_gate)

    result = await pipeline.execute_skill(
        "patrol",
        "停止巡检",
        source="text",
        conversation_session_id="thread-cancelled-result",
        voice_turn_id="turn-cancelled-result",
    )

    assert result == "execution stopped"
    skill_gate.classify_execution_result.assert_called_once_with(
        "execution stopped",
        skill_name="patrol",
    )
    settled_interaction, outcome = manager.settle.call_args.args
    assert settled_interaction is interaction
    assert outcome.status is TurnStatus.CANCELLED
    assert outcome.reason == "operator_cancelled"


async def test_execute_skill_cancels_canonical_turn_before_execution(
    tmp_path,
    monkeypatch,
):
    from askme.conversation import TurnStatus

    monkeypatch.setattr("askme.memory.episodic_memory.EpisodicMemory", MagicMock)
    pipeline, _, _ = _make_pipeline(tmp_path, monkeypatch)
    interaction = MagicMock()
    manager = MagicMock()
    manager.open.return_value = interaction
    pipeline._interaction_turns = manager
    skill_gate = MagicMock()
    skill_gate.execute_skill = AsyncMock()
    pipeline.set_skill_gate(skill_gate)
    cancel_token = MagicMock()
    cancel_token.is_set.return_value = True

    result = await pipeline.execute_skill(
        "patrol",
        "巡检 A 区",
        source="text",
        conversation_session_id="thread-cancelled-skill",
        voice_turn_id="turn-cancelled-skill",
        turn_cancel_token=cancel_token,
    )

    assert result == ""
    skill_gate.execute_skill.assert_not_awaited()
    manager.settle.assert_called_once()
    settled_interaction, outcome = manager.settle.call_args.args
    assert settled_interaction is interaction
    assert outcome.status is TurnStatus.CANCELLED
    assert outcome.reason == "cancelled_before_skill_execution"


async def test_vision_scene_logged_to_episodic(tmp_path, monkeypatch):
    """When vision returns a scene description, it gets logged to episodic memory."""
    pipeline, episodic, vision = _make_pipeline(
        tmp_path,
        monkeypatch,
        vision_desc="我看到了: 2个cup, 1个person",
        vision_available=True,
    )

    await pipeline.process("你好")

    # Should have at least 3 episodes: perception + command + action
    recent = episodic.get_recent(10)
    types = [ep.event_type for ep in recent]
    assert "perception" in types

    # The perception episode should contain the scene description
    perception_eps = [ep for ep in recent if ep.event_type == "perception"]
    assert len(perception_eps) >= 1
    assert "cup" in perception_eps[0].description


async def test_no_vision_no_perception_log(tmp_path, monkeypatch):
    """When vision is unavailable, no perception episode is logged."""
    pipeline, episodic, vision = _make_pipeline(
        tmp_path,
        monkeypatch,
        vision_available=False,
    )

    await pipeline.process("你好")

    recent = episodic.get_recent(10)
    types = [ep.event_type for ep in recent]
    assert "perception" not in types


async def test_vision_empty_scene_no_log(tmp_path, monkeypatch):
    """When vision returns empty string, no perception episode is logged."""
    pipeline, episodic, vision = _make_pipeline(
        tmp_path,
        monkeypatch,
        vision_desc="",
        vision_available=True,
    )

    await pipeline.process("你好")

    recent = episodic.get_recent(10)
    types = [ep.event_type for ep in recent]
    assert "perception" not in types


async def test_scene_description_in_system_prompt(tmp_path, monkeypatch):
    """Scene description appears in the system prompt when vision is active."""
    pipeline, episodic, vision = _make_pipeline(
        tmp_path,
        monkeypatch,
        vision_desc="我看到了: 1个bottle",
        vision_available=True,
    )

    prompt = pipeline._build_system_prompt("", scene_desc="我看到了: 1个bottle")
    assert "当前视野" in prompt
    assert "bottle" in prompt


async def test_no_scene_no_vision_in_prompt(tmp_path, monkeypatch):
    """Without vision, no 当前视野 section in system prompt."""
    pipeline, episodic, vision = _make_pipeline(
        tmp_path,
        monkeypatch,
        vision_available=False,
    )

    prompt = pipeline._build_system_prompt("")
    assert "当前视野" not in prompt
