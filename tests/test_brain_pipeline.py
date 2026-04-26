"""Tests for BrainPipeline — vision integration, system prompt assembly."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def _make_pipeline(
    tmp_path,
    monkeypatch,
    *,
    vision_desc: str = "",
    vision_available: bool = False,
):
    """Build a BrainPipeline with mocked dependencies."""
    monkeypatch.setattr(
        "askme.memory.episodic_memory.project_root", lambda: tmp_path
    )
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
    )

    return pipeline, episodic, vision


async def test_vision_scene_logged_to_episodic(tmp_path, monkeypatch):
    """When vision returns a scene description, it gets logged to episodic memory."""
    pipeline, episodic, vision = _make_pipeline(
        tmp_path, monkeypatch,
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
        tmp_path, monkeypatch,
        vision_available=False,
    )

    await pipeline.process("你好")

    recent = episodic.get_recent(10)
    types = [ep.event_type for ep in recent]
    assert "perception" not in types


async def test_vision_empty_scene_no_log(tmp_path, monkeypatch):
    """When vision returns empty string, no perception episode is logged."""
    pipeline, episodic, vision = _make_pipeline(
        tmp_path, monkeypatch,
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
        tmp_path, monkeypatch,
        vision_desc="我看到了: 1个bottle",
        vision_available=True,
    )

    prompt = pipeline._build_system_prompt("", scene_desc="我看到了: 1个bottle")
    assert "当前视野" in prompt
    assert "bottle" in prompt


async def test_no_scene_no_vision_in_prompt(tmp_path, monkeypatch):
    """Without vision, no 当前视野 section in system prompt."""
    pipeline, episodic, vision = _make_pipeline(
        tmp_path, monkeypatch,
        vision_available=False,
    )

    prompt = pipeline._build_system_prompt("")
    assert "当前视野" not in prompt
