"""Focused tests for BrainPipeline memory prompt assembly."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def _make_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "askme.brain.episodic_memory.project_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        "askme.brain.episodic_memory.get_config",
        lambda: {"app": {"data_dir": str(tmp_path / "data")}},
    )

    from askme.brain.episodic_memory import EpisodicMemory
    from askme.pipeline.brain_pipeline import BrainPipeline

    llm = AsyncMock()
    conversation = MagicMock()
    conversation.history = []
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

    pipeline = BrainPipeline(
        llm=llm,
        conversation=conversation,
        memory=memory,
        tools=tools,
        skill_manager=skill_manager,
        skill_executor=skill_executor,
        audio=audio,
        splitter=splitter,
        episodic_memory=EpisodicMemory(),
    )
    return pipeline


def test_system_prompt_includes_relevant_episodic_context(tmp_path, monkeypatch):
    pipeline = _make_pipeline(tmp_path, monkeypatch)
    pipeline._episodic.log("perception", "检测到猫在走廊")  # noqa: SLF001
    prompt = pipeline._build_system_prompt("", user_text="猫")
    assert "Relevant episodes" in prompt
    assert "猫" in prompt


def test_system_prompt_skips_empty_long_term_memory(tmp_path, monkeypatch):
    pipeline = _make_pipeline(tmp_path, monkeypatch)
    prompt = pipeline._build_system_prompt("", user_text="你好")
    assert "Relevant memory" not in prompt
