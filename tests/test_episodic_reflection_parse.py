"""Regression tests for tolerant episodic reflection parsing."""

from __future__ import annotations

from askme.memory.core.episodic_memory import EpisodicMemory


def _memory(tmp_path) -> EpisodicMemory:
    return EpisodicMemory(
        llm=None,
        data_dir=tmp_path,
    )


def test_reflection_parser_repairs_trailing_commas(tmp_path):
    memory = _memory(tmp_path)
    parsed = memory._parse_reflection(
        '```json\n{"summary":"完成巡检", "new_facts":[], "patterns":[], "updates":[],}\n```'
    )

    assert parsed is not None
    assert parsed["summary"] == "完成巡检"


def test_reflection_parser_keeps_plain_text_as_safe_summary(tmp_path):
    memory = _memory(tmp_path)
    parsed = memory._parse_reflection("本轮完成了大厅巡检，没有发现异常。")

    assert parsed is not None
    assert parsed["summary"] == "本轮完成了大厅巡检，没有发现异常。"
    assert parsed["new_facts"] == []


def test_reflection_parser_rejects_placeholder_plain_text(tmp_path):
    memory = _memory(tmp_path)

    assert memory._parse_reflection("not json") is None


def test_reflection_parser_salvages_summary_from_truncated_json(tmp_path):
    memory = _memory(tmp_path)

    parsed = memory._parse_reflection(
        '{"summary":"本轮完成了大厅巡检，没有发现异常。","new_facts":['
    )

    assert parsed is not None
    assert parsed["summary"] == "本轮完成了大厅巡检，没有发现异常。"
    assert parsed["new_facts"] == []
