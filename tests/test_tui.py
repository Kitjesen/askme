"""Tests for the askme terminal UI."""

from __future__ import annotations

from types import SimpleNamespace

from askme.tui import AskmeTerminalUI


class _FakeApp:
    def __init__(self) -> None:
        self.profile = SimpleNamespace(name="text")
        self.audio = SimpleNamespace()
        self.conversation = SimpleNamespace(
            history=[
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "我在。"},
            ]
        )
        self.skill_manager = SimpleNamespace(get_enabled=lambda: [])

    def capabilities_snapshot(self) -> dict:
        return {
            "app": {
                "name": "askme",
                "version": "4.1.0",
                "voice_mode": False,
                "robot_mode": False,
            },
            "profile": {"name": "text", "primary_loop": "text"},
            "components": {
                "memory": {"health": {"status": "ok"}},
                "skill_runtime": {"health": {"status": "ok"}},
            },
            "skills": {
                "count": 2,
                "enabled_count": 2,
                "catalog": [
                    {"name": "navigate", "enabled": True},
                    {"name": "find_object", "enabled": True},
                ],
            },
        }

    def health_snapshot(self) -> dict:
        return {
            "status": "ok",
            "total_conversations": 3,
            "last_llm_latency_ms": 245.0,
            "active_skill_count": 2,
            "model_name": "test-model",
        }


def test_tui_render_text_includes_history_and_status() -> None:
    ui = AskmeTerminalUI(_FakeApp())

    rendered = ui.render_text(width=100, height=24)

    assert "profile=text" in rendered
    assert "[你] 你好" in rendered
    assert "[askme] 我在。" in rendered
    assert "状态" in rendered
    assert "memory: ok" in rendered
