"""Tests for the askme terminal UI."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from askme.tui import (
    AskmeTerminalUI,
    SilentAudio,
    _display_width,
    _pad_to_width,
    _truncate_to_width,
)


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


# ── Rendering ────────────────────────────────────────────────────────────────


def test_tui_render_text_includes_history_and_status() -> None:
    ui = AskmeTerminalUI(_FakeApp())
    rendered = ui.render_text(width=100, height=24)
    assert "THUNDER" in rendered
    assert "你好" in rendered
    assert "我在" in rendered
    assert "memory" in rendered


def test_tui_render_has_color_codes() -> None:
    ui = AskmeTerminalUI(_FakeApp())
    rendered = ui.render_text(width=100, height=24)
    assert "\x1b[" in rendered  # ANSI escape present


def test_tui_render_shows_component_health_dots() -> None:
    ui = AskmeTerminalUI(_FakeApp())
    rendered = ui.render_text(width=100, height=24)
    assert "●" in rendered


# ── CJK Width ────────────────────────────────────────────────────────────────


def test_display_width_ascii() -> None:
    assert _display_width("hello") == 5


def test_display_width_cjk() -> None:
    assert _display_width("你好") == 4  # 2 chars x 2 width


def test_display_width_mixed() -> None:
    assert _display_width("hi你好") == 6  # 2 ascii + 2 CJK


def test_pad_to_width() -> None:
    padded = _pad_to_width("你好", 8)
    assert _display_width(padded) == 8


def test_truncate_to_width_no_truncation() -> None:
    assert _truncate_to_width("hello", 10) == "hello"


def test_truncate_to_width_with_truncation() -> None:
    result = _truncate_to_width("hello world", 8)
    assert result.endswith("...")
    assert _display_width(result) <= 8


def test_truncate_to_width_cjk() -> None:
    result = _truncate_to_width("你好世界测试", 8)
    assert _display_width(result) <= 8


# ── SilentAudio ──────────────────────────────────────────────────────────────


def test_silent_audio_sync_methods() -> None:
    sa = SilentAudio()
    sa.speak("hello")
    sa.start_playback()
    sa.stop_playback()
    sa.wait_speaking_done()
    sa.drain_buffers()
    sa.acknowledge()
    sa.play_thinking()


@pytest.mark.asyncio
async def test_silent_audio_async_methods() -> None:
    sa = SilentAudio()
    await sa.speak_and_wait("test")


def test_silent_audio_properties() -> None:
    sa = SilentAudio()
    assert sa.awaiting_confirmation is False
    assert sa.is_muted is False


# ── Slash Commands ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_handle_input_quit() -> None:
    ui = AskmeTerminalUI(_FakeApp())
    assert await ui._handle_input("/quit") is True
    assert await ui._handle_input("/exit") is True
    assert await ui._handle_input("quit") is True


@pytest.mark.asyncio
async def test_handle_input_help() -> None:
    ui = AskmeTerminalUI(_FakeApp())
    result = await ui._handle_input("/help")
    assert result is False
    assert any("help" in e.content.lower() for e in ui._entries)


@pytest.mark.asyncio
async def test_handle_input_clear() -> None:
    app = _FakeApp()
    app.conversation.clear = lambda: None
    ui = AskmeTerminalUI(app)
    initial_entries = len(ui._entries)
    result = await ui._handle_input("/clear")
    assert result is False
    assert len(ui._entries) == 1  # only the "cleared" message


@pytest.mark.asyncio
async def test_handle_input_skills() -> None:
    ui = AskmeTerminalUI(_FakeApp())
    result = await ui._handle_input("/skills")
    assert result is False


@pytest.mark.asyncio
async def test_handle_input_status() -> None:
    ui = AskmeTerminalUI(_FakeApp())
    result = await ui._handle_input("/status")
    assert result is False
    assert any("ok" in e.content for e in ui._entries)


# ── History Sync ─────────────────────────────────────────────────────────────


def test_sync_history_growth() -> None:
    app = _FakeApp()
    ui = AskmeTerminalUI(app)
    initial = len(ui._entries)
    app.conversation.history.append({"role": "user", "content": "新消息"})
    ui._sync_history()
    assert len(ui._entries) == initial + 1


def test_sync_history_shrink() -> None:
    app = _FakeApp()
    ui = AskmeTerminalUI(app)
    app.conversation.history.clear()
    ui._sync_history()
    # entries rebuilt from empty history + welcome message stays
    assert ui._known_history_len == 0


# ── ESTOP Handler ────────────────────────────────────────────────────────────


def test_estop_handler_adds_entry() -> None:
    app = _FakeApp()
    app.pipeline = SimpleNamespace(handle_estop=lambda: None)
    ui = AskmeTerminalUI(app)
    ui._handle_estop()
    assert any("ESTOP" in e.content for e in ui._entries)
    assert ui._status_text == "ESTOP"
