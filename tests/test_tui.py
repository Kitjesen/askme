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


def _fake_profile():
    return SimpleNamespace(
        name="text",
        snapshot=lambda: {"name": "text", "primary_loop": "text"},
    )


def _fake_conversation(history=None):
    if history is None:
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "我在。"},
        ]
    return SimpleNamespace(
        history=history,
        clear=lambda: history.clear(),
    )


def _fake_modules(*, conversation=None, skill_manager=None, pipeline=None):
    """Build a modules dict that mimics RuntimeApp.modules."""
    if conversation is None:
        conversation = _fake_conversation()
    if skill_manager is None:
        skill_manager = SimpleNamespace(
            get_enabled=lambda: [],
            get_all=lambda: [],
            get_contracts=lambda: [],
            get_contract_catalog=lambda: [],
            openapi_document=lambda: {"info": {"title": "", "version": ""}, "paths": {}},
        )

    llm_mod = SimpleNamespace(
        client=SimpleNamespace(model="test-model"),
        ota_metrics=SimpleNamespace(
            snapshot=lambda: {"voice_pipeline": {}},
        ),
        health=lambda: {"status": "ok", "model": "test-model"},
        capabilities=lambda: {},
    )
    mem_mod = SimpleNamespace(
        conversation=conversation,
        health=lambda: {"status": "ok", "conversation_len": len(conversation.history)},
        capabilities=lambda: {},
    )
    skill_mod = SimpleNamespace(
        skill_manager=skill_manager,
        skill_dispatcher=SimpleNamespace(current_mission=None),
        health=lambda: {"status": "ok", "skill_count": 0, "enabled_count": 0},
        capabilities=lambda: {},
    )
    pipeline_mod = SimpleNamespace(
        brain_pipeline=pipeline or SimpleNamespace(handle_estop=lambda: None, _audio=None),
        health=lambda: {"status": "ok"},
        capabilities=lambda: {},
    )
    safety_mod = SimpleNamespace(
        client=None,
        health=lambda: {"status": "ok"},
        capabilities=lambda: {},
    )
    text_mod = SimpleNamespace(
        text_loop=SimpleNamespace(process_turn=lambda text: text),
        health=lambda: {"status": "ok"},
        capabilities=lambda: {},
    )
    tools_mod = SimpleNamespace(
        health=lambda: {"status": "ok"},
        capabilities=lambda: {},
    )

    return {
        "llm": llm_mod,
        "memory": mem_mod,
        "skill": skill_mod,
        "pipeline": pipeline_mod,
        "safety": safety_mod,
        "text": text_mod,
        "tools": tools_mod,
    }


class _FakeApp:
    """Mimics RuntimeApp with a modules dict."""

    def __init__(self, modules=None) -> None:
        self.modules = modules or _fake_modules()

    def get(self, name):
        return self.modules.get(name)

    def health(self):
        return {name: mod.health() for name, mod in self.modules.items()}


def _make_ui(app=None, profile=None):
    """Build a TUI with fake app and profile."""
    if app is None:
        app = _FakeApp()
    if profile is None:
        profile = _fake_profile()
    return AskmeTerminalUI(app, profile)


# ── Rendering ────────────────────────────────────────────────────────────────


def test_tui_render_text_includes_history_and_status() -> None:
    ui = _make_ui()
    rendered = ui.render_text(width=100, height=24)
    assert "THUNDER" in rendered
    assert "你好" in rendered
    assert "我在" in rendered
    assert "memory" in rendered


def test_tui_render_has_color_codes() -> None:
    ui = _make_ui()
    rendered = ui.render_text(width=100, height=24)
    assert "\x1b[" in rendered  # ANSI escape present


def test_tui_render_shows_component_health_dots() -> None:
    ui = _make_ui()
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
    ui = _make_ui()
    assert await ui._handle_input("/quit") is True
    assert await ui._handle_input("/exit") is True
    assert await ui._handle_input("quit") is True


@pytest.mark.asyncio
async def test_handle_input_help() -> None:
    ui = _make_ui()
    result = await ui._handle_input("/help")
    assert result is False
    assert any("help" in e.content.lower() for e in ui._entries)


@pytest.mark.asyncio
async def test_handle_input_clear() -> None:
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "我在。"},
    ]
    conversation = _fake_conversation(history)
    modules = _fake_modules(conversation=conversation)
    app = _FakeApp(modules)
    ui = _make_ui(app)
    result = await ui._handle_input("/clear")
    assert result is False
    assert len(ui._entries) == 1  # only the "cleared" message


@pytest.mark.asyncio
async def test_handle_input_skills() -> None:
    ui = _make_ui()
    result = await ui._handle_input("/skills")
    assert result is False


@pytest.mark.asyncio
async def test_handle_input_status() -> None:
    ui = _make_ui()
    result = await ui._handle_input("/status")
    assert result is False
    # Status command should produce a system entry with status info
    status_entries = [e for e in ui._entries if "状态=" in e.content]
    assert len(status_entries) >= 1


# ── History Sync ─────────────────────────────────────────────────────────────


def test_sync_history_growth() -> None:
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "我在。"},
    ]
    conversation = _fake_conversation(history)
    modules = _fake_modules(conversation=conversation)
    app = _FakeApp(modules)
    ui = _make_ui(app)
    initial = len(ui._entries)
    history.append({"role": "user", "content": "新消息"})
    ui._sync_history()
    assert len(ui._entries) == initial + 1


def test_sync_history_shrink() -> None:
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "我在。"},
    ]
    conversation = _fake_conversation(history)
    modules = _fake_modules(conversation=conversation)
    app = _FakeApp(modules)
    ui = _make_ui(app)
    history.clear()
    ui._sync_history()
    # entries rebuilt from empty history + welcome message stays
    assert ui._known_history_len == 0


# ── ESTOP Handler ────────────────────────────────────────────────────────────


def test_estop_handler_adds_entry() -> None:
    pipeline = SimpleNamespace(handle_estop=lambda: None, _audio=None)
    modules = _fake_modules(pipeline=pipeline)
    app = _FakeApp(modules)
    ui = _make_ui(app)
    ui._handle_estop()
    assert any("ESTOP" in e.content for e in ui._entries)
    assert ui._status_text == "ESTOP"
