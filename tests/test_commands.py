"""Tests for CommandHandler — built-in slash command processing."""

from __future__ import annotations

from unittest.mock import MagicMock

from askme.pipeline.commands import CommandHandler


def _make_handler():
    conversation = MagicMock()
    conversation.history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    skill_manager = MagicMock()
    return CommandHandler(conversation=conversation, skill_manager=skill_manager)


class TestQuitCommands:
    def test_quit_returns_true(self):
        h = _make_handler()
        assert h.handle("/quit") is True

    def test_exit_returns_true(self):
        h = _make_handler()
        assert h.handle("/exit") is True

    def test_bare_quit_returns_true(self):
        h = _make_handler()
        assert h.handle("quit") is True

    def test_bare_exit_returns_true(self):
        h = _make_handler()
        assert h.handle("exit") is True

    def test_quit_commands_set_completeness(self):
        assert CommandHandler.QUIT_COMMANDS == {"/quit", "/exit", "exit", "quit"}


class TestClearCommand:
    def test_clear_calls_conversation_clear(self):
        h = _make_handler()
        result = h.handle("/clear")
        h._conversation.clear.assert_called_once()
        assert result is False

    def test_clear_returns_false(self):
        h = _make_handler()
        assert h.handle("/clear") is False


class TestHistoryCommand:
    def test_history_returns_false(self):
        h = _make_handler()
        assert h.handle("/history") is False


class TestSkillsCommand:
    def test_skills_calls_get_enabled(self):
        h = _make_handler()
        h._skill_manager.get_enabled.return_value = []
        h.handle("/skills")
        h._skill_manager.get_enabled.assert_called_once()

    def test_skills_returns_false(self):
        h = _make_handler()
        h._skill_manager.get_enabled.return_value = []
        assert h.handle("/skills") is False

    def test_skills_with_voice_trigger(self):
        h = _make_handler()
        skill = MagicMock()
        skill.name = "navigate"
        skill.description = "Go somewhere"
        skill.voice_trigger = "去那里"
        h._skill_manager.get_enabled.return_value = [skill]
        h.handle("/skills")  # should not raise

    def test_skills_without_voice_trigger(self):
        h = _make_handler()
        skill = MagicMock()
        skill.name = "find"
        skill.description = "Find object"
        skill.voice_trigger = None
        h._skill_manager.get_enabled.return_value = [skill]
        h.handle("/skills")  # should not raise


class TestHelpCommand:
    def test_help_returns_false(self):
        h = _make_handler()
        assert h.handle("/help") is False


class TestUnknownCommand:
    def test_unknown_command_returns_false(self):
        h = _make_handler()
        assert h.handle("/unknown") is False

    def test_unknown_command_does_not_call_clear(self):
        h = _make_handler()
        h.handle("/unknown")
        h._conversation.clear.assert_not_called()

    def test_all_commands_set_includes_standard(self):
        assert "/clear" in CommandHandler.ALL_COMMANDS
        assert "/help" in CommandHandler.ALL_COMMANDS
        assert "/history" in CommandHandler.ALL_COMMANDS
        assert "/skills" in CommandHandler.ALL_COMMANDS
        assert "/quit" in CommandHandler.ALL_COMMANDS
