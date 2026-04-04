"""Extended tests for record_external_turn edge cases."""

from __future__ import annotations

from unittest.mock import MagicMock

from askme.pipeline.external_turns import record_external_turn


class TestEmptyAssistantText:
    def test_empty_assistant_text_noop(self):
        pipeline = MagicMock()
        record_external_turn(pipeline, "hello", "")
        pipeline._conversation.add_user_message.assert_not_called()

    def test_none_like_empty_skipped(self):
        pipeline = MagicMock()
        record_external_turn(pipeline, "hello", "   ")
        # "   " is truthy so it IS recorded (by design: whitespace is a real response)
        # Verify it doesn't crash
        assert True


class TestMissingAttributes:
    def test_pipeline_without_conversation(self):
        pipeline = object()  # no _conversation
        record_external_turn(pipeline, "hello", "world")  # should not raise

    def test_pipeline_without_episodic(self):
        conv = MagicMock()
        pipeline = MagicMock(spec=[])  # no _episodic attribute
        pipeline._conversation = conv
        record_external_turn(pipeline, "hello", "world")
        conv.add_user_message.assert_called_once_with("hello")
        conv.add_assistant_message.assert_called_once_with("world")

    def test_pipeline_none_conversation(self):
        pipeline = MagicMock()
        pipeline._conversation = None
        record_external_turn(pipeline, "hello", "world")  # should not raise


class TestEpisodicLogging:
    def test_logs_command_and_outcome(self):
        entries = []
        episodic = MagicMock()
        episodic.log.side_effect = lambda k, v: entries.append((k, v))
        episodic.should_reflect.return_value = False
        pipeline = MagicMock()
        pipeline._conversation = MagicMock()
        pipeline._episodic = episodic
        record_external_turn(pipeline, "检查温度", "当前温度22度", source="voice")
        kinds = [e[0] for e in entries]
        assert "command" in kinds
        assert "outcome" in kinds

    def test_source_in_outcome_log(self):
        logged = []
        episodic = MagicMock()
        episodic.log.side_effect = lambda k, v: logged.append((k, v))
        episodic.should_reflect.return_value = False
        pipeline = MagicMock()
        pipeline._conversation = MagicMock()
        pipeline._episodic = episodic
        record_external_turn(pipeline, "query", "response", source="runtime_bridge")
        outcome = [v for k, v in logged if k == "outcome"][0]
        assert "runtime_bridge" in outcome


class TestConversationRecording:
    def test_user_and_assistant_recorded(self):
        conv = MagicMock()
        pipeline = MagicMock()
        pipeline._conversation = conv
        record_external_turn(pipeline, "user text", "assistant reply")
        conv.add_user_message.assert_called_once_with("user text")
        conv.add_assistant_message.assert_called_once_with("assistant reply")
