"""Extended tests for record_external_turn edge cases."""

from __future__ import annotations

from askme.pipeline.external_turns import record_external_turn


class _FakeConversation:
    def __init__(self):
        self.user_msgs = []
        self.assistant_msgs = []

    def add_user_message(self, msg):
        self.user_msgs.append(msg)

    def add_assistant_message(self, msg):
        self.assistant_msgs.append(msg)


class _FakeEpisodic:
    def __init__(self):
        self.entries = []
        self._should_reflect = False

    def log(self, kind, content):
        self.entries.append((kind, content))

    def should_reflect(self):
        return self._should_reflect


class _FakePipeline:
    def __init__(self, conversation=None, episodic=None):
        self._conversation = conversation
        self._episodic = episodic


def test_empty_assistant_text_is_noop():
    conv = _FakeConversation()
    pipeline = _FakePipeline(conversation=conv)
    record_external_turn(pipeline, "user text", "")
    assert conv.user_msgs == []


def test_user_text_recorded():
    conv = _FakeConversation()
    epi = _FakeEpisodic()
    pipeline = _FakePipeline(conversation=conv, episodic=epi)
    record_external_turn(pipeline, "你好", "你好啊")
    assert conv.user_msgs == ["你好"]


def test_assistant_text_recorded():
    conv = _FakeConversation()
    pipeline = _FakePipeline(conversation=conv)
    record_external_turn(pipeline, "问题", "回答")
    assert conv.assistant_msgs == ["回答"]


def test_episodic_logged_twice():
    epi = _FakeEpisodic()
    pipeline = _FakePipeline(episodic=epi)
    record_external_turn(pipeline, "cmd", "response")
    assert len(epi.entries) == 2
    kinds = [e[0] for e in epi.entries]
    assert "command" in kinds
    assert "outcome" in kinds


def test_episodic_content_truncated_to_100():
    epi = _FakeEpisodic()
    pipeline = _FakePipeline(episodic=epi)
    long_text = "A" * 200
    record_external_turn(pipeline, "cmd", long_text)
    outcome = next(c for k, c in epi.entries if k == "outcome")
    # Should contain source + truncated text (100 chars)
    assert "A" * 100 in outcome
    assert "A" * 101 not in outcome


def test_no_conversation_no_crash():
    pipeline = _FakePipeline(conversation=None, episodic=_FakeEpisodic())
    record_external_turn(pipeline, "q", "a")  # should not raise


def test_no_episodic_no_crash():
    conv = _FakeConversation()
    pipeline = _FakePipeline(conversation=conv, episodic=None)
    record_external_turn(pipeline, "q", "a")  # should not raise


def test_no_pipeline_attributes_no_crash():
    class EmptyPipeline:
        pass
    record_external_turn(EmptyPipeline(), "q", "a")  # should not raise


def test_source_appears_in_episodic_entry():
    epi = _FakeEpisodic()
    pipeline = _FakePipeline(episodic=epi)
    record_external_turn(pipeline, "cmd", "reply", source="runtime")
    outcome = next(c for k, c in epi.entries if k == "outcome")
    assert "runtime" in outcome
