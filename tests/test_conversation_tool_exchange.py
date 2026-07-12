"""Integration tests for ConversationManager tool-exchange handling.

Covers:
- add_tool_exchange([], []) returns immediately without modifying history
- add_tool_exchange([tc], [tr]) adds both messages to history
- maybe_compress() skips when history has a pending tool_calls message
- _strip_orphan_tool_messages() removes orphan tool role messages
- _strip_orphan_tool_messages() keeps properly paired tool messages
- _strip_orphan_tool_messages() removes degenerate empty assistant messages
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

# ── Fixture helpers ────────────────────────────────────────────────


def _make_conv(tmp_path, monkeypatch, max_history: int = 40):
    """Construct a ConversationManager backed by tmp_path, no side-effects."""
    monkeypatch.setattr(
        "askme.llm.conversation.project_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        "askme.llm.conversation.get_config",
        lambda: {
            "conversation": {
                "history_file": str(tmp_path / "data" / "conv.json"),
                "max_history": max_history,
            }
        },
    )
    from askme.llm.conversation import ConversationManager

    return ConversationManager()


def _tool_call(call_id: str = "call_1", name: str = "dispatch_skill") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def _tool_result(call_id: str = "call_1", content: str = "结果") -> dict:
    return {"tool_call_id": call_id, "content": content}


# ── add_tool_exchange ──────────────────────────────────────────────


class TestAddToolExchange:
    def test_empty_calls_returns_immediately(self, tmp_path, monkeypatch):
        """add_tool_exchange([], []) must not modify history."""
        conv = _make_conv(tmp_path, monkeypatch)
        conv.add_user_message("你好")
        original_len = len(conv.history)
        conv.add_tool_exchange([], [])
        assert len(conv.history) == original_len

    def test_empty_calls_no_messages_appended(self, tmp_path, monkeypatch):
        """An empty tool_calls list means nothing is added."""
        conv = _make_conv(tmp_path, monkeypatch)
        conv.add_tool_exchange([], [{"tool_call_id": "x", "content": "orphan"}])
        assert conv.history == []

    def test_single_call_and_result_added(self, tmp_path, monkeypatch):
        """One tool call + one result → two messages appended."""
        conv = _make_conv(tmp_path, monkeypatch)
        tc = _tool_call("call_1")
        tr = _tool_result("call_1", "导航已启动")
        conv.add_tool_exchange([tc], [tr])
        assert len(conv.history) == 2
        assert conv.history[0]["role"] == "assistant"
        assert conv.history[0].get("tool_calls") == [tc]
        assert conv.history[1]["role"] == "tool"
        assert conv.history[1]["tool_call_id"] == "call_1"
        assert conv.history[1]["content"] == "导航已启动"

    def test_assistant_message_has_none_content(self, tmp_path, monkeypatch):
        """The assistant tool-call message must have content=None."""
        conv = _make_conv(tmp_path, monkeypatch)
        conv.add_tool_exchange([_tool_call()], [_tool_result()])
        assert conv.history[0]["content"] is None

    def test_multiple_results_all_appended(self, tmp_path, monkeypatch):
        """Multiple tool results are each added as separate tool messages."""
        conv = _make_conv(tmp_path, monkeypatch)
        tc1 = _tool_call("call_1", "skill_a")
        tc2 = _tool_call("call_2", "skill_b")
        tr1 = _tool_result("call_1", "A完成")
        tr2 = _tool_result("call_2", "B完成")
        conv.add_tool_exchange([tc1, tc2], [tr1, tr2])
        # 1 assistant message + 2 tool result messages
        assert len(conv.history) == 3
        roles = [m["role"] for m in conv.history]
        assert roles == ["assistant", "tool", "tool"]

    def test_tool_exchange_is_scoped_to_conversation_session(self, tmp_path, monkeypatch):
        """Tool messages for one session are not included in another prompt."""
        conv = _make_conv(tmp_path, monkeypatch)
        conv.add_user_message("a user", conversation_session_id="a")
        conv.add_tool_exchange(
            [_tool_call("call_a")],
            [_tool_result("call_a", "a result")],
            conversation_session_id="a",
        )
        conv.add_user_message("b user", conversation_session_id="b")

        session_a_roles = [
            m["role"] for m in conv.get_messages("sys", conversation_session_id="a")
        ]
        session_b_roles = [
            m["role"] for m in conv.get_messages("sys", conversation_session_id="b")
        ]

        assert session_a_roles == ["system", "user", "assistant", "tool"]
        assert session_b_roles == ["system", "user"]

    def test_last_assistant_metadata_is_scoped_to_conversation_session(
        self,
        tmp_path,
        monkeypatch,
    ):
        conv = _make_conv(tmp_path, monkeypatch)
        conv.add_assistant_message("answer a", conversation_session_id="a")
        conv.add_assistant_message("answer b", conversation_session_id="b")

        updated = conv.update_last_assistant_metadata(
            {"evidence": [{"record_id": "rec-a"}]},
            conversation_session_id="a",
        )

        assert updated is True
        session_a = conv.get_messages("sys", conversation_session_id="a")
        session_b = conv.get_messages("sys", conversation_session_id="b")
        assert session_a[-1]["evidence"] == [{"record_id": "rec-a"}]
        assert "evidence" not in session_b[-1]

    def test_no_save_called_for_tool_exchange(self, tmp_path, monkeypatch):
        """Tool exchanges are transient — the history file must not be rewritten."""
        conv = _make_conv(tmp_path, monkeypatch)
        history_file = tmp_path / "data" / "conv.json"
        # Baseline: add a regular message so the file exists
        conv.add_user_message("baseline")
        mtime_before = history_file.stat().st_mtime

        conv.add_tool_exchange([_tool_call()], [_tool_result()])
        mtime_after = history_file.stat().st_mtime

        assert mtime_before == mtime_after, (
            "History file must not be modified by add_tool_exchange()"
        )


# ── _trim strips transient tool exchanges ──────────────────────────


class TestTrimStripsToolExchange:
    def test_regular_turn_strips_tool_exchange_from_persisted_history(
        self,
        tmp_path,
        monkeypatch,
    ):
        """The next regular assistant message removes transient tool context."""
        conv = _make_conv(tmp_path, monkeypatch)
        conv.add_user_message("请查天气")
        conv.add_tool_exchange(
            [_tool_call("call_weather", "weather")],
            [_tool_result("call_weather", "晴")],
        )

        assert [m["role"] for m in conv.history] == ["user", "assistant", "tool"]

        conv.add_assistant_message("天气晴朗。")

        assert conv.history == [
            {"role": "user", "content": "请查天气"},
            {"role": "assistant", "content": "天气晴朗。"},
        ]

        history_file = tmp_path / "data" / "conv.json"
        with history_file.open(encoding="utf-8") as fh:
            persisted = json.load(fh)
        assert persisted == conv.history

    def test_max_history_applies_after_tool_exchange_is_stripped(
        self,
        tmp_path,
        monkeypatch,
    ):
        """max_history counts only durable user/assistant text messages."""
        conv = _make_conv(tmp_path, monkeypatch, max_history=4)
        conv.add_user_message("u0")
        conv.add_assistant_message("a0")
        conv.add_user_message("u1")
        conv.add_tool_exchange([_tool_call()], [_tool_result()])
        conv.add_assistant_message("a1")
        conv.add_user_message("u2")
        conv.add_assistant_message("a2")

        assert conv.history == [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
        ]

    async def test_session_memory_receives_only_dropped_text_messages(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Dropped tool context is not forwarded to SessionMemory summaries."""
        conv = _make_conv(tmp_path, monkeypatch, max_history=2)
        session_memory = AsyncMock()
        conv._session_memory = session_memory
        conv.history = [
            {"role": "user", "content": "old user"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c1")],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "tool result"},
            {"role": "assistant", "content": "old assistant"},
            {"role": "user", "content": "new user"},
            {"role": "assistant", "content": "new assistant"},
        ]

        conv._trim()
        await asyncio.sleep(0)

        assert conv.history == [
            {"role": "user", "content": "new user"},
            {"role": "assistant", "content": "new assistant"},
        ]
        session_memory.summarize_and_save.assert_awaited_once_with([
            {"role": "user", "content": "old user"},
            {"role": "assistant", "content": "old assistant"},
        ])


# ── maybe_compress with pending tool_calls ─────────────────────────


class TestMaybeCompressSkipsOnPendingToolCalls:
    async def test_skips_when_tool_calls_in_history(self, tmp_path, monkeypatch):
        """maybe_compress() must not run while a tool exchange is in flight."""
        monkeypatch.setattr(
            "askme.llm.conversation.project_root", lambda: tmp_path
        )
        monkeypatch.setattr(
            "askme.llm.conversation.get_config",
            lambda: {
                "conversation": {
                    "history_file": str(tmp_path / "data" / "conv.json"),
                    "max_history": 100,
                }
            },
        )
        # Lower threshold so the history would normally trigger compression
        monkeypatch.setattr("askme.llm.conversation.COMPRESS_THRESHOLD", 4)
        monkeypatch.setattr("askme.llm.conversation.KEEP_RECENT", 2)

        from askme.llm.conversation import ConversationManager

        conv = ConversationManager()
        # Add enough messages to exceed the threshold
        for i in range(5):
            conv.history.append({"role": "user", "content": f"msg {i}"})
            conv.history.append({"role": "assistant", "content": f"reply {i}"})

        # Inject a pending tool_calls message to simulate mid-exchange state
        conv.history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call()],
        })

        mock_llm = AsyncMock()
        await conv.maybe_compress(mock_llm)

        # LLM must not have been called — compression is suppressed
        mock_llm.chat.assert_not_called()

    async def test_does_not_skip_without_tool_calls(self, tmp_path, monkeypatch):
        """maybe_compress() proceeds normally when there are no pending tool calls."""
        monkeypatch.setattr(
            "askme.llm.conversation.project_root", lambda: tmp_path
        )
        monkeypatch.setattr(
            "askme.llm.conversation.get_config",
            lambda: {
                "conversation": {
                    "history_file": str(tmp_path / "data" / "conv.json"),
                    "max_history": 100,
                }
            },
        )
        monkeypatch.setattr("askme.llm.conversation.COMPRESS_THRESHOLD", 4)
        monkeypatch.setattr("askme.llm.conversation.KEEP_RECENT", 2)

        from askme.llm.conversation import ConversationManager

        conv = ConversationManager()
        for i in range(5):
            conv.history.append({"role": "user", "content": f"msg {i}"})
            conv.history.append({"role": "assistant", "content": f"reply {i}"})

        mock_llm = AsyncMock()
        mock_llm.chat.return_value = "摘要内容"
        await conv.maybe_compress(mock_llm)

        mock_llm.chat.assert_called_once()


# ── _strip_orphan_tool_messages ────────────────────────────────────


class TestStripOrphanToolMessages:
    """Static method tests — no ConversationManager state needed."""

    def _strip(self, history: list[dict]) -> list[dict]:
        from askme.llm.conversation import ConversationManager

        return ConversationManager._strip_orphan_tool_messages(history)

    def test_empty_history_returns_empty(self):
        assert self._strip([]) == []

    def test_removes_orphan_tool_message_at_start(self):
        """A tool message with no preceding assistant+tool_calls is dropped."""
        history = [
            {"role": "tool", "tool_call_id": "c1", "content": "orphan"},
            {"role": "user", "content": "hello"},
        ]
        result = self._strip(history)
        roles = [m["role"] for m in result]
        assert "tool" not in roles
        assert "user" in roles

    def test_removes_orphan_tool_after_plain_assistant(self):
        """A tool message preceded by a normal (non-tool_calls) assistant message is orphaned."""
        history = [
            {"role": "assistant", "content": "普通回复"},
            {"role": "tool", "tool_call_id": "c1", "content": "孤立结果"},
        ]
        result = self._strip(history)
        roles = [m["role"] for m in result]
        assert "tool" not in roles

    def test_keeps_paired_tool_message(self):
        """A tool message immediately after an assistant+tool_calls message is kept."""
        history = [
            {"role": "user", "content": "请执行"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c1")],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "结果"},
        ]
        result = self._strip(history)
        roles = [m["role"] for m in result]
        assert roles == ["user", "assistant", "tool"]

    def test_keeps_multiple_paired_tool_messages(self):
        """All tool results from the same assistant tool_calls turn are kept.

        When an assistant requests multiple tool calls, consecutive tool result
        messages must all be preserved — not just the first one.
        """
        history = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c1"), _tool_call("c2")],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "R1"},
            {"role": "tool", "tool_call_id": "c2", "content": "R2"},
        ]
        result = self._strip(history)
        assert len(result) == 3
        assert result[0]["role"] == "assistant"
        assert result[1]["tool_call_id"] == "c1"
        assert result[2]["tool_call_id"] == "c2"

    def test_removes_degenerate_empty_assistant_message(self):
        """An assistant message with no content and no tool_calls is dropped."""
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant"},          # no content, no tool_calls
            {"role": "assistant", "content": "正常回复"},
        ]
        result = self._strip(history)
        roles = [m["role"] for m in result]
        # The degenerate assistant entry is removed
        contents = [m.get("content") for m in result if m["role"] == "assistant"]
        assert "正常回复" in contents
        # The empty one should not appear — it has no content
        assert any(c is None or c == "" or c == "正常回复" for c in contents)
        # Exact check: only "正常回复" among assistant messages
        assert all(c == "正常回复" for c in contents if c is not None)

    def test_removes_degenerate_empty_content_and_no_tool_calls(self):
        """assistant with content=None and no tool_calls key is dropped."""
        history = [
            {"role": "assistant", "content": None},
            {"role": "user", "content": "after"},
        ]
        result = self._strip(history)
        roles = [m["role"] for m in result]
        # The degenerate assistant (content=None, no tool_calls) must be removed
        assert "assistant" not in roles

    def test_normal_user_assistant_exchange_preserved(self):
        """Ordinary user/assistant pairs pass through unchanged."""
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
            {"role": "user", "content": "再见"},
            {"role": "assistant", "content": "再见！"},
        ]
        result = self._strip(history)
        assert result == history

    def test_second_orphan_after_valid_pair_removed(self):
        """After a valid pair, a subsequent orphan tool message is still removed."""
        history = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("c1")],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "R1"},
            {"role": "user", "content": "继续"},
            {"role": "tool", "tool_call_id": "c_orphan", "content": "孤立"},
        ]
        result = self._strip(history)
        tool_msgs = [m for m in result if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "c1"
