from __future__ import annotations

from askme.cognition import WorkingMemory
from askme.cognition.memory import working_memory as working_memory_module


def test_record_turn_keeps_session_and_turn_metadata() -> None:
    memory = WorkingMemory(retention_seconds=60)

    memory.record_turn(
        "inspect area A",
        assistant_text="starting draft",
        observations=["door is open"],
        task_id="task-1",
        conversation_session_id="conv-a",
        turn_id="turn-1",
    )

    snapshot = memory.snapshot()

    assert snapshot["item_count"] == 3
    assert {item["session_id"] for item in snapshot["items"]} == {"conv-a"}
    assert {item["conversation_session_id"] for item in snapshot["items"]} == {"conv-a"}
    assert {item["turn_id"] for item in snapshot["items"]} == {"turn-1"}


def test_select_context_filters_by_session_task_kind_and_tags() -> None:
    memory = WorkingMemory(retention_seconds=60)
    memory.record(
        "observation",
        "session A task observation",
        salience=0.8,
        task_id="task-a",
        conversation_session_id="session-a",
        tags=("scene", "urgent"),
    )
    memory.record(
        "observation",
        "session B task observation",
        salience=0.9,
        task_id="task-a",
        session_id="session-b",
        tags=("scene", "urgent"),
    )
    memory.record(
        "operator_utterance",
        "session A different kind",
        salience=0.95,
        task_id="task-a",
        conversation_session_id="session-a",
        tags=("dialog",),
    )

    context = memory.select_context(
        kinds=("observation",),
        tags=("scene",),
        task_id="task-a",
        conversation_session_id="session-a",
        include_focus=False,
    )

    assert context["item_count"] == 1
    assert context["items"][0]["content"] == "session A task observation"


def test_select_context_applies_item_and_char_budgets() -> None:
    memory = WorkingMemory(retention_seconds=60)
    memory.record("note", "alpha " * 10, salience=0.9, session_id="session-a")
    memory.record("note", "beta", salience=0.8, session_id="session-a")
    memory.record("note", "gamma", salience=0.7, session_id="session-a")

    item_limited = memory.select_context(
        max_items=2,
        conversation_session_id="session-a",
        include_focus=False,
    )
    char_limited = memory.select_context(
        max_chars=20,
        conversation_session_id="session-a",
        include_focus=False,
    )

    assert item_limited["item_count"] == 2
    assert len(char_limited["text"]) <= 20
    assert char_limited["items"][0]["content"] != "alpha " * 10


def test_prune_removes_expired_items_before_retrieval(monkeypatch) -> None:
    clock = {"now": 1000.0}
    monkeypatch.setattr(
        working_memory_module.time,
        "time",
        lambda: clock["now"],
    )
    memory = WorkingMemory(retention_seconds=60)

    memory.record("note", "expired context", conversation_session_id="session-a", ttl_s=1)
    memory.record("note", "fresh context", conversation_session_id="session-a", ttl_s=10)
    clock["now"] = 1002.0

    snapshot = memory.snapshot()
    context = memory.select_context(conversation_session_id="session-a")
    candidates = memory.promote_candidates(
        conversation_session_id="session-a",
        min_salience=0.0,
    )

    assert [item["content"] for item in snapshot["items"]] == ["fresh context"]
    assert [item["content"] for item in context["items"]] == ["fresh context"]
    assert [item["content"] for item in candidates] == ["fresh context"]


def test_prune_keeps_other_sessions_and_their_focus(monkeypatch) -> None:
    clock = {"now": 1000.0}
    monkeypatch.setattr(
        working_memory_module.time,
        "time",
        lambda: clock["now"],
    )
    memory = WorkingMemory(retention_seconds=60)
    memory.set_focus(conversation_session_id="session-a", route="A")
    memory.set_focus(conversation_session_id="session-b", route="B")

    memory.record("note", "session A expired", conversation_session_id="session-a", ttl_s=1)
    memory.record("note", "session B fresh", conversation_session_id="session-b", ttl_s=10)
    clock["now"] = 1002.0

    context_a = memory.select_context(conversation_session_id="session-a")
    context_b = memory.select_context(conversation_session_id="session-b")
    snapshot = memory.snapshot()

    assert context_a["items"] == []
    assert context_a["focus"]["route"] == "A"
    assert [item["content"] for item in context_b["items"]] == ["session B fresh"]
    assert context_b["focus"]["route"] == "B"
    assert snapshot["item_count"] == 1
    assert snapshot["session_focus"]["session-a"]["route"] == "A"
    assert snapshot["session_focus"]["session-b"]["route"] == "B"


def test_record_respects_max_items_without_leaking_evicted_session_context() -> None:
    memory = WorkingMemory(max_items=2, retention_seconds=60)

    memory.record("note", "old session item", conversation_session_id="session-old")
    memory.record("note", "new session item one", conversation_session_id="session-new")
    memory.record("note", "new session item two", conversation_session_id="session-new")

    old_context = memory.select_context(conversation_session_id="session-old")
    new_context = memory.select_context(conversation_session_id="session-new")

    assert old_context["items"] == []
    assert [item["content"] for item in new_context["items"]] == [
        "new session item two",
        "new session item one",
    ]


def test_focus_is_scoped_by_conversation_session() -> None:
    memory = WorkingMemory(retention_seconds=60)

    memory.set_focus(conversation_session_id="session-a", route="A")
    memory.set_focus(conversation_session_id="session-b", route="B")

    context_a = memory.select_context(conversation_session_id="session-a")
    context_b = memory.select_context(conversation_session_id="session-b")
    default_context = memory.select_context()
    snapshot = memory.snapshot()

    assert context_a["focus"]["route"] == "A"
    assert context_b["focus"]["route"] == "B"
    assert default_context["focus"] == {}
    assert "route=A" in context_a["text"]
    assert "route=B" not in context_a["text"]
    assert "route=A" not in default_context["text"]
    assert "route=B" not in default_context["text"]
    assert snapshot["focus"] == {}
    assert snapshot["session_focus"]["session-a"]["route"] == "A"

    removed = memory.clear_session(conversation_session_id="session-a")

    assert removed == 0
    assert "session-a" not in memory.snapshot()["session_focus"]


def test_promote_candidates_returns_high_salience_items_for_session() -> None:
    memory = WorkingMemory(retention_seconds=60)
    memory.record("note", "promote me", salience=0.91, session_id="session-a")
    memory.record("note", "too low", salience=0.74, session_id="session-a")
    memory.record("note", "other session", salience=0.99, session_id="session-b")

    candidates = memory.promote_candidates(
        limit=5,
        min_salience=0.75,
        session_id="session-a",
    )

    assert [candidate["content"] for candidate in candidates] == ["promote me"]


def test_clear_session_removes_only_that_session() -> None:
    memory = WorkingMemory(retention_seconds=60)
    memory.record("note", "session A one", conversation_session_id="session-a")
    memory.record("note", "session A two", session_id="session-a")
    memory.record("note", "session B", session_id="session-b")

    removed = memory.clear_session(conversation_session_id="session-a")
    snapshot = memory.snapshot()

    assert removed == 2
    assert snapshot["item_count"] == 1
    assert snapshot["items"][0]["content"] == "session B"


def test_summary_and_snapshot_preserve_existing_shape() -> None:
    memory = WorkingMemory(retention_seconds=60)

    memory.record_turn("inspect area A", observations=["scene has a door"])
    memory.set_focus(route="A")
    snapshot = memory.snapshot()
    summary = memory.summary()

    assert snapshot["persist_enabled"] is False
    assert snapshot["item_count"] == 2
    assert snapshot["focus"]["route"] == "A"
    assert "inspect area A" in summary
    assert "focus: route=A" in summary
