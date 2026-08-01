from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import replace
from typing import Any

from askme.tools.tool_registry import BaseTool, ToolRegistry

from askme.conversation import InteractionTurnContext


class _SafeTool(BaseTool):
    name = "safe_tool"
    description = "safe tool"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "normal"

    def execute(self, **kwargs: Any) -> str:
        return "safe"


class _DangerousTool(BaseTool):
    name = "dangerous_tool"
    description = "dangerous tool"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "dangerous"

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, **kwargs: Any) -> str:
        self.calls += 1
        return "dangerous"


class _CriticalTool(BaseTool):
    name = "critical_tool"
    description = "critical tool"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "critical"

    def execute(self, **kwargs: Any) -> str:
        return "critical"


class _SecondDangerousTool(BaseTool):
    name = "second_dangerous_tool"
    description = "another dangerous tool"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "dangerous"

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, **kwargs: Any) -> str:
        self.calls += 1
        return "second-dangerous"


class _EmergencyStopTool(BaseTool):
    name = "robot_emergency_stop"
    description = "emergency stop"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "critical"

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, **kwargs: Any) -> str:
        self.calls += 1
        return "estop"


class _SlowTool(BaseTool):
    name = "slow_tool"
    description = "slow tool"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "normal"

    def execute(self, **kwargs: Any) -> str:
        time.sleep(0.05)
        return "slow"


class _BrokenTool(BaseTool):
    name = "broken_tool"
    description = "broken tool"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "normal"

    def execute(self, **kwargs: Any) -> str:
        raise RuntimeError("boom")


class _BackgroundTool(BaseTool):
    name = "background_tool"
    description = "background tool"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "normal"
    backgroundable = True

    def __init__(self, result: str = "background-ok") -> None:
        self.calls = 0
        self.result = result

    def execute(self, **kwargs: Any) -> str:
        self.calls += 1
        return self.result


class _BlockingBackgroundTool(BaseTool):
    name = "blocking_background_tool"
    description = "blocking background tool"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "normal"
    backgroundable = True
    queue_priority = 0

    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self.started = started
        self.release = release

    def execute(self, **kwargs: Any) -> str:
        self.started.set()
        self.release.wait(timeout=2.0)
        return "released"


class _OrderedBackgroundTool(BaseTool):
    description = "ordered background tool"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "normal"
    backgroundable = True

    def __init__(self, name: str, priority: int, order: list[str]) -> None:
        self.name = name
        self.queue_priority = priority
        self.order = order

    def execute(self, **kwargs: Any) -> str:
        self.order.append(self.name)
        return self.name


def _make_registry(**overrides: Any) -> ToolRegistry:
    config = {
        "default_timeout": 0.2,
        "dangerous_timeout": 0.2,
        "critical_timeout": 0.2,
        "timeout_cooldown": 0.0,
        "approval_timeout_seconds": 30.0,
        "require_confirmation_levels": ["dangerous", "critical"],
        "confirmation_bypass_tools": ["robot_emergency_stop"],
        "confirmation_phrases": ["确认执行", "approve"],
        "rejection_phrases": ["取消", "cancel"],
        **overrides,
    }
    return ToolRegistry(config=config)


def _interaction_context(
    thread_id: str,
    turn_id: str,
    *,
    person_id: str = "person-1",
    operator_id: str = "operator-1",
) -> InteractionTurnContext:
    return InteractionTurnContext(
        thread_id=thread_id,
        turn_id=turn_id,
        channel="voice",
        source="voice",
        user_text="run tool",
        person_id=person_id,
        operator_id=operator_id,
    )


def _wait_job(
    registry: ToolRegistry,
    job_id: str,
    *,
    terminal: set[str] | None = None,
    timeout_s: float = 2.0,
) -> dict[str, Any]:
    terminal = terminal or {"completed", "failed", "cancelled"}
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = registry.get_job(job_id)
        if last and last.get("status") in terminal:
            return last
        time.sleep(0.01)
    return last or {}


def test_get_definitions_filters_by_max_safety_level() -> None:
    registry = _make_registry()
    registry.register(_SafeTool())
    registry.register(_DangerousTool())
    registry.register(_CriticalTool())

    definitions = registry.get_definitions(max_safety_level="normal")
    names = [definition["function"]["name"] for definition in definitions]

    assert names == ["safe_tool"]


def test_execute_blocks_tool_outside_context_policy() -> None:
    registry = _make_registry()
    registry.register(_DangerousTool())

    result = registry.execute("dangerous_tool", "{}", max_safety_level="normal")

    assert result == (
        "[Error] Tool 'dangerous_tool' requires safety level 'dangerous', "
        "but this request only allows 'normal'."
    )


def test_execute_respects_explicit_allowlist() -> None:
    registry = _make_registry()
    registry.register(_SafeTool())
    registry.register(_DangerousTool())

    result = registry.execute(
        "dangerous_tool",
        "{}",
        allowed_names={"dangerous_tool"},
        max_safety_level="dangerous",
    )

    assert result.startswith("[Approval Required]")


def test_dangerous_tool_requires_operator_approval() -> None:
    registry = _make_registry()
    tool = _DangerousTool()
    registry.register(tool)

    result = registry.execute(
        "dangerous_tool",
        '{"target": "bin-a"}',
        max_safety_level="dangerous",
    )

    assert result.startswith("[Approval Required]")
    assert registry.has_pending_approval() is True
    assert tool.calls == 0


def test_scoped_approvals_can_be_pending_for_multiple_threads() -> None:
    registry = _make_registry()
    first = _DangerousTool()
    second = _SecondDangerousTool()
    registry.register(first)
    registry.register(second)
    thread_a = _interaction_context("thread-a", "turn-a")
    thread_b = _interaction_context("thread-b", "turn-b")

    first_result = registry.execute(
        "dangerous_tool",
        '{"z": 1, "target": "bin-a", "a": 2}',
        max_safety_level="dangerous",
        interaction_context=thread_a,
    )
    second_result = registry.execute(
        "second_dangerous_tool",
        '{"target": "bin-b"}',
        max_safety_level="dangerous",
        interaction_context=thread_b,
    )

    scope_a = registry.pending_approval_scope(thread_a)
    scope_b = registry.pending_approval_scope(thread_b)
    assert first_result.startswith("[Approval Required]")
    assert second_result.startswith("[Approval Required]")
    assert scope_a is not None
    assert scope_b is not None
    assert scope_a.approval_id != scope_b.approval_id
    assert scope_a.subject == "dangerous_tool"
    assert scope_b.subject == "second_dangerous_tool"
    expected_payload = json.dumps(
        {"a": 2, "target": "bin-a", "z": 1},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert scope_a.payload_digest == hashlib.sha256(expected_payload.encode()).hexdigest()
    assert first.calls == 0
    assert second.calls == 0

    approved = registry.approve_pending(
        replace(thread_a, turn_id="turn-a-confirm"),
        approval_id=scope_a.approval_id,
    )
    assert approved == "dangerous"
    assert first.calls == 1
    assert second.calls == 0
    assert registry.pending_approval_scope(thread_b) == scope_b


def test_scoped_approval_accepts_exact_id_from_later_turn() -> None:
    registry = _make_registry()
    tool = _DangerousTool()
    registry.register(tool)
    prompt = _interaction_context("thread-a", "turn-prompt")
    registry.execute(
        "dangerous_tool",
        '{"target": "bin-a"}',
        max_safety_level="dangerous",
        interaction_context=prompt,
    )
    scope = registry.pending_approval_scope(prompt)
    assert scope is not None
    same_turn = registry.approve_pending(
        prompt,
        approval_id=scope.approval_id,
    )
    assert same_turn.startswith("[Approval]")
    assert tool.calls == 0

    response = replace(prompt, turn_id="turn-response", user_text="approve")

    assert registry.matches_confirmation(
        "approve",
        interaction_context=response,
        approval_id=scope.approval_id,
    )
    result = registry.handle_pending_input(
        "approve",
        interaction_context=response,
        approval_id=scope.approval_id,
    )

    assert result == "dangerous"
    assert tool.calls == 1
    assert registry.pending_approval_scope(response) is None


def test_scoped_approval_rejects_wrong_challenge_or_identity() -> None:
    registry = _make_registry()
    tool = _DangerousTool()
    registry.register(tool)
    prompt = _interaction_context("thread-a", "turn-prompt")
    registry.execute(
        "dangerous_tool",
        "{}",
        max_safety_level="dangerous",
        interaction_context=prompt,
    )
    scope = registry.pending_approval_scope(prompt)
    assert scope is not None
    later_turn = replace(prompt, turn_id="turn-response")

    assert registry.pending_approval_scope(replace(later_turn, thread_id="thread-b")) is None
    assert registry.pending_approval_scope(replace(later_turn, person_id="person-2")) is None
    assert registry.pending_approval_scope(replace(later_turn, operator_id="operator-2")) is None
    assert registry.approve_pending(
        interaction_context=replace(later_turn, thread_id="thread-b"),
        approval_id=scope.approval_id,
    ).startswith("[Approval]")
    assert registry.approve_pending(
        interaction_context=replace(later_turn, person_id="person-2"),
        approval_id=scope.approval_id,
    ).startswith("[Approval]")
    assert registry.reject_pending(
        interaction_context=replace(later_turn, operator_id="operator-2"),
        approval_id=scope.approval_id,
    ).startswith("[Approval]")
    assert registry.reject_pending(
        interaction_context=later_turn,
        approval_id="wrong-id",
    ).startswith("[Approval]")
    assert tool.calls == 0
    assert registry.pending_approval_scope(later_turn) == scope

    cancelled = registry.reject_pending(
        interaction_context=later_turn,
        approval_id=scope.approval_id,
    )
    assert cancelled.startswith("[Approval Cancelled]")
    assert registry.pending_approval_scope(later_turn) is None


def test_legacy_no_context_cannot_resolve_scoped_approval() -> None:
    registry = _make_registry()
    tool = _DangerousTool()
    registry.register(tool)
    context = _interaction_context("thread-a", "turn-prompt")
    registry.execute(
        "dangerous_tool",
        "{}",
        max_safety_level="dangerous",
        interaction_context=context,
    )
    scope = registry.pending_approval_scope(context)
    assert scope is not None

    assert registry.has_pending_approval() is False
    assert registry.matches_confirmation("approve") is False
    assert registry.matches_rejection("cancel") is False
    assert registry.handle_pending_input("approve") is None
    assert registry.approve_pending().startswith("[Approval]")
    assert registry.reject_pending().startswith("[Approval]")
    assert registry.pending_approval_scope(context) == scope
    assert tool.calls == 0


def test_anonymous_voice_cannot_queue_scoped_approval() -> None:
    registry = _make_registry()
    tool = _DangerousTool()
    registry.register(tool)
    prompt = replace(
        _interaction_context("public-voice", "turn-prompt"),
        person_id=None,
        operator_id=None,
    )

    result = registry.execute(
        "dangerous_tool",
        "{}",
        max_safety_level="dangerous",
        interaction_context=prompt,
    )

    assert "已认证操作员" in result


def test_scoped_approval_expires_before_exact_response() -> None:
    registry = _make_registry(approval_timeout_seconds=0.01)
    tool = _DangerousTool()
    registry.register(tool)
    prompt = _interaction_context("thread-a", "turn-prompt")
    registry.execute(
        "dangerous_tool",
        "{}",
        max_safety_level="dangerous",
        interaction_context=prompt,
    )
    scope = registry.pending_approval_scope(prompt)
    assert scope is not None
    response = replace(prompt, turn_id="turn-response")

    time.sleep(0.02)
    result = registry.approve_pending(
        interaction_context=response,
        approval_id=scope.approval_id,
    )

    assert result.startswith("[Approval Expired]")
    assert registry.pending_approval_scope(response) is None
    assert tool.calls == 0


def test_submit_background_propagates_scoped_approval_context() -> None:
    registry = _make_registry()
    tool = _DangerousTool()
    tool.backgroundable = True
    registry.register(tool)
    prompt = _interaction_context("thread-a", "turn-prompt")

    job = registry.submit_background(
        "dangerous_tool",
        '{"target": "bin-a"}',
        max_safety_level="dangerous",
        interaction_context=prompt,
    )
    scope = registry.pending_approval_scope(prompt)

    assert job["status"] == "pending_approval"
    assert scope is not None
    assert scope.subject == "dangerous_tool"
    assert scope.approval_id in job["result"]


def test_scoped_approval_executes_at_most_once_under_race() -> None:
    registry = _make_registry()
    tool = _DangerousTool()
    registry.register(tool)
    prompt = _interaction_context("thread-a", "turn-prompt")
    registry.execute(
        "dangerous_tool",
        "{}",
        max_safety_level="dangerous",
        interaction_context=prompt,
    )
    scope = registry.pending_approval_scope(prompt)
    assert scope is not None
    response = replace(prompt, turn_id="turn-response")
    barrier = threading.Barrier(3)
    results: list[str] = []

    def approve() -> None:
        barrier.wait()
        results.append(
            registry.approve_pending(
                interaction_context=response,
                approval_id=scope.approval_id,
            )
        )

    workers = [threading.Thread(target=approve) for _ in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait()
    for worker in workers:
        worker.join(timeout=1.0)

    assert tool.calls == 1
    assert sorted(result == "dangerous" for result in results) == [False, True]


def test_approve_pending_executes_dangerous_tool() -> None:
    registry = _make_registry()
    tool = _DangerousTool()
    registry.register(tool)
    registry.execute("dangerous_tool", "{}", max_safety_level="dangerous")

    result = registry.approve_pending()

    assert result == "dangerous"
    assert registry.has_pending_approval() is False
    assert tool.calls == 1


def test_reject_pending_cancels_dangerous_tool() -> None:
    registry = _make_registry()
    tool = _DangerousTool()
    registry.register(tool)
    registry.execute("dangerous_tool", "{}", max_safety_level="dangerous")

    result = registry.reject_pending()

    assert result.startswith("[Approval Cancelled]")
    assert registry.has_pending_approval() is False
    assert tool.calls == 0


def test_pending_approval_expires() -> None:
    registry = _make_registry(approval_timeout_seconds=0.01)
    registry.register(_DangerousTool())
    registry.execute("dangerous_tool", "{}", max_safety_level="dangerous")

    time.sleep(0.02)
    result = registry.approve_pending()

    assert result.startswith("[Approval Expired]")
    assert registry.has_pending_approval() is False


def test_pending_approval_blocks_new_high_risk_request() -> None:
    registry = _make_registry()
    first = _DangerousTool()
    second = _SecondDangerousTool()
    registry.register(first)
    registry.register(second)

    registry.execute("dangerous_tool", '{"target": "bin-a"}', max_safety_level="dangerous")
    result = registry.execute(
        "second_dangerous_tool",
        '{"target": "bin-b"}',
        max_safety_level="dangerous",
    )

    assert result.startswith("[Approval Pending]")
    assert "dangerous_tool" in result

    approved = registry.approve_pending()
    assert approved == "dangerous"
    assert first.calls == 1
    assert second.calls == 0


def test_handle_pending_input_requires_explicit_resolution() -> None:
    registry = _make_registry()
    registry.register(_DangerousTool())
    registry.execute("dangerous_tool", '{"target": "bin-a"}', max_safety_level="dangerous")

    result = registry.handle_pending_input("status update")

    assert result is not None
    assert result.startswith("[Approval Pending]")
    assert "dangerous_tool" in result
    assert registry.has_pending_approval() is True


def test_handle_pending_input_reports_expired_request() -> None:
    registry = _make_registry(approval_timeout_seconds=0.01)
    registry.register(_DangerousTool())
    registry.execute("dangerous_tool", "{}", max_safety_level="dangerous")

    time.sleep(0.02)
    result = registry.handle_pending_input("approve")

    assert result is not None
    assert result.startswith("[Approval Expired]")
    assert registry.has_pending_approval() is False


def test_confirmation_phrase_matching_requires_pending_request() -> None:
    registry = _make_registry()
    registry.register(_DangerousTool())

    assert registry.matches_confirmation("确认执行") is False

    registry.execute("dangerous_tool", "{}", max_safety_level="dangerous")
    assert registry.matches_confirmation("确认执行。") is True
    assert registry.matches_rejection("取消") is True


def test_critical_bypass_tool_executes_immediately() -> None:
    registry = _make_registry()
    tool = _EmergencyStopTool()
    registry.register(tool)

    result = registry.execute("robot_emergency_stop", "{}", max_safety_level="critical")

    assert result == "estop"
    assert registry.has_pending_approval() is False
    assert tool.calls == 1


def test_tool_executor_is_reused_and_shutdown_is_callable() -> None:
    registry = _make_registry()
    registry.register(_SafeTool())

    try:
        assert registry.execute("safe_tool", "{}") == "safe"
        first_executor = registry._executor
        assert first_executor is not None

        assert registry.execute("safe_tool", "{}") == "safe"
        assert registry._executor is first_executor

        registry.shutdown()
        assert registry._executor is None

        assert registry.execute("safe_tool", "{}") == "safe"
        assert registry._executor is not None
        assert registry._executor is not first_executor
    finally:
        registry.shutdown()


def test_diagnostics_report_executor_and_cooldown_state() -> None:
    registry = _make_registry(default_timeout=0.01, timeout_cooldown=1.0)
    registry.register(_SafeTool())
    registry.register(_SlowTool())

    try:
        before = registry.diagnostics()
        assert before["tool_count"] == 2
        assert before["executor"]["active"] is False
        assert before["executor"]["max_workers"] == 4
        assert before["executor"]["queue_max_size"] == 256
        assert before["cooldown_count"] == 0
        assert before["pending_approval"] is False

        assert registry.execute("safe_tool", "{}") == "safe"
        after_execute = registry.diagnostics()
        assert after_execute["executor"]["active"] is True

        timeout_result = registry.execute("slow_tool", "{}")
        assert timeout_result.startswith("[Timeout]")
        assert registry.diagnostics()["cooldown_count"] == 1
    finally:
        registry.shutdown()


def test_timeout_places_tool_on_cooldown() -> None:
    registry = _make_registry(default_timeout=0.01, timeout_cooldown=1.0)
    registry.register(_SlowTool())

    first = registry.execute("slow_tool", "{}")
    second = registry.execute("slow_tool", "{}")

    assert first.startswith("[Timeout]")
    assert second.startswith("[Error] Tool 'slow_tool' is temporarily unavailable")


def test_rate_limit_blocks_excess_tool_calls() -> None:
    registry = _make_registry(rate_limit_per_minute=1)
    registry.register(_SafeTool())

    first = registry.execute("safe_tool", "{}")
    second = registry.execute("safe_tool", "{}")

    assert first == "safe"
    assert second.startswith("[Rate Limited]")


def test_circuit_breaker_opens_after_repeated_failures() -> None:
    registry = _make_registry(
        circuit_failure_threshold=2,
        circuit_cooldown_seconds=1.0,
    )
    registry.register(_BrokenTool())

    first = registry.execute("broken_tool", "{}")
    second = registry.execute("broken_tool", "{}")
    third = registry.execute("broken_tool", "{}")

    assert first.startswith("[Error]")
    assert second.startswith("[Error]")
    assert third.startswith("[Circuit Open]")


def test_submit_background_tracks_job_to_completion() -> None:
    registry = _make_registry(executor_max_workers=1)
    tool = _BackgroundTool()
    registry.register(tool)

    job = registry.submit_background("background_tool", "{}")
    completed = _wait_job(registry, job["job_id"])

    assert job["status"] in {"queued", "running", "completed"}
    assert completed["status"] == "completed"
    assert completed["result"] == "background-ok"
    assert tool.calls == 1


def test_cancel_background_job_before_start() -> None:
    started = threading.Event()
    release = threading.Event()
    registry = _make_registry(executor_max_workers=1)
    queued_tool = _BackgroundTool()
    registry.register(_BlockingBackgroundTool(started, release))
    registry.register(queued_tool)

    running = registry.submit_background("blocking_background_tool", "{}")
    assert started.wait(timeout=1.0)
    queued = registry.submit_background("background_tool", "{}")
    cancelled = registry.cancel_job(queued["job_id"])
    release.set()
    _wait_job(registry, running["job_id"])

    assert cancelled["cancelled"] is True
    assert cancelled["status"] == "cancelled"
    assert queued_tool.calls == 0


def test_bounded_queue_rejects_excess_background_jobs() -> None:
    started = threading.Event()
    release = threading.Event()
    registry = _make_registry(executor_max_workers=1, queue_max_size=1)
    registry.register(_BlockingBackgroundTool(started, release))
    registry.register(_BackgroundTool())

    running = registry.submit_background("blocking_background_tool", "{}")
    assert started.wait(timeout=1.0)
    accepted = registry.submit_background("background_tool", "{}")
    rejected = registry.submit_background("background_tool", "{}")
    release.set()
    _wait_job(registry, running["job_id"])
    _wait_job(registry, accepted["job_id"])

    assert accepted["status"] == "queued"
    assert rejected["status"] == "failed"
    assert "queue is full" in rejected["error"]


def test_priority_queue_runs_higher_priority_queued_job_first() -> None:
    started = threading.Event()
    release = threading.Event()
    order: list[str] = []
    registry = _make_registry(executor_max_workers=1)
    registry.register(_BlockingBackgroundTool(started, release))
    registry.register(_OrderedBackgroundTool("low_priority_tool", 100, order))
    registry.register(_OrderedBackgroundTool("high_priority_tool", 10, order))

    running = registry.submit_background("blocking_background_tool", "{}")
    assert started.wait(timeout=1.0)
    low = registry.submit_background("low_priority_tool", "{}")
    high = registry.submit_background("high_priority_tool", "{}")
    release.set()

    _wait_job(registry, running["job_id"])
    _wait_job(registry, low["job_id"])
    _wait_job(registry, high["job_id"])

    assert order == ["high_priority_tool", "low_priority_tool"]
