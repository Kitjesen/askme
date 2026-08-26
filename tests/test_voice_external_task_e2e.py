import asyncio
from types import SimpleNamespace

import pytest

from askme.conversation import TurnStatus, VoiceTurnLedger
from askme.pipeline.channels.voice_loop import VoiceLoop, _CapturedUtterance
from askme.runtime.task.voice_lifecycle import (
    CancelRequestResult,
    TaskLifecycleEvent,
    TaskStatusSnapshot,
    VoiceTaskOperatorContext,
)


class _Router:
    def route(self, text):
        raise AssertionError(f"unexpected route: {text}")


class _Pipeline:
    last_spoken_text = ""

    def __init__(self, ledger=None):
        self._turn_ledger = ledger
        self.process_calls = []
        self.skill_calls = []
        self.cancel_calls = 0

    def start_idle_reflection(self):
        return None

    def has_pending_tool_approval(self):
        return False

    def cancel_active_turn(self, *, reason):
        self.cancel_calls += 1


class _Audio:
    full_duplex_enabled = False
    awaiting_confirmation = False
    is_muted = False

    def __init__(self):
        self.spoken = []
        self.drained = 0
        self.listen_calls = 0
        self._barge_in = None

    def listen_loop(self):
        self.listen_calls += 1
        return "hello"

    async def speak_and_wait(self, text):
        self.spoken.append(text)

    def drain_buffers(self):
        self.drained += 1

    def set_barge_in_callback(self, callback):
        self._barge_in = callback


class _Lifecycle:
    def __init__(self):
        self.calls = []
        self.events = []
        self.receipts = {}
        self.retry_calls = []
        self.ready = False
        self.snapshot = TaskStatusSnapshot(thread_id="session-a")

    async def start(self):
        self.calls.append("start")

    async def close(self):
        self.calls.append("close")

    def reserve_task(self, user_text, thread_id, turn_id):
        self.calls.append(("reserve", user_text, thread_id, turn_id))
        return SimpleNamespace(
            reservation_id="reservation-a",
            state="reserved",
            run_id="",
            turn_id=turn_id,
            task_type="status_report",
            target="",
            confirmation_prompt="",
            approval_id="",
        )

    reserve_status_report = reserve_task

    def confirm_pending(self, thread_id, turn_id):
        self.calls.append(("confirm", thread_id, turn_id))
        return SimpleNamespace(
            reservation_id="reservation-navigation",
            state="confirmed",
            run_id="run-a",
            turn_id="origin-turn",
            task_type="navigate_to",
            target="A 区",
            confirmation_prompt="",
            approval_id="approval-a",
        )

    async def commit_ack_and_submit(self, reservation_id):
        self.calls.append(("submit", reservation_id))
        return SimpleNamespace(
            run_id="run-a",
            remote_task_id="remote-a",
            correlation_id="run-a",
            state="queued",
            accepted=True,
        )

    def abandon(self, reservation_id):
        self.calls.append(("abandon", reservation_id))
        return True

    def status_snapshot(self, thread_id):
        self.calls.append(("status", thread_id))
        return self.snapshot

    async def cancel_active(self, thread_id, **_kwargs):
        self.calls.append(("cancel", thread_id))
        return CancelRequestResult(True, self.snapshot)

    async def wait_ready(self, thread_id, timeout=None):
        self.calls.append(("wait", thread_id))
        return self.ready

    def claim_next(self, thread_id):
        self.calls.append(("claim", thread_id))
        return self.events.pop(0) if self.events else None

    def settle_delivery(self, event_id, state):
        self.receipts[event_id] = state

    def retry_delivery(self, event_id, *, error_code):
        self.retry_calls.append((event_id, error_code))
        self.receipts[event_id] = "pending"
        return True


def _loop(*, pipeline=None, audio=None, lifecycle=None, operator_provider=None):
    return VoiceLoop(
        router=_Router(),
        pipeline=pipeline or _Pipeline(),
        audio=audio or _Audio(),
        voice_task_lifecycle=lifecycle,
        voice_task_operator_provider=operator_provider,
    )


@pytest.mark.asyncio
async def test_external_task_ack_is_committed_before_submit() -> None:
    lifecycle = _Lifecycle()
    audio = _Audio()
    committed = []

    class RecordingPipeline(_Pipeline):
        async def record_direct_reply(self, user_text, assistant_text, **kwargs):
            committed.append((user_text, assistant_text, kwargs["metadata"]))

    pipeline = RecordingPipeline()
    loop = _loop(pipeline=pipeline, audio=audio, lifecycle=lifecycle)
    original_submit = lifecycle.commit_ack_and_submit

    async def submit(reservation_id):
        assert committed
        assert audio.spoken == ["好的，我准备提交状态报告任务。"]
        return await original_submit(reservation_id)

    lifecycle.commit_ack_and_submit = submit
    await lifecycle.start()

    await loop._handle_local_external_task_start(
        user_text="请生成一份巡检状态报告",
        thread_id="session-a",
        turn_id="origin-turn",
        interaction_cancel=SimpleNamespace(is_set=lambda: False),
    )

    assert lifecycle.calls[-1] == ("submit", "reservation-a")
    assert audio.spoken == [
        "好的，我准备提交状态报告任务。",
        "任务已受理。当前是半双工模式，请稍后问我任务状态。",
    ]
    metadata = committed[0][2]
    assert metadata["task_reservation_id"] == "reservation-a"
    assert metadata["task_run_correlation_id"] == "reservation-a"
    assert metadata["task_turn_id"] == "origin-turn"


@pytest.mark.asyncio
async def test_cancelled_ack_abandons_reserved_task_before_submission() -> None:
    lifecycle = _Lifecycle()

    class CancellingAudio(_Audio):
        async def speak_and_wait(self, text):
            self.spoken.append(text)
            raise asyncio.CancelledError

    loop = _loop(audio=CancellingAudio(), lifecycle=lifecycle)

    with pytest.raises(asyncio.CancelledError):
        await loop._handle_local_external_task_start(
            user_text="请生成一份巡检状态报告",
            thread_id="session-a",
            turn_id="origin-turn-cancelled-ack",
            interaction_cancel=SimpleNamespace(is_set=lambda: False),
        )

    assert ("abandon", "reservation-a") in lifecycle.calls
    assert all(not (isinstance(call, tuple) and call[0] == "submit") for call in lifecycle.calls)


@pytest.mark.asyncio
async def test_acknowledged_local_submit_failure_is_reported_as_new_assistant_turn() -> None:
    lifecycle = _Lifecycle()
    audio = _Audio()
    committed = []

    class RecordingPipeline(_Pipeline):
        async def record_direct_reply(self, user_text, assistant_text, **kwargs):
            committed.append((user_text, assistant_text, kwargs["metadata"]))

    async def fail_submit(reservation_id):
        lifecycle.calls.append(("submit", reservation_id))
        raise RuntimeError("local persistence unavailable")

    lifecycle.commit_ack_and_submit = fail_submit
    loop = _loop(pipeline=RecordingPipeline(), audio=audio, lifecycle=lifecycle)

    await loop._handle_local_external_task_start(
        user_text="请生成一份巡检状态报告",
        thread_id="session-a",
        turn_id="origin-turn-fail",
        interaction_cancel=SimpleNamespace(is_set=lambda: False),
    )

    assert audio.spoken == [
        "好的，我准备提交状态报告任务。",
        "任务提交失败，没有进入外部执行器，请稍后重试。",
    ]
    assert committed[-1][0] == ""
    assert committed[-1][2]["submitted"] is False
    assert committed[-1][2]["task_state"] == "failed"


@pytest.mark.asyncio
async def test_unknown_submission_is_not_announced_as_accepted() -> None:
    lifecycle = _Lifecycle()
    audio = _Audio()
    committed = []

    class RecordingPipeline(_Pipeline):
        async def record_direct_reply(self, user_text, assistant_text, **kwargs):
            committed.append((user_text, assistant_text, kwargs["metadata"]))

    async def unknown_submit(reservation_id):
        lifecycle.calls.append(("submit", reservation_id))
        return SimpleNamespace(
            run_id="run-unknown",
            remote_task_id="",
            correlation_id="run-unknown",
            state="submission_unknown",
            accepted=False,
        )

    lifecycle.commit_ack_and_submit = unknown_submit
    loop = _loop(pipeline=RecordingPipeline(), audio=audio, lifecycle=lifecycle)

    await loop._handle_local_external_task_start(
        user_text="请生成一份巡检状态报告",
        thread_id="session-a",
        turn_id="origin-turn-unknown",
        interaction_cancel=SimpleNamespace(is_set=lambda: False),
    )

    assert audio.spoken == [
        "好的，我准备提交状态报告任务。",
        "提交结果暂时无法确认，我正在使用同一任务标识对账，请不要重复提交。",
    ]
    assert all("任务已受理" not in text for text in audio.spoken)
    assert committed[-1][2]["task_state"] == "submission_unknown"
    assert committed[-1][2]["submitted"] == "unknown"


@pytest.mark.asyncio
async def test_navigation_waits_for_explicit_confirmation_before_submission() -> None:
    lifecycle = _Lifecycle()
    audio = _Audio()

    def reserve_navigation(user_text, thread_id, turn_id):
        lifecycle.calls.append(("reserve", user_text, thread_id, turn_id))
        return SimpleNamespace(
            reservation_id="reservation-navigation",
            state="waiting_user",
            run_id="run-navigation",
            turn_id=turn_id,
            task_type="navigate_to",
            target="A 区",
            confirmation_prompt=(
                "将移动机器人前往A 区。请说确认执行或取消任务。"
            ),
            approval_id="approval-navigation",
        )

    lifecycle.reserve_task = reserve_navigation
    loop = _loop(audio=audio, lifecycle=lifecycle)
    cancel = SimpleNamespace(is_set=lambda: False)

    await loop._handle_local_external_task_start(
        user_text="请前往 A 区",
        thread_id="session-a",
        turn_id="navigation-prompt-turn",
        interaction_cancel=cancel,
    )

    assert not any(call[0] == "submit" for call in lifecycle.calls if isinstance(call, tuple))
    assert audio.spoken == ["将移动机器人前往A 区。请说确认执行或取消任务。"]

    await loop._handle_task_control(
        "task_confirm",
        user_text="确认执行",
        thread_id="session-a",
        turn_id="navigation-confirm-turn",
        interaction_cancel=cancel,
    )

    assert ("confirm", "session-a", "navigation-confirm-turn") in lifecycle.calls
    assert ("submit", "reservation-navigation") in lifecycle.calls
    assert audio.spoken[-2:] == [
        "确认收到，我准备提交导航任务。",
        "任务已受理。当前是半双工模式，请稍后问我任务状态。",
    ]


@pytest.mark.asyncio
async def test_voice_loop_revalidates_current_speaker_before_physical_confirmation() -> None:
    class IdentityLifecycle(_Lifecycle):
        owner_person_id = ""

        def reserve_task(
            self,
            user_text,
            thread_id,
            turn_id,
            *,
            operator_context,
        ):
            self.calls.append(("reserve_identity", operator_context.person_id, turn_id))
            self.owner_person_id = operator_context.person_id
            return SimpleNamespace(
                reservation_id="reservation-navigation",
                state="waiting_user",
                run_id="run-navigation",
                turn_id=turn_id,
                task_type="navigate_to",
                target="北门",
                confirmation_prompt="将移动机器人前往北门。请说确认执行或取消任务。",
                approval_id="approval-navigation",
            )

        def confirm_pending(
            self,
            thread_id,
            turn_id,
            *,
            operator_context,
        ):
            self.calls.append(("confirm_identity", operator_context.person_id, turn_id))
            if operator_context.person_id != self.owner_person_id:
                raise PermissionError("pending task belongs to a different speaker")
            return SimpleNamespace(
                reservation_id="reservation-navigation",
                state="confirmed",
                run_id="run-navigation",
                turn_id="turn-owner-prompt",
                task_type="navigate_to",
                target="北门",
                confirmation_prompt="",
                approval_id="approval-navigation",
            )

    people = {
        "turn-owner-prompt": "person-a",
        "turn-other-confirm": "person-b",
        "turn-owner-confirm": "person-a",
    }

    def provider(_session_id, turn_id):
        return VoiceTaskOperatorContext(
            operator_id="shared-robot-device",
            roles=("operator",),
            authenticated=True,
            source="speaker_verification",
            person_id=people[turn_id],
            permissions=("runtime:read", "runtime:submit", "runtime:cancel"),
        )

    lifecycle = IdentityLifecycle()
    audio = _Audio()
    loop = _loop(
        audio=audio,
        lifecycle=lifecycle,
        operator_provider=provider,
    )
    cancel = SimpleNamespace(is_set=lambda: False)

    await loop._handle_local_external_task_start(
        user_text="导航到北门",
        thread_id="shared-session",
        turn_id="turn-owner-prompt",
        interaction_cancel=cancel,
    )
    await loop._handle_task_control(
        "task_confirm",
        user_text="确认执行",
        thread_id="shared-session",
        turn_id="turn-other-confirm",
        interaction_cancel=cancel,
    )

    assert ("submit", "reservation-navigation") not in lifecycle.calls
    assert audio.spoken[-1] == "当前语音操作者未通过任务确认授权。"

    await loop._handle_task_control(
        "task_confirm",
        user_text="确认执行",
        thread_id="shared-session",
        turn_id="turn-owner-confirm",
        interaction_cancel=cancel,
    )

    assert ("submit", "reservation-navigation") in lifecycle.calls
    assert ("confirm_identity", "person-b", "turn-other-confirm") in lifecycle.calls
    assert ("confirm_identity", "person-a", "turn-owner-confirm") in lifecycle.calls


@pytest.mark.asyncio
async def test_voice_loop_without_turn_identity_fails_closed_before_reservation() -> None:
    class IdentityRequiredLifecycle(_Lifecycle):
        def reserve_task(
            self,
            _user_text,
            _session_id,
            _turn_id,
            *,
            operator_context,
        ):
            assert operator_context.authenticated is False
            raise PermissionError("physical_task_speaker_identity_required")

    lifecycle = IdentityRequiredLifecycle()
    audio = _Audio()
    loop = _loop(audio=audio, lifecycle=lifecycle)

    await loop._handle_local_external_task_start(
        user_text="导航到北门",
        thread_id="shared-session",
        turn_id="turn-unverified",
        interaction_cancel=SimpleNamespace(is_set=lambda: False),
    )

    assert audio.spoken == ["当前没有可信说话人身份，巡检和导航任务未提交。"]
    assert not any(call[0] == "submit" for call in lifecycle.calls if isinstance(call, tuple))


@pytest.mark.asyncio
async def test_unauthorized_external_task_is_not_acknowledged_or_submitted() -> None:
    lifecycle = _Lifecycle()
    audio = _Audio()

    def reject(*_args):
        raise PermissionError("operator context missing")

    lifecycle.reserve_task = reject
    loop = _loop(audio=audio, lifecycle=lifecycle)

    await loop._handle_local_external_task_start(
        user_text="请生成巡检状态报告",
        thread_id="session-a",
        turn_id="unauthorized-turn",
        interaction_cancel=SimpleNamespace(is_set=lambda: False),
    )

    assert audio.spoken == ["当前语音操作者未通过外部任务授权，任务未提交。"]
    assert not any(call[0] == "submit" for call in lifecycle.calls if isinstance(call, tuple))


@pytest.mark.asyncio
async def test_unsupported_agent_task_is_rejected_without_submission() -> None:
    lifecycle = _Lifecycle()
    audio = _Audio()

    def reject_unsupported(*_args):
        raise ValueError("unsupported_voice_task:custom")

    lifecycle.reserve_task = reject_unsupported
    loop = _loop(audio=audio, lifecycle=lifecycle)

    await loop._handle_local_external_task_start(
        user_text="帮我随便执行一个系统命令",
        thread_id="session-a",
        turn_id="unsupported-turn",
        interaction_cancel=SimpleNamespace(is_set=lambda: False),
    )

    assert not any(
        call[0] in {"reserve", "submit"} for call in lifecycle.calls if isinstance(call, tuple)
    )
    assert audio.spoken == ["当前语音任务只支持状态报告、区域巡检和导航，任务没有提交。"]


@pytest.mark.parametrize(
    "bridge_result",
    [RuntimeError("bridge timeout"), None, "invalid", {}, {"handled": True}],
)
@pytest.mark.asyncio
async def test_local_taskrun_authority_bypasses_bridge_for_runtime_task(bridge_result) -> None:
    lifecycle = _Lifecycle()
    audio = _Audio()

    class Bridge:
        calls = 0

        def handle_voice_text(self, _text):
            self.calls += 1
            if isinstance(bridge_result, Exception):
                raise bridge_result
            return bridge_result

    bridge = Bridge()
    loop = VoiceLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        audio=audio,
        voice_runtime_bridge=bridge,
        voice_task_lifecycle=lifecycle,
    )

    outcome = await loop._handle_runtime_task_turn(
        user_text="请生成巡检状态报告",
        thread_id="session-a",
        turn_id="ambiguous-turn",
        interaction_cancel=SimpleNamespace(is_set=lambda: False),
    )

    assert outcome.handled is True
    assert bridge.calls == 0
    assert ("submit", "reservation-a") in lifecycle.calls
    assert audio.spoken == [
        "好的，我准备提交状态报告任务。",
        "任务已受理。当前是半双工模式，请稍后问我任务状态。",
    ]


@pytest.mark.asyncio
async def test_bridge_is_used_only_when_local_task_lifecycle_is_unavailable() -> None:
    audio = _Audio()

    class Bridge:
        calls = 0

        def handle_voice_text(self, _text):
            self.calls += 1
            return {
                "handled": False,
                "disposition": "declined",
                "reason": "runtime_bridge_disabled",
            }

    bridge = Bridge()
    loop = VoiceLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        audio=audio,
        voice_runtime_bridge=bridge,
        voice_task_lifecycle=None,
    )

    outcome = await loop._handle_runtime_task_turn(
        user_text="请生成巡检状态报告",
        thread_id="session-a",
        turn_id="declined-turn",
        interaction_cancel=SimpleNamespace(is_set=lambda: False),
    )

    assert outcome.explicitly_declined is True
    assert bridge.calls == 1
    assert audio.spoken == ["当前没有接入可跟踪的外部任务服务。"]


@pytest.mark.asyncio
async def test_ambiguous_bridge_result_uses_the_canonical_thread_for_reply() -> None:
    audio = _Audio()

    class Bridge:
        def handle_voice_text(self, _text):
            raise TimeoutError("submission outcome unknown")

    loop = VoiceLoop(
        router=_Router(),
        pipeline=_Pipeline(),
        audio=audio,
        voice_runtime_bridge=Bridge(),
        voice_task_lifecycle=None,
    )

    outcome = await loop._handle_runtime_task_turn(
        user_text="请生成巡检状态报告",
        thread_id="session-a",
        turn_id="ambiguous-turn",
        interaction_cancel=SimpleNamespace(is_set=lambda: False),
    )

    assert outcome.ambiguous is True
    assert audio.spoken == [
        "远端执行状态暂时无法确认。为避免重复执行，我没有在本地再次提交，请稍后询问任务状态。"
    ]


@pytest.mark.asyncio
async def test_full_duplex_ack_promises_proactive_completion_delivery() -> None:
    lifecycle = _Lifecycle()
    audio = _Audio()
    audio.full_duplex_enabled = True
    loop = _loop(audio=audio, lifecycle=lifecycle)
    loop._full_duplex_active = True

    await loop._handle_local_external_task_start(
        user_text="请生成一份巡检状态报告",
        thread_id="session-a",
        turn_id="origin-turn",
        interaction_cancel=SimpleNamespace(is_set=lambda: False),
    )

    assert audio.spoken == [
        "好的，我准备提交状态报告任务。",
        "任务已受理，完成后我会播报结果。",
    ]


@pytest.mark.asyncio
async def test_task_status_and_cancel_are_truthful_lifecycle_controls() -> None:
    lifecycle = _Lifecycle()
    lifecycle.snapshot = TaskStatusSnapshot(
        thread_id="session-a",
        reservation_id="reservation-a",
        run_id="run-a",
        remote_task_id="remote-a",
        turn_id="origin-turn",
        state="executing",
        active=True,
    )
    audio = _Audio()
    loop = _loop(audio=audio, lifecycle=lifecycle)
    cancel = SimpleNamespace(is_set=lambda: False)

    await loop._handle_task_control(
        "task_status",
        user_text="任务怎么样了",
        thread_id="session-a",
        turn_id="status-turn",
        interaction_cancel=cancel,
    )
    await loop._handle_task_control(
        "task_cancel",
        user_text="取消任务",
        thread_id="session-a",
        turn_id="cancel-turn",
        interaction_cancel=cancel,
    )

    assert audio.spoken == [
        "当前任务状态是executing。",
        "已向外部任务发送取消请求，我会继续同步最终状态。",
    ]
    assert ("cancel", "session-a") in lifecycle.calls


@pytest.mark.asyncio
async def test_task_event_creates_one_new_correlated_assistant_turn(tmp_path) -> None:
    ledger = VoiceTurnLedger(tmp_path / "task-events.jsonl")
    pipeline = _Pipeline(ledger)
    lifecycle = _Lifecycle()
    lifecycle.snapshot = TaskStatusSnapshot(
        thread_id="session-a",
        reservation_id="reservation-a",
        run_id="run-a",
        remote_task_id="remote-a",
        turn_id="origin-turn",
        state="completed",
    )
    event = SimpleNamespace(
        event_id="runtime:event-a",
        reservation_id="reservation-a",
        run_id="run-a",
        correlation_id="correlation-event-a",
        remote_task_id="remote-event-a",
        thread_id="session-a",
        turn_id="origin-turn",
        kind="completed",
        state="completed",
        message="任务完成。",
        result_summary="巡检状态正常。",
    )
    lifecycle.events.append(event)
    loop = _loop(pipeline=pipeline, lifecycle=lifecycle)

    await loop._deliver_next_task_event("session-a")
    await loop._deliver_next_task_event("session-a")

    turns = ledger.list_turns(thread_id="session-a")
    assert len(turns) == 1
    assert turns[0].status is TurnStatus.COMMITTED
    assert turns[0].user_text == ""
    assert turns[0].assistant_text == "巡检状态正常。"
    assert turns[0].metadata["runtime_run_id"] == "run-a"
    assert turns[0].metadata["task_run_correlation_id"] == "correlation-event-a"
    assert turns[0].metadata["remote_task_id"] == "remote-event-a"
    assert turns[0].metadata["task_turn_id"] == "origin-turn"
    assert lifecycle.receipts[event.event_id] == "delivered"
    assert pipeline.last_spoken_text == "巡检状态正常。"


@pytest.mark.asyncio
async def test_task_delivery_receipt_failure_is_bounded_and_does_not_crash() -> None:
    lifecycle = _Lifecycle()
    event = TaskLifecycleEvent(
        event_id="runtime:event-settlement-failure",
        reservation_id="reservation-a",
        run_id="run-a",
        thread_id="session-a",
        turn_id="origin-turn",
        kind="completed",
        state="completed",
        message="任务完成。",
    )
    lifecycle.events.append(event)
    settlement_attempts = 0

    def fail_settlement(_event_id, _state):
        nonlocal settlement_attempts
        settlement_attempts += 1
        raise RuntimeError("store closing")

    lifecycle.settle_delivery = fail_settlement
    loop = _loop(lifecycle=lifecycle)

    await loop._deliver_next_task_event("session-a")

    assert settlement_attempts == 2


@pytest.mark.asyncio
async def test_transient_task_notification_failure_is_requeued_not_suppressed() -> None:
    lifecycle = _Lifecycle()
    event = TaskLifecycleEvent(
        event_id="runtime:event-transient-delivery",
        reservation_id="reservation-a",
        run_id="run-a",
        thread_id="session-a",
        turn_id="origin-turn",
        kind="completed",
        state="completed",
        message="任务完成。",
    )
    lifecycle.events.append(event)
    audio = _Audio()

    async def fail_speech(_text):
        raise RuntimeError("tts unavailable")

    audio.speak_and_wait = fail_speech
    loop = _loop(audio=audio, lifecycle=lifecycle)

    await loop._deliver_next_task_event("session-a")

    assert lifecycle.retry_calls == [(event.event_id, "voice_notification_delivery_failed")]
    assert lifecycle.receipts[event.event_id] == "pending"


@pytest.mark.asyncio
async def test_full_duplex_event_keeps_single_listener_and_user_wins_tie() -> None:
    lifecycle = _Lifecycle()
    lifecycle.ready = True
    audio = _Audio()
    audio.full_duplex_enabled = True
    loop = _loop(audio=audio, lifecycle=lifecycle)
    loop._full_duplex_active = True
    capture_gate = asyncio.Event()

    async def capture():
        audio.listen_calls += 1
        await capture_gate.wait()
        return _CapturedUtterance("user", False, "none")

    loop._capture_once = capture
    utterance, event_ready = await loop._next_voice_activity()
    listener = loop._listen_task
    assert utterance is None and event_ready is True
    assert listener is not None and not listener.done()
    assert audio.listen_calls == 1

    capture_gate.set()
    await asyncio.sleep(0)
    utterance, event_ready = await loop._next_voice_activity()
    assert utterance is not None and utterance.text == "user"
    assert event_ready is False
    assert audio.listen_calls == 2  # one consumed plus one normal prefetch
    await loop.stop()


@pytest.mark.asyncio
async def test_half_duplex_defers_task_events_and_notification_barge_in_only_interrupts_speech(
    tmp_path,
) -> None:
    lifecycle = _Lifecycle()
    lifecycle.ready = True
    audio = _Audio()
    pipeline = _Pipeline(VoiceTurnLedger(tmp_path / "barge-event.jsonl"))
    pipeline.last_spoken_text = "上一条已听到的回复。"
    loop = _loop(pipeline=pipeline, audio=audio, lifecycle=lifecycle)

    utterance, event_ready = await loop._next_voice_activity()
    assert utterance is not None and event_ready is False
    assert not any(call[0] == "wait" for call in lifecycle.calls if isinstance(call, tuple))

    lifecycle.snapshot = TaskStatusSnapshot(
        thread_id="session-a",
        reservation_id="reservation-a",
        run_id="run-a",
        remote_task_id="remote-a",
        turn_id="origin-turn",
        state="completed",
    )
    event = TaskLifecycleEvent(
        event_id="runtime:event-barge",
        reservation_id="reservation-a",
        run_id="run-a",
        thread_id="session-a",
        turn_id="origin-turn",
        kind="completed",
        state="completed",
        message="任务已经完成。",
    )
    lifecycle.events.append(event)

    async def interrupted_speak(text):
        audio.spoken.append(text)
        loop._on_confirmed_barge_in()

    audio.speak_and_wait = interrupted_speak
    await loop._deliver_next_task_event("session-a")

    assert lifecycle.receipts[event.event_id] == "interrupted"
    assert lifecycle.snapshot.state == "completed"
    assert pipeline.cancel_calls == 0
    assert pipeline.last_spoken_text == "上一条已听到的回复。"
