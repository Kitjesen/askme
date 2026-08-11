from __future__ import annotations

import asyncio

import pytest

from askme.conversation import VoiceTurnLedger
from askme.pipeline.channels.voice_loop import VoiceLoop
from askme.robot_interaction import Intent, IntentType


class _Approval:
    initial_text = "你好。"
    completed = True

    def wait(self, timeout: float | None = None) -> str:
        return "你好，很高兴见到你。"


class _PreparedApproval(_Approval):
    generation = 4


class _RealtimeAudio:
    awaiting_confirmation = False
    last_turn_wake_authorized = False
    last_turn_wake_source = "none"
    last_turn_realtime_generation = 4
    last_turn_realtime_baseline_generation = 3
    full_duplex_enabled = False

    def __init__(self) -> None:
        self._inputs = iter(["你好", "exit"])
        self.approved: list[str] = []
        self.discarded: list[tuple[str, int, int]] = []
        self.ack_count = 0
        self.stop_count = 0
        self.released_voice_turn_ids: list[str] = []
        self.finished_generations: list[int] = []
        self._muted = False

    def listen_loop(self) -> str:
        return next(self._inputs)

    def realtime_general_chat_ready(self) -> bool:
        return True

    def realtime_capture_active(self) -> bool:
        return True

    def try_realtime_general_chat(
        self,
        text: str,
        *,
        expected_generation: int = 0,
    ) -> _Approval:
        assert expected_generation == 4
        self.approved.append(text)
        return _Approval()

    def prepare_realtime_general_chat(
        self,
        text: str,
        *,
        expected_generation: int = 0,
    ) -> _PreparedApproval:
        assert expected_generation == 4
        self.approved.append(text)
        return _PreparedApproval()

    def release_realtime_general_chat(
        self,
        approval: _PreparedApproval,
        *,
        expected_generation: int = 0,
        voice_turn_id: str | None = None,
    ) -> bool:
        if voice_turn_id:
            self.released_voice_turn_ids.append(voice_turn_id)
        return expected_generation == approval.generation

    def finish_realtime_playback(self, *, expected_generation: int) -> bool:
        self.finished_generations.append(expected_generation)
        self.stop_count += 1
        return expected_generation == 4

    def discard_realtime_turn(
        self,
        reason: str,
        *,
        expected_generation: int = 0,
        after_generation: int = 0,
    ) -> None:
        self.discarded.append(
            (reason, expected_generation, after_generation)
        )

    def realtime_playback_started(self) -> bool:
        return True

    def realtime_context_snapshot(self) -> dict[str, object]:
        return {
            "session_id": "local-runtime-session",
            "provider_session_id": "provider-session-1",
            "provider": "volcengine_s2s",
            "dialog_id": "provider-dialog-1",
            "physical_played_ms": 780,
        }

    def abort_realtime_playback(self, reason: str) -> None:
        self.discard_realtime_turn(reason)

    def wait_speaking_done(self) -> bool:
        return True

    def stop_playback(self) -> None:
        self.stop_count += 1

    def acknowledge(self) -> None:
        self.ack_count += 1

    async def speak_and_wait(self, text: str) -> None:
        raise AssertionError(f"cascade TTS must not run for approved S2S: {text}")

    def mark_interaction_turn(self) -> None:
        return None

    @property
    def is_muted(self) -> bool:
        return self._muted


class _Router:
    def route(self, text: str) -> Intent:
        if text == "exit":
            return Intent(type=IntentType.COMMAND, command="exit", raw_text=text)
        return Intent(type=IntentType.GENERAL, raw_text=text)


class _Conversation:
    def __init__(self) -> None:
        self.turns: list[tuple[str, str]] = []

    def add_user_message(self, text: str) -> None:
        self.turns.append(("user", text))

    def add_assistant_message(self, text: str) -> None:
        self.turns.append(("assistant", text))


class _Pipeline:
    last_spoken_text = ""

    def __init__(self) -> None:
        self.process_calls: list[str] = []
        self._conversation = _Conversation()

    def has_pending_tool_approval(self) -> bool:
        return False

    def start_idle_reflection(self):
        return None

    def start_memory_prefetch(self, user_text: str):
        return asyncio.create_task(asyncio.sleep(0, result=""))

    async def handle_pending_tool_response(self, user_text: str):
        return None

    async def process(self, user_text: str, **kwargs):
        self.process_calls.append(user_text)
        return "cascade"


@pytest.mark.asyncio
async def test_general_chat_uses_approved_realtime_audio_and_records_history(
    tmp_path,
) -> None:
    audio = _RealtimeAudio()
    pipeline = _Pipeline()
    pipeline._turn_ledger = VoiceTurnLedger(tmp_path / "realtime-session.jsonl")
    loop = VoiceLoop(router=_Router(), pipeline=pipeline, audio=audio)

    await loop._run_session()

    assert audio.approved == ["你好"]
    assert pipeline.process_calls == []
    assert pipeline.last_spoken_text == "你好，很高兴见到你。"
    assert pipeline._conversation.turns == [
        ("user", "你好"),
        ("assistant", "你好，很高兴见到你。"),
    ]
    # The final local "exit" command keeps its legacy ACK; the S2S turn adds none.
    assert audio.ack_count == 1
    assert audio.stop_count == 1
    assert len(audio.released_voice_turn_ids) == 1
    assert audio.finished_generations == [4]
    assert any(
        reason.startswith("intent_command") and generation == 4 and baseline == 3
        for reason, generation, baseline in audio.discarded
    )


@pytest.mark.asyncio
async def test_provider_turn_is_deleted_when_realtime_approval_falls_back() -> None:
    class _FallbackAudio(_RealtimeAudio):
        def prepare_realtime_general_chat(
            self,
            text: str,
            *,
            expected_generation: int = 0,
        ) -> None:
            assert text == "你好"
            assert expected_generation == 4
            return None

    audio = _FallbackAudio()
    pipeline = _Pipeline()
    loop = VoiceLoop(router=_Router(), pipeline=pipeline, audio=audio)

    await loop._run_session()

    assert pipeline.process_calls == ["你好"]
    assert (
        "realtime_general_fallback",
        4,
        3,
    ) in audio.discarded


@pytest.mark.asyncio
async def test_shadow_provider_turn_is_deleted_before_local_cascade() -> None:
    class _ShadowAudio(_RealtimeAudio):
        def realtime_general_chat_ready(self) -> bool:
            return False

    audio = _ShadowAudio()
    pipeline = _Pipeline()
    loop = VoiceLoop(router=_Router(), pipeline=pipeline, audio=audio)

    await loop._run_session()

    assert pipeline.process_calls == ["你好"]
    assert ("local_cascade", 4, 3) in audio.discarded


@pytest.mark.parametrize(
    ("utterance", "expected_reason"),
    [
        ("小算，检查一下", "robot_or_tool_route"),
        ("小算，暂停一下", "emergency"),
        ("小算，帮我查一下最新天气", "robot_or_tool_route"),
    ],
)
async def test_wake_word_does_not_reclassify_robot_control_as_s2s_chat(
    utterance: str,
    expected_reason: str,
) -> None:
    class _ControlAudio(_RealtimeAudio):
        def __init__(self) -> None:
            super().__init__()
            self._inputs = iter([utterance, "exit"])

    audio = _ControlAudio()
    pipeline = _Pipeline()
    loop = VoiceLoop(router=_Router(), pipeline=pipeline, audio=audio)

    await loop._run_session()

    assert audio.approved == []
    assert pipeline.process_calls == [utterance]
    assert (expected_reason, 4, 3) in audio.discarded


@pytest.mark.asyncio
async def test_ledger_conflict_blocks_two_phase_pcm_release(tmp_path) -> None:
    class _TwoPhaseAudio(_RealtimeAudio):
        def __init__(self) -> None:
            super().__init__()
            self.prepared: list[str] = []
            self.release_count = 0
            self.released_pcm_bytes = 0

        def prepare_realtime_general_chat(
            self,
            text: str,
            *,
            expected_generation: int = 0,
        ) -> _PreparedApproval:
            assert expected_generation == 4
            self.prepared.append(text)
            return _PreparedApproval()

        def release_realtime_general_chat(
            self,
            approval: _PreparedApproval,
            *,
            expected_generation: int = 0,
        ) -> bool:
            self.release_count += 1
            self.released_pcm_bytes += 2
            return True

        def try_realtime_general_chat(self, text: str, **_kwargs):
            raise AssertionError("production path must use prepare/release")

    audio = _TwoPhaseAudio()
    pipeline = _Pipeline()
    ledger = VoiceTurnLedger(tmp_path / "realtime-conflict.jsonl")
    pipeline._turn_ledger = ledger
    thread = ledger.resolve_thread(
        conversation_session_id="thread-conflict",
        channel="voice",
    )
    ledger.start_turn(
        thread.thread_id,
        turn_id="already-active",
        source="voice",
        user_text="上一轮还在处理",
    )
    loop = VoiceLoop(router=_Router(), pipeline=pipeline, audio=audio)
    loop._active_realtime_generation = 4
    loop._active_realtime_baseline_generation = 3

    handled = await loop._try_handle_realtime_general_chat(
        "你好",
        expected_generation=4,
        conversation_session_id="thread-conflict",
        voice_turn_id="new-turn",
        turn_cancel_token=None,
    )

    assert handled is False
    assert audio.prepared == ["你好"]
    assert audio.release_count == 0
    assert audio.released_pcm_bytes == 0
    assert ("conversation_turn_conflict", 4, 3) in audio.discarded


@pytest.mark.asyncio
async def test_two_phase_s2s_requires_a_durable_conversation_ledger() -> None:
    class _TwoPhaseAudio(_RealtimeAudio):
        def __init__(self) -> None:
            super().__init__()
            self.release_count = 0

        def prepare_realtime_general_chat(
            self,
            text: str,
            *,
            expected_generation: int = 0,
        ) -> _PreparedApproval:
            return _PreparedApproval()

        def release_realtime_general_chat(
            self,
            approval: _PreparedApproval,
            *,
            expected_generation: int = 0,
        ) -> bool:
            self.release_count += 1
            return True

    audio = _TwoPhaseAudio()
    pipeline = _Pipeline()
    loop = VoiceLoop(router=_Router(), pipeline=pipeline, audio=audio)
    loop._active_realtime_generation = 4
    loop._active_realtime_baseline_generation = 3

    handled = await loop._try_handle_realtime_general_chat(
        "你好",
        expected_generation=4,
        conversation_session_id="thread-without-ledger",
        voice_turn_id="turn-without-ledger",
        turn_cancel_token=None,
    )

    assert handled is False
    assert audio.release_count == 0
    assert ("conversation_turn_begin_failed", 4, 3) in audio.discarded


@pytest.mark.asyncio
async def test_voice_loop_never_invokes_legacy_one_step_s2s_admission() -> None:
    audio = _RealtimeAudio()
    audio.prepare_realtime_general_chat = None  # type: ignore[method-assign]
    audio.release_realtime_general_chat = None  # type: ignore[method-assign]
    pipeline = _Pipeline()
    loop = VoiceLoop(router=_Router(), pipeline=pipeline, audio=audio)
    loop._active_realtime_generation = 4
    loop._active_realtime_baseline_generation = 3

    handled = await loop._try_handle_realtime_general_chat(
        "你好",
        expected_generation=4,
        conversation_session_id="legacy-thread",
        voice_turn_id="legacy-turn",
        turn_cancel_token=None,
    )

    assert handled is False
    assert audio.approved == []
    assert ("two_phase_admission_unavailable", 4, 3) in audio.discarded


@pytest.mark.asyncio
async def test_provider_generation_persistence_failure_blocks_pcm_release(
    tmp_path,
) -> None:
    class _GenerationFailLedger(VoiceTurnLedger):
        def start_generation(self, *args, **kwargs):
            raise OSError("generation store unavailable")

    class _TwoPhaseAudio(_RealtimeAudio):
        def __init__(self) -> None:
            super().__init__()
            self.release_count = 0

        def release_realtime_general_chat(
            self,
            approval: _PreparedApproval,
            *,
            expected_generation: int = 0,
        ) -> bool:
            self.release_count += 1
            return True

    audio = _TwoPhaseAudio()
    pipeline = _Pipeline()
    pipeline._turn_ledger = _GenerationFailLedger(
        tmp_path / "realtime-generation-failure.jsonl"
    )
    loop = VoiceLoop(router=_Router(), pipeline=pipeline, audio=audio)
    loop._active_realtime_generation = 4
    loop._active_realtime_baseline_generation = 3

    handled = await loop._try_handle_realtime_general_chat(
        "你好",
        expected_generation=4,
        conversation_session_id="generation-failure-thread",
        voice_turn_id="generation-failure-turn",
        turn_cancel_token=None,
    )

    assert handled is False
    assert audio.release_count == 0
    assert ("conversation_generation_begin_failed", 4, 3) in audio.discarded


@pytest.mark.asyncio
async def test_realtime_success_commits_provider_generation(tmp_path) -> None:
    audio = _RealtimeAudio()
    pipeline = _Pipeline()
    pipeline._turn_ledger = VoiceTurnLedger(tmp_path / "realtime-success.jsonl")
    loop = VoiceLoop(router=_Router(), pipeline=pipeline, audio=audio)

    handled = await loop._try_handle_realtime_general_chat(
        "你好",
        expected_generation=4,
        conversation_session_id="thread-runtime",
        voice_turn_id="turn-runtime",
        turn_cancel_token=None,
    )

    assert handled is True
    turn = pipeline._turn_ledger.get_turn("turn-runtime")
    assert turn is not None
    assert turn.status.value == "committed"
    generation = pipeline._turn_ledger.list_generations(turn_id="turn-runtime")[0]
    assert generation.provider == "volcengine"
    assert generation.provider_session_id == "provider-session-1"
    assert generation.status.value == "approved"


@pytest.mark.asyncio
async def test_qwen_realtime_success_uses_qwen_ledger_identity(tmp_path) -> None:
    class _QwenRealtimeAudio(_RealtimeAudio):
        def realtime_context_snapshot(self) -> dict[str, object]:
            return {
                **super().realtime_context_snapshot(),
                "provider": "qwen3_5_omni",
                "model": "qwen3.5-omni-flash-realtime",
            }

    audio = _QwenRealtimeAudio()
    pipeline = _Pipeline()
    pipeline._turn_ledger = VoiceTurnLedger(tmp_path / "qwen-realtime-success.jsonl")
    loop = VoiceLoop(router=_Router(), pipeline=pipeline, audio=audio)

    handled = await loop._try_handle_realtime_general_chat(
        "你好",
        expected_generation=4,
        conversation_session_id="thread-qwen",
        voice_turn_id="turn-qwen",
        turn_cancel_token=None,
    )

    assert handled is True
    generation = pipeline._turn_ledger.list_generations(turn_id="turn-qwen")[0]
    assert generation.provider == "qwen"
    assert generation.generation_id == "turn-qwen:qwen:4"


@pytest.mark.asyncio
async def test_realtime_never_records_local_session_as_provider_session(tmp_path) -> None:
    class _NoProviderSessionAudio(_RealtimeAudio):
        def realtime_context_snapshot(self) -> dict[str, object]:
            snapshot = super().realtime_context_snapshot()
            snapshot.pop("provider_session_id")
            return snapshot

    audio = _NoProviderSessionAudio()
    pipeline = _Pipeline()
    pipeline._turn_ledger = VoiceTurnLedger(tmp_path / "no-provider-session.jsonl")
    loop = VoiceLoop(router=_Router(), pipeline=pipeline, audio=audio)

    assert await loop._try_handle_realtime_general_chat(
        "你好",
        expected_generation=4,
        conversation_session_id="thread-no-provider-session",
        voice_turn_id="turn-no-provider-session",
        turn_cancel_token=None,
    )

    generation = pipeline._turn_ledger.list_generations(
        turn_id="turn-no-provider-session"
    )[0]
    assert generation.provider_session_id is None


@pytest.mark.asyncio
async def test_realtime_missing_provider_identity_fails_closed_before_release(
    tmp_path,
) -> None:
    class _MissingProviderAudio(_RealtimeAudio):
        def realtime_context_snapshot(self) -> dict[str, object]:
            snapshot = super().realtime_context_snapshot()
            snapshot.pop("provider")
            return snapshot

    audio = _MissingProviderAudio()
    pipeline = _Pipeline()
    pipeline._turn_ledger = VoiceTurnLedger(tmp_path / "missing-provider.jsonl")
    loop = VoiceLoop(router=_Router(), pipeline=pipeline, audio=audio)
    loop._active_realtime_generation = 4
    loop._active_realtime_baseline_generation = 3

    handled = await loop._try_handle_realtime_general_chat(
        "你好",
        expected_generation=4,
        conversation_session_id="thread-missing-provider",
        voice_turn_id="turn-missing-provider",
        turn_cancel_token=None,
    )

    assert handled is False
    assert audio.approved == []
    assert ("unsupported_realtime_provider", 4, 3) in audio.discarded
    assert pipeline._turn_ledger.get_turn("turn-missing-provider") is None


@pytest.mark.asyncio
async def test_two_phase_release_observes_durable_turn_and_generation(tmp_path) -> None:
    ledger = VoiceTurnLedger(tmp_path / "realtime-two-phase-success.jsonl")

    class _TwoPhaseAudio(_RealtimeAudio):
        def __init__(self) -> None:
            super().__init__()
            self.events: list[str] = []

        def prepare_realtime_general_chat(
            self,
            text: str,
            *,
            expected_generation: int = 0,
        ) -> _PreparedApproval:
            self.events.append("prepare")
            return _PreparedApproval()

        def release_realtime_general_chat(
            self,
            approval: _PreparedApproval,
            *,
            expected_generation: int = 0,
        ) -> bool:
            turn = ledger.get_turn("turn-two-phase")
            assert turn is not None
            assert turn.status.value == "generating"
            generations = ledger.list_generations(turn_id=turn.turn_id)
            assert len(generations) == 1
            assert generations[0].status.value == "started"
            self.events.append("release")
            return True

        def try_realtime_general_chat(self, text: str, **_kwargs):
            raise AssertionError("production path must use prepare/release")

    audio = _TwoPhaseAudio()
    pipeline = _Pipeline()
    pipeline._turn_ledger = ledger
    loop = VoiceLoop(router=_Router(), pipeline=pipeline, audio=audio)

    handled = await loop._try_handle_realtime_general_chat(
        "你好",
        expected_generation=4,
        conversation_session_id="thread-two-phase",
        voice_turn_id="turn-two-phase",
        turn_cancel_token=None,
    )

    assert handled is True
    assert audio.events == ["prepare", "release"]
    turn = ledger.get_turn("turn-two-phase")
    assert turn is not None
    assert turn.status.value == "committed"


@pytest.mark.asyncio
async def test_realtime_audio_without_text_is_settled_as_truncated(tmp_path) -> None:
    class _BlankApproval:
        generation = 4
        initial_text = ""
        completed = True

        def wait(self, timeout: float | None = None) -> str:
            return ""

    class _BlankTextAudio(_RealtimeAudio):
        def prepare_realtime_general_chat(
            self,
            text: str,
            *,
            expected_generation: int = 0,
        ) -> _BlankApproval:
            return _BlankApproval()

    audio = _BlankTextAudio()
    pipeline = _Pipeline()
    pipeline._turn_ledger = VoiceTurnLedger(tmp_path / "realtime-blank.jsonl")
    loop = VoiceLoop(router=_Router(), pipeline=pipeline, audio=audio)

    handled = await loop._try_handle_realtime_general_chat(
        "继续",
        expected_generation=4,
        conversation_session_id="thread-blank",
        voice_turn_id="turn-blank",
        turn_cancel_token=None,
    )

    assert handled is True
    turn = pipeline._turn_ledger.get_turn("turn-blank")
    assert turn is not None
    assert turn.status.value == "cancelled"
    assert turn.played_ms == 780
    generation = pipeline._turn_ledger.list_generations(turn_id="turn-blank")[0]
    assert generation.status.value == "truncated"


@pytest.mark.asyncio
async def test_realtime_cancel_uses_playhead_frozen_by_physical_stop(tmp_path) -> None:
    class _AdvancingStopAudio(_RealtimeAudio):
        def __init__(self) -> None:
            super().__init__()
            self.played_ms = 100

        def realtime_context_snapshot(self) -> dict[str, object]:
            return {
                "session_id": "local-runtime-session",
                "provider_session_id": "provider-session-stop",
                "provider": "volcengine_s2s",
                "physical_played_ms": self.played_ms,
            }

        def abort_realtime_playback(self, reason: str) -> None:
            self.played_ms = 145
            super().abort_realtime_playback(reason)

        def wait_speaking_done(self) -> bool:
            return False

    audio = _AdvancingStopAudio()
    pipeline = _Pipeline()
    pipeline._turn_ledger = VoiceTurnLedger(tmp_path / "realtime-stop.jsonl")
    loop = VoiceLoop(router=_Router(), pipeline=pipeline, audio=audio)

    handled = await loop._try_handle_realtime_general_chat(
        "打断",
        expected_generation=4,
        conversation_session_id="thread-stop",
        voice_turn_id="turn-stop",
        turn_cancel_token=None,
    )

    assert handled is True
    turn = pipeline._turn_ledger.get_turn("turn-stop")
    assert turn is not None
    assert turn.status.value == "cancelled"
    assert turn.played_ms == 145
