"""Safety gate between a speculative S2S session and the local audio player."""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from typing import Any
from uuid import uuid4

from askme.voice.core.media_contracts import VoiceMediaFrame
from askme.voice.core.realtime_contracts import (
    RealtimeDialogueSession,
    RealtimeVoiceEvent,
    RealtimeVoiceEventType,
    RealtimeVoiceSessionContext,
)

AudioSink = Callable[[VoiceMediaFrame, bool], None]


@dataclass
class _TurnBuffer:
    generation: int
    transcript: str = ""
    transcript_final: bool = False
    text: str = ""
    pending_audio: deque[VoiceMediaFrame] = field(default_factory=deque)
    pending_audio_ms: float = 0.0
    response_started: bool = False
    approved: bool = False
    discarded: bool = False
    overflowed: bool = False
    question_id: str = ""
    reply_id: str = ""
    committed_audio_ms: int = 0
    truncated: bool = False
    completed: threading.Event = field(default_factory=threading.Event)


@dataclass(frozen=True)
class _RollbackRequest:
    generation: int
    reason: str
    turn: _TurnBuffer | None = field(default=None, repr=False)
    allow_missing: bool = False


@dataclass(frozen=True)
class RealtimeApproval:
    """Handle for a locally validated provider response.

    ``prepare_general_chat`` creates this handle while provider PCM is still
    buffered.  Only ``release_general_chat`` may mark the bound turn approved
    and move that PCM to the physical playback sink.
    """

    generation: int
    initial_text: str
    _completed: threading.Event = field(repr=False)
    _text_snapshot: Callable[[], str] = field(repr=False)
    _turn: _TurnBuffer = field(repr=False)

    def wait(self, timeout: float | None = None) -> str:
        self._completed.wait(timeout=timeout)
        return self._text_snapshot()

    @property
    def completed(self) -> bool:
        return self._completed.is_set()


class RealtimeDialogueCoordinator:
    """Streams input immediately but holds output until AskMe admits the turn.

    The cloud model may run speculatively to save latency.  Its audio cannot
    reach the speaker until the interaction gate, intent router, runtime
    bridge, and robot safety rules have classified the utterance as ordinary
    general conversation.
    """

    def __init__(
        self,
        session: RealtimeDialogueSession,
        context: RealtimeVoiceSessionContext,
        *,
        mode: str,
        audio_sink: AudioSink | None = None,
        pending_output_ms: int = 2_000,
        transcript_match_threshold: float = 0.66,
    ) -> None:
        self._session = session
        self._context = context
        self._mode = str(mode or "split").strip().lower()
        self._audio_sink = audio_sink
        self._pending_output_ms = max(20.0, float(pending_output_ms))
        self._transcript_match_threshold = min(
            1.0, max(0.0, float(transcript_match_threshold))
        )
        self._condition = threading.Condition()
        self._audio_emit_lock = threading.RLock()
        self._recovery_lock = threading.RLock()
        self._lifecycle_epoch = 0
        self._stop = threading.Event()
        self._consumer: threading.Thread | None = None
        self._active = False
        self._recovering = False
        self._turn: _TurnBuffer | None = None
        self._max_generation_seen = 0
        self._generation_epoch = 0
        self._overflow_count = 0
        self._dropped_late_audio = 0
        self._shadow_audio_frames = 0
        self._quarantined = False
        self._rollback_thread: threading.Thread | None = None
        self._rollback_queue: deque[_RollbackRequest] = deque()
        self._rollback_pending_generations: set[int] = set()
        self._rolled_back_generations: set[int] = set()
        self._rollback_generation = 0
        self._rollback_count = 0
        self._rollback_failures = 0
        self._truncation_count = 0
        self._recovery_count = 0
        self._recovery_failures = 0
        self._last_error = ""

    def start(self) -> bool:
        with self._condition:
            if self._active:
                return True
        start_error = ""
        try:
            started = bool(self._session.start(self._context))
        except Exception as exc:
            started = False
            start_error = type(exc).__name__
        if not started:
            try:
                self._session.close("session_start_failed")
            except Exception as exc:
                start_error = start_error or type(exc).__name__
            with self._condition:
                self._active = False
                self._recovering = False
                self._quarantined = True
                self._stop.set()
                self._last_error = start_error or "session_start_failed"
                self._condition.notify_all()
            return False
        with self._condition:
            self._stop.clear()
            self._active = True
            self._quarantined = False
            self._consumer = threading.Thread(
                target=self._consume_events,
                name="realtime-dialogue-events",
                daemon=True,
            )
            self._consumer.start()
            return True

    def offer_audio(self, frame: VoiceMediaFrame) -> bool:
        with self._condition:
            active = (
                self._active
                and not self._stop.is_set()
                and not self._quarantined
            )
        if not active:
            return False
        try:
            return bool(self._session.offer_audio(frame))
        except Exception as exc:
            with self._condition:
                self._last_error = type(exc).__name__
            return False

    def finish_input(self) -> bool:
        """Flush the provider packetizer after local endpoint detection."""

        try:
            result = self._session.finish_input()
        except Exception as exc:
            with self._condition:
                self._last_error = type(exc).__name__
            return False
        # Older provider adapters returned ``None`` for success.  Preserve that
        # compatibility while allowing the production adapter to propagate an
        # explicit queue/send failure.
        return result is not False

    def prepare_general_chat(
        self,
        local_transcript: str,
        *,
        expected_generation: int | None = None,
        wait_timeout: float = 0.65,
    ) -> RealtimeApproval | None:
        """Validate the latest response while keeping every PCM frame buffered."""

        if self._mode != "general_chat":
            return None
        deadline = time.monotonic() + max(0.0, wait_timeout)
        interrupt_reason = ""
        generation = 0
        with self._condition:
            while True:
                turn = self._turn
                ready = bool(
                    turn
                    and turn.transcript
                    and turn.transcript_final
                    and (
                        turn.response_started
                        or turn.text
                        or turn.pending_audio
                        or turn.discarded
                    )
                )
                if ready:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    generation = (
                        turn.generation
                        if turn is not None
                        else int(expected_generation or 0)
                    )
                    interrupt_reason = "approval_timeout"
                    break
                if not self._active or self._quarantined:
                    return None
                self._condition.wait(timeout=min(0.05, remaining))

            if interrupt_reason:
                approval = None
            elif turn is None:
                return None
            elif turn.discarded or turn.overflowed:
                generation = turn.generation
                interrupt_reason = "provider_turn_rejected"
                approval = None
            elif (
                expected_generation is not None
                and expected_generation > 0
                and turn.generation != expected_generation
            ):
                # The full-duplex listener may already have captured the next
                # person turn.  Never approve "latest" when the local capture
                # was bound to a different provider generation.
                generation = int(expected_generation)
                interrupt_reason = "generation_mismatch"
                approval = None
            elif not self._transcripts_match(local_transcript, turn.transcript):
                generation = turn.generation
                interrupt_reason = "transcript_mismatch"
                approval = None
            else:
                generation = turn.generation

                def _snapshot() -> str:
                    with self._condition:
                        current = self._turn
                        if current is None or current.generation != generation:
                            return ""
                        return current.text

                approval = RealtimeApproval(
                    generation=generation,
                    initial_text=turn.text,
                    _completed=turn.completed,
                    _text_snapshot=_snapshot,
                    _turn=turn,
                )
            self._condition.notify_all()

        if interrupt_reason:
            self.discard_current(
                interrupt_reason,
                expected_generation=generation,
            )
            return None
        return approval

    def release_general_chat(self, approval: RealtimeApproval) -> bool:
        """Atomically release one prepared response to the playback sink.

        The generation and private turn identity form a one-turn fence.  If a
        newer provider turn arrived, the prepared turn overflowed, or the lane
        was quarantined while the ledger was being written, no audio is emitted.
        """

        if self._mode != "general_chat" or not isinstance(approval, RealtimeApproval):
            return False
        frames: list[VoiceMediaFrame] = []
        final_already = False
        # Serialize the buffered prefix with live frames that can arrive as soon
        # as ``approved`` becomes visible.  _handle_event never holds the
        # condition while waiting for this emit lock, so this lock order is safe.
        with self._audio_emit_lock:
            with self._condition:
                turn = self._turn
                if (
                    not self._active
                    or self._stop.is_set()
                    or self._quarantined
                    or turn is None
                    or turn is not approval._turn
                    or turn.generation != approval.generation
                    or turn.discarded
                    or turn.overflowed
                ):
                    return False
                if turn.approved:
                    return True
                turn.approved = True
                frames = list(turn.pending_audio)
                turn.pending_audio.clear()
                turn.pending_audio_ms = 0.0
                final_already = turn.completed.is_set()
                self._condition.notify_all()

            for frame in frames:
                self._emit_audio(frame, final=False)
            if final_already:
                self._emit_final_marker()
        return True

    def approve_general_chat(
        self,
        local_transcript: str,
        *,
        expected_generation: int | None = None,
        wait_timeout: float = 0.65,
    ) -> RealtimeApproval | None:
        """Compatibility wrapper for legacy one-step callers.

        Production admission uses ``prepare_general_chat`` followed by a
        durable Conversation Ledger begin and then ``release_general_chat``.
        """

        approval = self.prepare_general_chat(
            local_transcript,
            expected_generation=expected_generation,
            wait_timeout=wait_timeout,
        )
        if approval is None or not self.release_general_chat(approval):
            return None
        return approval

    def recover_at_turn_boundary(self, reason: str = "turn_boundary_recovery") -> bool:
        """Restart the optional realtime lane with fresh cloud context.

        Recovery is intentionally narrow: it is only allowed after the current
        session has already failed/closed and while no history rollback is
        pending.  The new cloud dialog starts empty, and provider generation
        numbers are mapped above the maximum generation observed before the
        restart so old events cannot be mistaken for current output/history.
        """

        clean_reason = str(reason or "turn_boundary_recovery")
        new_context: RealtimeVoiceSessionContext
        with self._recovery_lock:
            with self._condition:
                rollback_active = bool(
                    self._rollback_pending_generations
                    or self._rollback_queue
                    or (
                        self._rollback_thread is not None
                        and self._rollback_thread.is_alive()
                    )
                )
                if self._active or not self._stop.is_set() or rollback_active:
                    return False
                if self._recovering:
                    return False
                recovery_epoch = self._lifecycle_epoch
                previous_consumer = self._consumer
                self._recovering = True
                self._quarantined = True
                self._condition.notify_all()

            # ``close()`` may have been called by the old consumer itself, in
            # which case it could not join that thread.  Never clear ``_stop``
            # until the old consumer has exited or it could consume events from
            # the newly started session alongside the replacement consumer.
            if (
                previous_consumer is not None
                and previous_consumer is not threading.current_thread()
            ):
                previous_consumer.join(timeout=1.0)
            if previous_consumer is not None and previous_consumer.is_alive():
                with self._condition:
                    self._recovering = False
                    self._recovery_failures += 1
                    self._last_error = "previous_event_consumer_still_running"
                    self._stop.set()
                    self._condition.notify_all()
                return False

            with self._condition:
                if (
                    self._lifecycle_epoch != recovery_epoch
                    or not self._recovering
                ):
                    self._recovering = False
                    self._stop.set()
                    self._quarantined = True
                    self._condition.notify_all()
                    return False
                prior_max = max(
                    self._max_generation_seen,
                    self._turn.generation if self._turn is not None else 0,
                )
                self._consumer = None
                self._stop.clear()
                self._turn = None
                self._rollback_queue.clear()
                self._rollback_pending_generations.clear()
                self._rolled_back_generations.clear()
                self._rollback_generation = 0
                self._generation_epoch = prior_max
                self._max_generation_seen = prior_max
                new_context = replace(
                    self._context,
                    session_id=self._new_recovery_session_id(prior_max),
                    dialog_id="",
                )
                self._condition.notify_all()

            started = False
            try:
                started = bool(self._session.start(new_context))
            except Exception as exc:
                with self._condition:
                    self._last_error = type(exc).__name__

            should_close_started_session = False
            with self._condition:
                cancelled = bool(
                    self._stop.is_set()
                    or self._lifecycle_epoch != recovery_epoch
                    or not self._recovering
                )
                if not started or cancelled:
                    self._active = False
                    self._stop.set()
                    self._quarantined = True
                    self._recovering = False
                    if not cancelled:
                        self._recovery_failures += 1
                        if not self._last_error:
                            self._last_error = clean_reason
                    should_close_started_session = bool(started)
                    self._condition.notify_all()
                else:
                    self._context = new_context
                    self._active = True
                    self._quarantined = False
                    self._recovering = False
                    self._last_error = ""
                    self._recovery_count += 1
                    consumer = threading.Thread(
                        target=self._consume_events,
                        name="realtime-dialogue-events",
                        daemon=True,
                    )
                    self._consumer = consumer
                    consumer.start()
                    self._condition.notify_all()

            if should_close_started_session:
                try:
                    self._session.close(clean_reason)
                except Exception:
                    pass
                return False
            return bool(started and not cancelled)

    def discard_current(
        self,
        reason: str,
        *,
        expected_generation: int = 0,
        after_generation: int = 0,
    ) -> None:
        """Fence output and roll the rejected QA pair out of cloud history."""

        clean_reason = str(reason or "discarded")
        target_generation = 0
        target_turn: _TurnBuffer | None = None
        allow_missing = False
        generation_mismatch = False
        rollback_known = False
        with self._condition:
            turn = self._turn
            if expected_generation > 0:
                if turn is None or turn.generation != expected_generation:
                    # Never delete the current generation on behalf of an
                    # older local capture.  Ambiguous context is safer to
                    # abandon than to corrupt by cross-generation deletion.
                    generation_mismatch = bool(
                        turn is not None and turn.generation > expected_generation
                    )
                    target_generation = int(expected_generation)
                else:
                    target_generation = turn.generation
                    target_turn = turn
                    self._discard_turn_locked(turn)
            elif turn is not None and turn.generation > after_generation:
                target_generation = turn.generation
                target_turn = turn
                self._discard_turn_locked(turn)
            else:
                # Provider ASR may trail the local final transcript.  Stop
                # admitting new cloud audio until that generation appears and
                # can be deleted by its official question/reply item id.
                target_generation = max(1, int(after_generation) + 1)
                allow_missing = True
            rollback_known = bool(
                target_generation in self._rollback_pending_generations
                or target_generation in self._rolled_back_generations
            )
            self._quarantined = True
            self._condition.notify_all()

        if generation_mismatch:
            with self._condition:
                self._last_error = "rollback_generation_mismatch"
            self.close("rollback_generation_mismatch")
            return
        if target_turn is not None and not rollback_known:
            self._safe_interrupt(clean_reason)
        self._schedule_history_rollback(
            target_generation=target_generation,
            reason=clean_reason,
            turn=target_turn,
            allow_missing=allow_missing,
        )

    def truncate_current(
        self,
        reason: str,
        *,
        audio_end_ms: int,
        expected_generation: int = 0,
    ) -> bool:
        """Fence an approved reply at the amount of audio actually played.

        A successful provider truncate keeps the heard prefix in conversation
        history and deliberately avoids ConversationDelete.  If truncation is
        unavailable or fails, the whole QA pair is deleted instead; deletion
        failure closes this optional lane.
        """

        try:
            committed_ms = int(audio_end_ms)
        except (TypeError, ValueError):
            committed_ms = 0
        clean_reason = str(reason or "playback_interrupted")
        target_generation = int(expected_generation or 0)
        reply_id = ""
        eligible = False
        with self._condition:
            turn = self._turn
            if target_generation <= 0 and turn is not None:
                target_generation = turn.generation
            if turn is not None and turn.generation == target_generation:
                eligible = bool(
                    turn.approved
                    and turn.reply_id
                    and committed_ms > 0
                )
                reply_id = turn.reply_id if eligible else ""
                self._discard_turn_locked(turn)
                turn.committed_audio_ms = max(0, committed_ms)
            self._condition.notify_all()

        truncate_response = getattr(self._session, "truncate_response", None)
        success = False
        if eligible and callable(truncate_response):
            try:
                success = bool(
                    truncate_response(
                        reply_id=reply_id,
                        audio_end_ms=committed_ms,
                    )
                )
            except Exception as exc:
                with self._condition:
                    self._last_error = type(exc).__name__

        if success:
            self._safe_interrupt(clean_reason)
            with self._condition:
                current = self._turn
                if (
                    current is not None
                    and current.generation == target_generation
                ):
                    current.truncated = True
                self._truncation_count += 1
                self._condition.notify_all()
            return True

        self.discard_current(
            f"{clean_reason}_truncate_failed",
            expected_generation=target_generation,
        )
        return False

    def status_snapshot(self) -> dict[str, Any]:
        provider_identity: dict[str, Any] = {}
        try:
            session_status = self._session.status_snapshot()
        except Exception:
            session_status = {}
        if isinstance(session_status, dict):
            for key in ("provider", "model", "provider_session_id"):
                value = session_status.get(key)
                if value not in (None, ""):
                    provider_identity[key] = value
            provider_dialog_id = session_status.get("dialog_id")
            if provider_dialog_id not in (None, ""):
                provider_identity["provider_dialog_id"] = provider_dialog_id
        with self._condition:
            turn = self._turn
            return {
                **provider_identity,
                "mode": self._mode,
                "active": self._active and not self._stop.is_set(),
                "generation": turn.generation if turn else 0,
                "transcript_chars": len(turn.transcript) if turn else 0,
                "transcript_final": bool(turn and turn.transcript_final),
                "response_text_chars": len(turn.text) if turn else 0,
                "pending_audio_frames": len(turn.pending_audio) if turn else 0,
                "pending_audio_ms": round(turn.pending_audio_ms, 2) if turn else 0.0,
                "approved": bool(turn and turn.approved),
                "discarded": bool(turn and turn.discarded),
                "quarantined": self._quarantined,
                "overflow_count": self._overflow_count,
                "dropped_late_audio": self._dropped_late_audio,
                "shadow_audio_frames": self._shadow_audio_frames,
                "rollback_count": self._rollback_count,
                "rollback_failures": self._rollback_failures,
                "rollback_queue_depth": len(self._rollback_queue),
                "rollback_generation": self._rollback_generation,
                "truncation_count": self._truncation_count,
                "truncated": bool(turn and turn.truncated),
                "committed_audio_ms": turn.committed_audio_ms if turn else 0,
                "generation_epoch": self._generation_epoch,
                "max_generation_seen": self._max_generation_seen,
                "recovering": self._recovering,
                "recovery_count": self._recovery_count,
                "recovery_failures": self._recovery_failures,
                "session_id": self._context.session_id,
                "dialog_id": self._context.dialog_id,
                "last_error": self._last_error,
            }

    def close(self, reason: str = "shutdown") -> None:
        with self._condition:
            if not self._active and self._stop.is_set() and not self._recovering:
                return
            self._lifecycle_epoch += 1
            self._active = False
            self._recovering = False
            self._quarantined = True
            self._stop.set()
            if self._turn is not None:
                self._turn.completed.set()
            consumer = self._consumer
            rollback = self._rollback_thread
            self._condition.notify_all()
        try:
            self._session.close(reason)
        finally:
            if consumer is not None and consumer is not threading.current_thread():
                consumer.join(timeout=1.0)
            if rollback is not None and rollback is not threading.current_thread():
                rollback.join(timeout=1.0)
            with self._condition:
                if self._consumer is consumer and (
                    consumer is None or not consumer.is_alive()
                ):
                    self._consumer = None

    def _consume_events(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._session.next_event(timeout=0.1)
            except Exception as exc:
                with self._condition:
                    self._last_error = type(exc).__name__
                    self._condition.notify_all()
                self.close("event_consumer_failure")
                return
            if event is None:
                continue
            try:
                self._handle_event(event)
            except Exception as exc:
                with self._condition:
                    self._last_error = type(exc).__name__
                    self._condition.notify_all()
                self.close("event_handler_failure")
                return

    def _handle_event(self, event: RealtimeVoiceEvent) -> None:
        emit_frame: VoiceMediaFrame | None = None
        emit_final = False
        interrupt_reason = ""
        close_reason = ""
        with self._condition:
            if event.session_id and event.session_id != self._context.session_id:
                if event.event_type is RealtimeVoiceEventType.OUTPUT_AUDIO:
                    self._dropped_late_audio += 1
                self._condition.notify_all()
                return
            if event.event_type is RealtimeVoiceEventType.INPUT_SPEECH_STARTED:
                generation = self._provider_generation_to_effective(event.generation)
                self._max_generation_seen = max(self._max_generation_seen, generation)
                previous = self._turn
                question_id = str(event.metadata.get("question_id", "") or "")
                if previous is not None and previous.generation == generation:
                    if question_id:
                        previous.question_id = question_id
                    self._condition.notify_all()
                    return
                if previous is not None and previous.generation != generation:
                    previous.discarded = True
                    previous.completed.set()
                self._turn = _TurnBuffer(
                    generation=generation,
                    question_id=question_id,
                )
                self._condition.notify_all()
                return

            turn = self._turn
            generation = self._provider_generation_to_effective(event.generation)
            self._max_generation_seen = max(self._max_generation_seen, generation)
            if turn is None:
                turn = _TurnBuffer(generation=generation)
                self._turn = turn
            elif event.generation and generation != turn.generation:
                if generation < turn.generation:
                    if event.event_type is RealtimeVoiceEventType.OUTPUT_AUDIO:
                        self._dropped_late_audio += 1
                    return
                turn.discarded = True
                turn.completed.set()
                turn = _TurnBuffer(generation=generation)
                self._turn = turn

            question_id = str(event.metadata.get("question_id", "") or "")
            reply_id = str(event.metadata.get("reply_id", "") or "")
            if question_id:
                turn.question_id = question_id
            if reply_id:
                turn.reply_id = reply_id

            if event.event_type is RealtimeVoiceEventType.INPUT_TRANSCRIPT_DELTA:
                if event.transcript:
                    turn.transcript = event.transcript.strip()
            elif event.event_type is RealtimeVoiceEventType.INPUT_TRANSCRIPT_FINAL:
                if event.transcript:
                    turn.transcript = event.transcript.strip()
                turn.transcript_final = True
            elif event.event_type is RealtimeVoiceEventType.RESPONSE_STARTED:
                turn.response_started = True
                turn.text = self._merge_text(turn.text, event.text)
            elif event.event_type is RealtimeVoiceEventType.OUTPUT_TEXT_DELTA:
                turn.text = self._merge_text(turn.text, event.text)
            elif event.event_type is RealtimeVoiceEventType.OUTPUT_AUDIO:
                if event.audio is None:
                    return
                if self._mode == "shadow":
                    self._shadow_audio_frames += 1
                    return
                if turn.discarded:
                    self._dropped_late_audio += 1
                    return
                if turn.approved:
                    emit_frame = event.audio
                else:
                    turn.pending_audio.append(event.audio)
                    turn.pending_audio_ms += event.audio.duration_ms
                    if turn.pending_audio_ms > self._pending_output_ms:
                        turn.pending_audio.clear()
                        turn.pending_audio_ms = 0.0
                        turn.discarded = True
                        turn.overflowed = True
                        turn.completed.set()
                        self._overflow_count += 1
                        interrupt_reason = "pending_audio_overflow"
            elif event.event_type is RealtimeVoiceEventType.RESPONSE_DONE:
                turn.text = self._merge_text(turn.text, event.text)
                turn.completed.set()
                emit_final = turn.approved and not turn.discarded
            elif event.event_type is RealtimeVoiceEventType.INTERRUPTED:
                turn.discarded = True
                turn.pending_audio.clear()
                turn.pending_audio_ms = 0.0
                turn.completed.set()
            elif event.event_type is RealtimeVoiceEventType.ERROR:
                turn.discarded = True
                turn.pending_audio.clear()
                turn.pending_audio_ms = 0.0
                turn.completed.set()
                self._last_error = event.error[:160]
                close_reason = "provider_error"
            elif event.event_type is RealtimeVoiceEventType.SESSION_CLOSED:
                turn.discarded = True
                turn.pending_audio.clear()
                turn.pending_audio_ms = 0.0
                turn.completed.set()
                self._last_error = self._last_error or "provider_session_closed"
                close_reason = "provider_session_closed"
            self._condition.notify_all()

        if close_reason:
            self.close(close_reason)
            return
        if interrupt_reason:
            self.discard_current(
                interrupt_reason,
                expected_generation=generation,
            )
        if emit_frame is not None:
            self._emit_audio(emit_frame, final=False)
        if emit_final:
            self._emit_final_marker()

    def _emit_audio(self, frame: VoiceMediaFrame, *, final: bool) -> None:
        with self._audio_emit_lock:
            sink = self._audio_sink
            if sink is None:
                return
            try:
                sink(frame, final)
            except Exception as exc:
                with self._condition:
                    self._last_error = type(exc).__name__
                self.discard_current("audio_sink_failure")

    def _emit_final_marker(self) -> None:
        self._emit_audio(
            VoiceMediaFrame(
                pcm=b"",
                sample_rate=self._context.output_sample_rate,
                channels=1,
            ),
            final=True,
        )

    def _safe_interrupt(self, reason: str) -> None:
        try:
            self._session.interrupt(reason)
        except Exception as exc:
            with self._condition:
                self._last_error = type(exc).__name__

    @staticmethod
    def _discard_turn_locked(turn: _TurnBuffer) -> None:
        turn.discarded = True
        turn.pending_audio.clear()
        turn.pending_audio_ms = 0.0
        turn.completed.set()

    def _schedule_history_rollback(
        self,
        *,
        target_generation: int,
        reason: str,
        turn: _TurnBuffer | None,
        allow_missing: bool,
    ) -> None:
        generation = max(1, int(target_generation or 0))
        request = _RollbackRequest(
            generation=generation,
            reason=reason,
            turn=turn,
            allow_missing=allow_missing,
        )
        with self._condition:
            if generation in self._rolled_back_generations:
                if not self._rollback_pending_generations and self._active:
                    self._quarantined = False
                self._condition.notify_all()
                return
            if generation in self._rollback_pending_generations:
                return
            self._rollback_pending_generations.add(generation)
            self._rollback_queue.append(request)
            rollback = self._rollback_thread
            if rollback is not None and rollback.is_alive():
                self._condition.notify_all()
                return
            rollback = threading.Thread(
                target=self._rollback_worker,
                name="realtime-dialogue-rollback",
                daemon=True,
            )
            self._rollback_thread = rollback
            rollback.start()

    def _rollback_worker(self) -> None:
        while True:
            with self._condition:
                if self._stop.is_set() or not self._active:
                    self._rollback_queue.clear()
                    self._rollback_pending_generations.clear()
                    self._rollback_generation = 0
                    self._rollback_thread = None
                    self._condition.notify_all()
                    return
                if not self._rollback_queue:
                    self._rollback_generation = 0
                    self._rollback_thread = None
                    self._quarantined = False
                    self._condition.notify_all()
                    return
                request = self._rollback_queue.popleft()
                self._rollback_generation = request.generation

            result = self._rollback_one(request)
            close_reason = ""
            with self._condition:
                self._rollback_pending_generations.discard(request.generation)
                if result == "closed":
                    self._rollback_queue.clear()
                    self._rollback_pending_generations.clear()
                    self._rollback_generation = 0
                    self._rollback_thread = None
                    self._condition.notify_all()
                    return
                if result == "success":
                    self._rollback_count += 1
                    self._rolled_back_generations.add(request.generation)
                else:
                    self._rollback_failures += 1
                    self._last_error = "conversation_rollback_failed"
                    self._rollback_queue.clear()
                    self._rollback_pending_generations.clear()
                    self._rollback_generation = 0
                    self._rollback_thread = None
                    self._active = False
                    close_reason = "conversation_rollback_failed"
                self._condition.notify_all()

            if close_reason:
                # A rejected turn that cannot be removed must never share
                # future context.  Close only this optional lane; the local
                # cascade remains available.
                self.close(close_reason)
                return

    def _rollback_one(self, request: _RollbackRequest) -> str:
        deadline = time.monotonic() + 2.0
        turn = request.turn
        bound_after_delay = False
        item_id = ""
        if turn is None:
            with self._condition:
                while (
                    self._active
                    and not self._stop.is_set()
                    and time.monotonic() < deadline
                ):
                    current = self._turn
                    if current is not None and current.generation == request.generation:
                        turn = current
                        bound_after_delay = True
                        self._discard_turn_locked(current)
                        self._condition.notify_all()
                        break
                    if current is not None and current.generation > request.generation:
                        return "failure"
                    self._condition.wait(timeout=0.05)
            if turn is None:
                with self._condition:
                    if self._stop.is_set() or not self._active:
                        return "closed"
                return "absent" if request.allow_missing else "failure"

        if turn.generation != request.generation:
            return "failure"
        if bound_after_delay:
            self._safe_interrupt(request.reason)

        with self._condition:
            while (
                self._active
                and not self._stop.is_set()
                and time.monotonic() < deadline
            ):
                item_id = turn.question_id or turn.reply_id
                if item_id:
                    break
                self._condition.wait(timeout=0.05)
            if self._stop.is_set() or not self._active:
                return "closed"

        delete_turn = getattr(self._session, "delete_conversation_turn", None)
        remaining = max(0.05, deadline - time.monotonic())
        try:
            success = bool(
                item_id
                and callable(delete_turn)
                and delete_turn(item_id, timeout=remaining)
            )
        except Exception:
            success = False
        with self._condition:
            if self._stop.is_set() or not self._active:
                return "closed"
        return "success" if success else "failure"

    def _transcripts_match(self, local: str, provider: str) -> bool:
        left = self._normalize_transcript(local)
        right = self._normalize_transcript(provider)
        if not left or not right:
            return False
        if left == right or left in right or right in left:
            return True
        return SequenceMatcher(None, left, right).ratio() >= self._transcript_match_threshold

    @staticmethod
    def _normalize_transcript(value: str) -> str:
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(value or "").lower())

    @staticmethod
    def _merge_text(current: str, incoming: str) -> str:
        new = str(incoming or "")
        if not new:
            return current
        if not current:
            return new
        if new.startswith(current):
            return new
        if current.endswith(new):
            return current
        return current + new

    def _provider_generation_to_effective(self, provider_generation: int) -> int:
        return self._generation_epoch + max(1, int(provider_generation or 0))

    def _new_recovery_session_id(self, prior_max_generation: int) -> str:
        base = str(self._context.session_id or "realtime-session").strip()
        return f"{base}-r{self._recovery_count + 1}-g{prior_max_generation}-{uuid4().hex[:8]}"
