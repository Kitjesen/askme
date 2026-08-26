"""Behavior tests for per-turn cancellation."""

from __future__ import annotations

import asyncio

import pytest

from askme.pipeline.core.turn_control import TurnCancellationController


@pytest.mark.asyncio
async def test_compatibility_task_lease_does_not_advance_media_epoch() -> None:
    controller = TurnCancellationController(asyncio.Event())

    compatibility_lease = controller.begin_turn(owner="text")
    try:
        assert controller.epoch == 0

        canonical_lease = controller.begin("voice-turn")
        try:
            assert canonical_lease.epoch == 1
            assert controller.epoch == 1
        finally:
            controller.finish(canonical_lease)
    finally:
        controller.end_turn(compatibility_lease)


@pytest.mark.asyncio
async def test_cancel_current_turn_from_thread_cancels_only_active_lease() -> None:
    emergency_token = asyncio.Event()
    controller = TurnCancellationController(emergency_token)
    started = asyncio.Event()
    cancellation_seen: list[tuple[bool, str | None]] = []

    async def _run_turn() -> None:
        lease = controller.begin_turn()
        try:
            started.set()
            await asyncio.Future()
        except asyncio.CancelledError:
            cancellation_seen.append(
                (controller.token.is_set(), lease.reason)
            )
            raise
        finally:
            controller.end_turn(lease)

    task = asyncio.create_task(_run_turn())
    await started.wait()

    cancelled = await asyncio.to_thread(
        controller.cancel_current_turn,
        reason="barge_in",
    )

    assert cancelled is True
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancellation_seen == [(True, "barge_in")]
    assert controller.token.is_set() is False
    assert emergency_token.is_set() is False


@pytest.mark.asyncio
async def test_new_turn_clears_barge_in_but_emergency_stop_stays_sticky() -> None:
    emergency_token = asyncio.Event()
    controller = TurnCancellationController(emergency_token)

    interrupted = controller.begin_turn()
    assert controller.cancel_current_turn(reason="barge_in") is True
    assert controller.token.is_set() is True
    controller.end_turn(interrupted)

    next_turn = controller.begin_turn()
    assert controller.token.is_set() is False
    emergency_token.set()
    assert controller.token.is_set() is True
    controller.end_turn(next_turn)

    final_turn = controller.begin_turn()
    assert controller.token.is_set() is True
    emergency_token.clear()
    assert controller.token.is_set() is False
    controller.end_turn(final_turn)
    assert controller.cancel_current_turn() is False


@pytest.mark.asyncio
async def test_concurrent_turns_observe_only_their_own_lease() -> None:
    controller = TurnCancellationController(asyncio.Event())
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release_first = asyncio.Event()

    async def _turn(started: asyncio.Event, release: asyncio.Event) -> bool:
        lease = controller.begin_turn()
        try:
            started.set()
            await release.wait()
            return controller.token.is_set()
        except asyncio.CancelledError:
            return controller.token.is_set()
        finally:
            controller.end_turn(lease)

    first = asyncio.create_task(_turn(first_started, release_first))
    await first_started.wait()
    second = asyncio.create_task(_turn(second_started, asyncio.Event()))
    await second_started.wait()

    assert controller.cancel_current_turn() is True
    assert await second is True
    release_first.set()
    assert await first is False


@pytest.mark.asyncio
async def test_owner_filter_cancels_voice_when_text_turn_started_later() -> None:
    controller = TurnCancellationController(asyncio.Event())
    voice_started = asyncio.Event()
    text_started = asyncio.Event()
    release_voice = asyncio.Event()
    release_text = asyncio.Event()

    async def _turn(
        owner: str,
        started: asyncio.Event,
        release: asyncio.Event,
    ) -> bool:
        started.set()
        lease = controller.begin_turn(owner=owner)
        try:
            await release.wait()
            return controller.token.is_set()
        except asyncio.CancelledError:
            return controller.token.is_set()
        finally:
            controller.end_turn(lease)

    voice = asyncio.create_task(_turn("voice", voice_started, release_voice))
    await voice_started.wait()
    text = asyncio.create_task(_turn("text", text_started, release_text))
    await text_started.wait()

    try:
        assert controller.cancel_current_turn(owner="voice") is True
        assert await voice is True
        assert controller.cancel_current_turn(owner="voice") is False
        assert text.done() is False
        release_text.set()
        assert await text is False
    finally:
        release_voice.set()
        release_text.set()
        await asyncio.gather(voice, text, return_exceptions=True)
