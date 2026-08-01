"""Product lifecycle tests for provider warm-session management."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from askme.runtime.warm_sessions import (
    WarmSessionBinding,
    WarmSessionManager,
    WarmSessionPolicy,
    WarmSessionResult,
)


class _RecordingTarget:
    name = "llm"

    def __init__(self) -> None:
        self.called = asyncio.Event()
        self.calls = 0
        self.cancelled = False
        self.force_refresh_values: list[bool] = []

    async def ensure_warm(self, *, force_refresh: bool) -> WarmSessionResult:
        self.calls += 1
        self.force_refresh_values.append(force_refresh)
        self.called.set()
        return WarmSessionResult(
            ok=True,
            status="opened",
            elapsed_ms=12.5,
            reused=False,
            provider_session_key="test-provider:g1",
        )

    def cancel(self) -> None:
        self.cancelled = True


async def test_manager_warms_target_at_start_and_exposes_snapshot() -> None:
    target = _RecordingTarget()
    manager = WarmSessionManager(
        [
            WarmSessionBinding(
                name="llm",
                target=target,
                policy=WarmSessionPolicy(refresh_interval_seconds=60.0),
            )
        ]
    )

    await manager.start()
    try:
        await asyncio.wait_for(target.called.wait(), timeout=1.0)

        snapshot = manager.snapshot()
        assert snapshot["status"] == "running"
        target_snapshot = snapshot["targets"]["llm"]
        assert target_snapshot["status"] == "warm"
        assert target_snapshot["attempts"] == 1
        assert target_snapshot["successes"] == 1
        assert target_snapshot["failures"] == 0
        assert target_snapshot["skips"] == 0
        assert target_snapshot["consecutive_failures"] == 0
        assert target_snapshot["last_result"] == "opened"
        assert target_snapshot["last_reason"] == ""
        assert target_snapshot["last_latency_ms"] == 12.5
        assert target_snapshot["provider_session_key"] == "test-provider:g1"
        assert target_snapshot["last_success_age_seconds"] >= 0.0
        assert target_snapshot["next_attempt_in_seconds"] > 50.0
        assert target_snapshot["attempt_budget_remaining"] == 59
    finally:
        await manager.stop()


async def test_manager_refreshes_target_periodically_until_stopped() -> None:
    target = _RecordingTarget()
    manager = WarmSessionManager(
        [
            WarmSessionBinding(
                name="llm",
                target=target,
                policy=WarmSessionPolicy(
                    refresh_interval_seconds=0.02,
                    jitter_ratio=0.0,
                    max_attempts_per_hour=100,
                ),
            )
        ]
    )

    await manager.start()
    try:
        deadline = asyncio.get_running_loop().time() + 1.0
        while target.calls < 3 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.005)
        assert target.calls >= 3
        assert all(target.force_refresh_values)
    finally:
        await manager.stop()


async def test_manager_request_refresh_wakes_named_target_immediately() -> None:
    target = _RecordingTarget()
    manager = WarmSessionManager(
        [
            WarmSessionBinding(
                name="llm",
                target=target,
                policy=WarmSessionPolicy(
                    refresh_interval_seconds=60.0,
                    jitter_ratio=0.0,
                ),
            )
        ]
    )

    await manager.start()
    try:
        await asyncio.wait_for(target.called.wait(), timeout=1.0)
        assert target.calls == 1
        assert manager.request_refresh("missing") is False
        assert await asyncio.to_thread(manager.request_refresh, "llm") is True

        deadline = asyncio.get_running_loop().time() + 0.5
        while target.calls < 2 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.005)
        assert target.calls == 2
    finally:
        await manager.stop()

    assert manager.request_refresh("llm") is False


class _CancellationIgnoringTarget:
    name = "tts"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancel_calls = 0

    async def ensure_warm(self, *, force_refresh: bool) -> WarmSessionResult:
        self.started.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                continue
        return WarmSessionResult(ok=False, status="cancelled", neutral=True)

    def cancel(self) -> None:
        self.cancel_calls += 1


async def test_manager_stop_is_bounded_and_invokes_target_cancel() -> None:
    target = _CancellationIgnoringTarget()
    manager = WarmSessionManager(
        [
            WarmSessionBinding(
                name="tts",
                target=target,
                policy=WarmSessionPolicy(refresh_interval_seconds=60.0),
            )
        ],
        shutdown_timeout_seconds=0.02,
    )

    await manager.start()
    await asyncio.wait_for(target.started.wait(), timeout=1.0)
    started_at = asyncio.get_running_loop().time()
    try:
        await manager.stop()
        assert asyncio.get_running_loop().time() - started_at < 0.15
        assert target.cancel_calls == 1
        assert manager.snapshot()["status"] == "stopped"
    finally:
        target.release.set()
        await asyncio.sleep(0)


class _TimeoutTarget:
    name = "llm"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancel_calls = 0

    async def ensure_warm(self, *, force_refresh: bool) -> WarmSessionResult:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    def cancel(self) -> None:
        self.cancel_calls += 1


async def test_manager_cancels_provider_work_when_attempt_times_out() -> None:
    target = _TimeoutTarget()
    manager = WarmSessionManager(
        [
            WarmSessionBinding(
                name="llm",
                target=target,
                policy=WarmSessionPolicy(
                    refresh_interval_seconds=60.0,
                    timeout_seconds=0.02,
                    initial_backoff_seconds=60.0,
                    max_backoff_seconds=60.0,
                    jitter_ratio=0.0,
                ),
            )
        ]
    )

    await manager.start()
    try:
        await asyncio.wait_for(target.started.wait(), timeout=1.0)
        deadline = asyncio.get_running_loop().time() + 1.0
        snapshot = manager.snapshot()
        while (
            snapshot["targets"]["llm"]["failures"] < 1
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.005)
            snapshot = manager.snapshot()

        assert target.cancel_calls == 1
        assert snapshot["targets"]["llm"]["status"] == "degraded"
        assert snapshot["targets"]["llm"]["last_reason"] == "TimeoutError"
    finally:
        await manager.stop()


class _CancellationSuppressingTimeoutTarget:
    name = "llm"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancellation_seen = asyncio.Event()
        self.cancel_calls = 0

    async def ensure_warm(self, *, force_refresh: bool) -> WarmSessionResult:
        self.started.set()
        while not self.release.is_set():
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancellation_seen.set()
        return WarmSessionResult(ok=False, status="cancelled", neutral=True)

    def cancel(self) -> None:
        self.cancel_calls += 1


async def test_attempt_timeout_is_bounded_when_target_suppresses_cancellation() -> None:
    target = _CancellationSuppressingTimeoutTarget()
    manager = WarmSessionManager(
        [
            WarmSessionBinding(
                name="llm",
                target=target,
                policy=WarmSessionPolicy(
                    refresh_interval_seconds=60.0,
                    timeout_seconds=0.02,
                    initial_backoff_seconds=60.0,
                    max_backoff_seconds=60.0,
                    jitter_ratio=0.0,
                ),
            )
        ]
    )

    await manager.start()
    try:
        await asyncio.wait_for(target.started.wait(), timeout=1.0)
        deadline = asyncio.get_running_loop().time() + 0.5
        snapshot = manager.snapshot()["targets"]["llm"]
        while snapshot["failures"] < 1 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.005)
            snapshot = manager.snapshot()["targets"]["llm"]

        assert snapshot["status"] == "degraded"
        assert snapshot["last_reason"] == "TimeoutError"
        assert target.cancel_calls == 1
        await asyncio.wait_for(target.cancellation_seen.wait(), timeout=0.2)
    finally:
        target.release.set()
        await manager.stop()
        await asyncio.sleep(0)


async def test_manager_holds_until_budget_window_resets_after_limit(monkeypatch) -> None:
    monkeypatch.setattr("askme.runtime.warm_sessions.random.uniform", lambda low, high: low)
    target = _RecordingTarget()
    manager = WarmSessionManager(
        [
            WarmSessionBinding(
                name="llm",
                target=target,
                policy=WarmSessionPolicy(
                    refresh_interval_seconds=0.01,
                    jitter_ratio=0.5,
                    max_attempts_per_hour=1,
                ),
            )
        ]
    )

    await manager.start()
    try:
        deadline = asyncio.get_running_loop().time() + 1.0
        snapshot = manager.snapshot()
        while (
            snapshot["targets"]["llm"]["status"] != "throttled"
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.005)
            snapshot = manager.snapshot()

        target_snapshot = snapshot["targets"]["llm"]
        assert target.calls == 1
        assert target_snapshot["attempts"] == 1
        assert target_snapshot["skips"] == 1
        assert target_snapshot["attempt_budget_remaining"] == 0
        assert target_snapshot["next_attempt_in_seconds"] > 3500.0
    finally:
        await manager.stop()


class _BusyTarget:
    name = "tts"

    def __init__(self) -> None:
        self.calls = 0
        self.second_call = asyncio.Event()

    async def ensure_warm(self, *, force_refresh: bool) -> WarmSessionResult:
        self.calls += 1
        if self.calls >= 2:
            self.second_call.set()
        return WarmSessionResult(
            ok=False,
            status="busy",
            reason="real_request_priority",
            neutral=True,
        )

    def cancel(self) -> None:
        return


async def test_busy_result_is_neutral_and_retries_without_failure_backoff() -> None:
    target = _BusyTarget()
    manager = WarmSessionManager(
        [
            WarmSessionBinding(
                name="tts",
                target=target,
                policy=WarmSessionPolicy(
                    refresh_interval_seconds=60.0,
                    initial_backoff_seconds=0.2,
                    max_backoff_seconds=0.2,
                    busy_retry_seconds=0.01,
                    jitter_ratio=0.0,
                    max_attempts_per_hour=1,
                ),
            )
        ]
    )

    await manager.start()
    try:
        await asyncio.wait_for(target.second_call.wait(), timeout=0.15)
        snapshot = manager.snapshot()["targets"]["tts"]
        assert target.calls >= 2
        assert snapshot["status"] == "busy"
        assert snapshot["failures"] == 0
        assert snapshot["skips"] >= 2
        assert snapshot["consecutive_failures"] == 0
        assert snapshot["last_result"] == "busy"
        assert snapshot["attempt_budget_remaining"] == 1
    finally:
        await manager.stop()


class _WorkerBusyTarget:
    name = "tts"

    def __init__(self) -> None:
        self.called = asyncio.Event()

    async def ensure_warm(self, *, force_refresh: bool) -> WarmSessionResult:
        _ = force_refresh
        self.called.set()
        return WarmSessionResult(
            ok=False,
            status="busy",
            reason="prewarm_already_running",
            neutral=True,
            details={"prewarm_worker_age_seconds": 0.01},
        )

    def cancel(self) -> None:
        return


async def test_short_worker_busy_remains_neutral_and_snapshot_age_increases() -> None:
    target = _WorkerBusyTarget()
    manager = WarmSessionManager(
        [
            WarmSessionBinding(
                name="tts",
                target=target,
                policy=WarmSessionPolicy(
                    refresh_interval_seconds=60.0,
                    timeout_seconds=60.0,
                    busy_retry_seconds=60.0,
                    jitter_ratio=0.0,
                ),
            )
        ]
    )

    await manager.start()
    try:
        await asyncio.wait_for(target.called.wait(), timeout=1.0)
        first = manager.snapshot()["targets"]["tts"]
        await asyncio.sleep(0.02)
        second = manager.snapshot()["targets"]["tts"]

        assert first["status"] == "busy"
        assert first["last_result"] == "busy"
        assert first["failures"] == 0
        assert first["skips"] == 1
        assert first["active_worker_age_seconds"] < first["stuck_busy_threshold_seconds"]
        assert second["status"] == "busy"
        assert second["failures"] == 0
        assert second["active_worker_age_seconds"] > first["active_worker_age_seconds"]
    finally:
        await manager.stop()


class _FailingTarget:
    name = "llm"

    def __init__(self) -> None:
        self.call_times: list[float] = []
        self.third_call = asyncio.Event()

    async def ensure_warm(self, *, force_refresh: bool) -> WarmSessionResult:
        self.call_times.append(time.perf_counter())
        if len(self.call_times) >= 3:
            self.third_call.set()
        return WarmSessionResult(ok=False, status="failed", reason="upstream")

    def cancel(self) -> None:
        return


async def test_failures_use_capped_exponential_backoff() -> None:
    target = _FailingTarget()
    manager = WarmSessionManager(
        [
            WarmSessionBinding(
                name="llm",
                target=target,
                policy=WarmSessionPolicy(
                    refresh_interval_seconds=60.0,
                    initial_backoff_seconds=0.03,
                    max_backoff_seconds=0.06,
                    jitter_ratio=0.0,
                ),
            )
        ]
    )

    await manager.start()
    try:
        await asyncio.wait_for(target.third_call.wait(), timeout=1.0)
        first_gap = target.call_times[1] - target.call_times[0]
        second_gap = target.call_times[2] - target.call_times[1]
        snapshot = manager.snapshot()["targets"]["llm"]

        assert first_gap >= 0.02
        assert second_gap >= 0.05
        assert snapshot["failures"] == 3
        assert snapshot["consecutive_failures"] == 3
        assert snapshot["status"] == "degraded"
    finally:
        await manager.stop()


def test_tts_prewarm_worker_cannot_delay_asyncio_run_shutdown() -> None:
    script = textwrap.dedent(
        """
        import asyncio
        import threading

        from askme.runtime.warm_session_targets import TTSWarmSessionTarget
        from askme.runtime.warm_sessions import (
            WarmSessionBinding,
            WarmSessionManager,
            WarmSessionPolicy,
        )

        class BlockingTTS:
            backend = "blocking"

            def __init__(self):
                self.started = threading.Event()
                self.never = threading.Event()

            def prewarm_provider_session(self, *, force_refresh):
                self.started.set()
                self.never.wait(30.0)
                return {"ok": True}

            def cancel_provider_prewarm(self):
                return None

        async def main():
            engine = BlockingTTS()
            manager = WarmSessionManager(
                [
                    WarmSessionBinding(
                        name="tts",
                        target=TTSWarmSessionTarget(lambda: engine),
                        policy=WarmSessionPolicy(
                            refresh_interval_seconds=60.0,
                            timeout_seconds=60.0,
                        ),
                    )
                ],
                shutdown_timeout_seconds=0.02,
            )
            await manager.start()
            for _ in range(200):
                if engine.started.is_set():
                    break
                await asyncio.sleep(0.005)
            assert engine.started.is_set()
            await manager.stop()

        asyncio.run(main())
        print("bounded-exit")
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "bounded-exit" in completed.stdout


class _NeverReturningWarmModuleTTS:
    backend = "stuck-tts"

    def __init__(self) -> None:
        import threading

        self.started = threading.Event()
        self.never = threading.Event()
        self.cancel_calls = 0

    def prewarm_provider_session(self, *, force_refresh: bool):
        _ = force_refresh
        self.started.set()
        self.never.wait(30.0)
        return {"ok": True, "status": "refreshed"}

    def cancel_provider_prewarm(self) -> None:
        self.cancel_calls += 1


async def test_detached_stuck_tts_prewarm_degrades_health_after_restart_and_refresh() -> None:
    from askme.runtime.module import ModuleRegistry

    from askme.runtime.modules.warm_session_module import WarmSessionModule

    tts = _NeverReturningWarmModuleTTS()
    registry = ModuleRegistry()
    registry.register(_WarmModuleVoice(tts))
    module = WarmSessionModule()
    module.build(
        {
            "warm_sessions": {
                "enabled": True,
                "shutdown_timeout_seconds": 0.01,
                "llm": {"enabled": False},
                "tts": {
                    "enabled": True,
                    "startup_delay_seconds": 0.0,
                    "refresh_interval_seconds": 60.0,
                    "timeout_seconds": 0.05,
                    "initial_backoff_seconds": 60.0,
                    "max_backoff_seconds": 60.0,
                    "busy_retry_seconds": 0.01,
                    "jitter_ratio": 0.0,
                },
            }
        },
        registry,
    )

    await module.start()
    await asyncio.to_thread(tts.started.wait, 1.0)
    started_at = asyncio.get_running_loop().time()
    await module.stop()
    assert asyncio.get_running_loop().time() - started_at < 0.2
    assert tts.cancel_calls == 1

    await asyncio.sleep(0.06)
    await module.start()
    try:
        assert module.request_refresh("tts") is True
        deadline = asyncio.get_running_loop().time() + 1.0
        health = module.health()
        while "tts" not in health["degraded_targets"]:
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("stuck detached TTS worker did not degrade health")
            await asyncio.sleep(0.005)
            health = module.health()

        target = health["targets"]["tts"]
        assert health["status"] == "ok"
        assert health["latency_warm"] is False
        assert target["status"] == "degraded"
        assert target["last_result"] == "stuck_busy"
        assert target["last_reason"] == "prewarm_already_running"
        assert target["active_worker_age_seconds"] >= target["stuck_busy_threshold_seconds"]
        assert target["stuck_busy_threshold_seconds"] == 0.05
        first_age = target["active_worker_age_seconds"]
        await asyncio.sleep(0.02)
        refreshed_target = module.health()["targets"]["tts"]
        assert refreshed_target["last_result"] == "stuck_busy"
        assert refreshed_target["active_worker_age_seconds"] > first_age
    finally:
        await module.stop()


class _FakeLLMClient:
    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name
        self.calls: list[dict[str, object]] = []
        self.yielded: list[str] = []

    async def chat_stream(self, _messages, **kwargs):
        self.calls.append(kwargs)
        for chunk in ("first", "final"):
            self.yielded.append(chunk)
            yield chunk

    def request_activity(self) -> dict[str, int]:
        return {
            "active_business_requests": 0,
            "active_warm_probes": 0,
        }


async def test_llm_target_uses_health_probe_context_and_resolves_hot_switches() -> None:
    from askme.runtime.warm_session_targets import LLMWarmSessionTarget

    first = _FakeLLMClient("first-provider")
    second = _FakeLLMClient("second-provider")
    current: list[tuple[_FakeLLMClient, str]] = [(first, "voice-model-a")]
    target = LLMWarmSessionTarget(lambda: current[0])

    first_result = await target.ensure_warm(force_refresh=True)
    current[0] = (second, "voice-model-b")
    second_result = await target.ensure_warm(force_refresh=True)

    assert first.yielded == ["first", "final"]
    assert second.yielded == ["first", "final"]
    assert first.calls[0]["model"] == "voice-model-a"
    assert second.calls[0]["model"] == "voice-model-b"
    assert first.calls[0]["max_tokens"] == 1
    assert first.calls[0]["temperature"] == 0.0
    assert isinstance(first.calls[0]["cancel_token"], asyncio.Event)
    context = first.calls[0]["context"]
    assert context.purpose == "health_probe"
    assert context.request_class == "health_probe"
    assert context.allow_cache is False
    assert first_result.ok is True
    assert first_result.status == "probed"
    assert first_result.provider_session_key == "first-provider:voice-model-a"
    assert second_result.provider_session_key == "second-provider:voice-model-b"


class _BusinessActiveLLMClient:
    provider_name = "test-provider"
    model = "test-model"

    def __init__(self, *, business_active: bool = True) -> None:
        self.calls = 0
        self.business_active = business_active

    def request_activity(self) -> dict[str, int]:
        return {
            "active_business_requests": int(self.business_active),
            "active_warm_probes": 0,
        }

    async def chat_stream(self, _messages, **kwargs):
        self.calls += 1
        self.business_active = True
        kwargs["cancel_token"].set()
        if False:
            yield None


async def test_llm_target_uses_facade_warm_lease_to_avoid_resolve_before_activity_race() -> None:
    from askme.runtime.warm_session_targets import LLMWarmSessionTarget

    first = _FakeLLMClient("first-provider")
    second = _FakeLLMClient("second-provider")
    current: list[tuple[_FakeLLMClient, str]] = [(first, "health-old")]
    releases: list[str] = []

    class _Lease:
        def __init__(self, client: _FakeLLMClient, model: str) -> None:
            self.client = client
            self.model = model

        def release(self) -> None:
            releases.append(self.model)

    class _Facade:
        def acquire_warm_target(self) -> _Lease:
            client, model = current[0]
            return _Lease(client, model)

    facade = _Facade()
    target = LLMWarmSessionTarget(lambda: (facade, "stale-resolved-model"))
    current[0] = (second, "health-new")

    result = await target.ensure_warm(force_refresh=True)

    assert result.ok is True
    assert first.calls == []
    assert second.calls[0]["model"] == "health-new"
    assert result.provider_session_key == "second-provider:health-new"
    assert releases == ["health-new"]


async def test_llm_target_maps_business_active_deferral_to_busy() -> None:
    from askme.runtime.warm_session_targets import LLMWarmSessionTarget

    client = _BusinessActiveLLMClient()
    target = LLMWarmSessionTarget(lambda: (client, client.model))

    result = await target.ensure_warm(force_refresh=True)

    assert result.status == "busy"
    assert result.reason == "real_request_priority"
    assert result.neutral is True
    assert result.provider_session_key == "test-provider:test-model"
    assert client.calls == 0


async def test_llm_target_skips_probe_when_request_activity_is_unavailable() -> None:
    from askme.runtime.warm_session_targets import LLMWarmSessionTarget

    class ActivityUnavailableClient(_BusinessActiveLLMClient):
        def request_activity(self) -> dict[str, int]:
            raise RuntimeError("activity tracker unavailable")

    client = ActivityUnavailableClient(business_active=False)
    target = LLMWarmSessionTarget(lambda: (client, client.model))

    result = await target.ensure_warm(force_refresh=True)

    assert result.status == "skipped"
    assert result.reason == "activity_unavailable"
    assert result.neutral is True
    assert result.provider_session_key == "test-provider:test-model"
    assert client.calls == 0


async def test_llm_target_maps_inflight_business_preemption_to_busy() -> None:
    from askme.runtime.warm_session_targets import LLMWarmSessionTarget

    client = _BusinessActiveLLMClient(business_active=False)
    target = LLMWarmSessionTarget(lambda: (client, client.model))

    result = await target.ensure_warm(force_refresh=True)

    assert result.status == "busy"
    assert result.reason == "real_request_priority"
    assert result.neutral is True
    assert client.calls == 1


class _FakeTTSEngine:
    def __init__(self, backend: str, result: dict[str, object]) -> None:
        self.backend = backend
        self.result = result
        self.force_refresh_values: list[bool] = []
        self.cancel_calls = 0

    def prewarm_provider_session(self, *, force_refresh: bool) -> dict[str, object]:
        self.force_refresh_values.append(force_refresh)
        return dict(self.result)

    def cancel_provider_prewarm(self) -> None:
        self.cancel_calls += 1


async def test_tts_target_resolves_live_engine_and_normalizes_busy_result() -> None:
    from askme.runtime.warm_session_targets import TTSWarmSessionTarget

    first = _FakeTTSEngine(
        "minimax",
        {
            "ok": True,
            "status": "refreshed",
            "elapsed_ms": 23.5,
            "reused": False,
        },
    )
    second = _FakeTTSEngine(
        "minimax",
        {
            "ok": False,
            "status": "skipped",
            "reason": "synthesis_busy",
        },
    )
    current = [first]
    target = TTSWarmSessionTarget(lambda: current[0])

    first_result = await target.ensure_warm(force_refresh=True)
    current[0] = second
    second_result = await target.ensure_warm(force_refresh=True)

    assert first.force_refresh_values == [True]
    assert second.force_refresh_values == [True]
    assert first_result == WarmSessionResult(
        ok=True,
        status="refreshed",
        elapsed_ms=23.5,
        reused=False,
        provider_session_key="minimax",
    )
    assert second_result == WarmSessionResult(
        ok=False,
        status="busy",
        reason="synthesis_busy",
        neutral=True,
        provider_session_key="minimax",
    )


class _WarmModuleLLMClient:
    provider_name = "test-provider"
    model = "test-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def request_activity(self) -> dict[str, int]:
        return {
            "active_business_requests": 0,
            "active_warm_probes": 0,
        }

    async def chat_stream(self, _messages, **kwargs):
        self.calls.append(kwargs)
        yield "好"


class _WarmModuleLLM:
    name = "llm"

    def __init__(self, client: _WarmModuleLLMClient) -> None:
        self.client = client

    def resolve_warm_target(self):
        return self.client, self.client.model


class _WarmModuleTTS:
    backend = "minimax"

    def __init__(self) -> None:
        self.calls: list[bool] = []
        self.cancel_calls = 0

    def prewarm_provider_session(self, *, force_refresh: bool):
        self.calls.append(force_refresh)
        return {
            "ok": True,
            "status": "refreshed",
            "elapsed_ms": 4.0,
            "reused": False,
        }

    def cancel_provider_prewarm(self) -> None:
        self.cancel_calls += 1


class _WarmModuleVoice:
    name = "voice"

    def __init__(self, tts_provider: _WarmModuleTTS) -> None:
        self.tts_provider = tts_provider
        self.tts_activation_callback = None

    def set_tts_activation_callback(self, callback) -> None:
        self.tts_activation_callback = callback

    def activate_tts(self, tts_provider: _WarmModuleTTS) -> None:
        self.tts_provider = tts_provider
        if self.tts_activation_callback is not None:
            self.tts_activation_callback(tts_provider)


async def _wait_for_warm_module_calls(
    llm_client: _WarmModuleLLMClient,
    tts: _WarmModuleTTS,
) -> None:
    deadline = asyncio.get_running_loop().time() + 1.0
    while not llm_client.calls or not tts.calls:
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("warm-session module did not start both targets")
        await asyncio.sleep(0.005)


def test_warm_session_module_uses_calibrated_product_defaults() -> None:
    from askme.runtime.module import ModuleRegistry

    from askme.runtime.modules.warm_session_module import WarmSessionModule

    registry = ModuleRegistry()
    registry.register(_WarmModuleLLM(_WarmModuleLLMClient()))
    registry.register(_WarmModuleVoice(_WarmModuleTTS()))
    module = WarmSessionModule()

    module.build({"warm_sessions": {"enabled": True}}, registry)

    policies = {binding.name: binding.policy for binding in module._manager._bindings}
    assert module._manager._shutdown_timeout_seconds == 0.5
    assert policies["llm"].timeout_seconds == 20.0
    assert policies["tts"].timeout_seconds == 10.0


async def test_warm_session_module_runs_from_runtime_start_until_stop() -> None:
    from askme.runtime.module import ModuleRegistry

    from askme.runtime.modules.warm_session_module import WarmSessionModule

    llm_client = _WarmModuleLLMClient()
    tts = _WarmModuleTTS()
    registry = ModuleRegistry()
    registry.register(_WarmModuleLLM(llm_client))
    registry.register(_WarmModuleVoice(tts))
    module = WarmSessionModule()
    module.build(
        {
            "warm_sessions": {
                "enabled": "true",
                "shutdown_timeout_seconds": 0.2,
                "llm": {
                    "enabled": True,
                    "refresh_interval_seconds": 60.0,
                    "jitter_ratio": 0.0,
                },
                "tts": {
                    "enabled": True,
                    "refresh_interval_seconds": 60.0,
                    "jitter_ratio": 0.0,
                },
            }
        },
        registry,
    )

    await module.start()
    try:
        await _wait_for_warm_module_calls(llm_client, tts)
        deadline = asyncio.get_running_loop().time() + 1.0
        while any(target["status"] != "warm" for target in module.health()["targets"].values()):
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("warm-session targets did not become warm")
            await asyncio.sleep(0.005)
        health = module.health()

        assert health["status"] == "ok"
        assert health["enabled"] is True
        assert health["running"] is True
        assert health["latency_warm"] is True
        assert set(health["targets"]) == {"llm", "tts"}
        assert health["targets"]["llm"]["status"] == "warm"
        assert health["targets"]["tts"]["status"] == "warm"
        assert llm_client.calls[0]["model"] == "test-model"
        assert tts.calls == [True]
    finally:
        await module.stop()

    assert module.health()["running"] is False


async def test_tts_activation_requests_immediate_manager_owned_refresh() -> None:
    from askme.runtime.module import ModuleRegistry

    from askme.runtime.modules.warm_session_module import WarmSessionModule

    llm_client = _WarmModuleLLMClient()
    first_tts = _WarmModuleTTS()
    second_tts = _WarmModuleTTS()
    voice = _WarmModuleVoice(first_tts)
    registry = ModuleRegistry()
    registry.register(_WarmModuleLLM(llm_client))
    registry.register(voice)
    module = WarmSessionModule()
    module.build(
        {
            "warm_sessions": {
                "enabled": True,
                "llm": {"enabled": False},
                "tts": {
                    "enabled": True,
                    "startup_delay_seconds": 0.0,
                    "refresh_interval_seconds": 60.0,
                    "jitter_ratio": 0.0,
                },
            }
        },
        registry,
    )

    await module.start()
    try:
        deadline = asyncio.get_running_loop().time() + 1.0
        while not first_tts.calls and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.005)
        assert first_tts.calls == [True]

        voice.activate_tts(second_tts)
        deadline = asyncio.get_running_loop().time() + 0.5
        while not second_tts.calls and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.005)

        assert first_tts.calls == [True]
        assert second_tts.calls == [True]
    finally:
        await module.stop()

    assert voice.tts_activation_callback is None


async def test_configured_tts_target_absence_is_explicit_and_not_latency_warm() -> None:
    from askme.runtime.module import ModuleRegistry

    from askme.runtime.modules.warm_session_module import WarmSessionModule

    llm_client = _WarmModuleLLMClient()
    registry = ModuleRegistry()
    registry.register(_WarmModuleLLM(llm_client))
    module = WarmSessionModule()
    module.build(
        {
            "warm_sessions": {
                "enabled": True,
                "llm": {
                    "enabled": True,
                    "refresh_interval_seconds": 60.0,
                    "jitter_ratio": 0.0,
                },
                "tts": {"enabled": True},
            }
        },
        registry,
    )

    await module.start()
    try:
        deadline = asyncio.get_running_loop().time() + 1.0
        health = module.health()
        while (
            health["targets"]["llm"]["status"] != "warm"
            and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.005)
            health = module.health()

        assert health["status"] == "ok"
        assert health["latency_warm"] is False
        assert health["configured_targets"] == ["llm", "tts"]
        assert health["active_targets"] == ["llm"]
        assert health["unavailable_targets"] == ["tts"]
        assert health["degraded_targets"] == ["tts"]
        assert health["targets"]["tts"]["status"] == "unavailable"
        assert health["targets"]["tts"]["last_reason"] == "voice_module_not_wired"
        assert health["targets"]["tts"]["provider_session_key"] is None
    finally:
        await module.stop()


async def test_warm_session_module_parses_false_string_as_disabled() -> None:
    from askme.runtime.module import ModuleRegistry

    from askme.runtime.modules.warm_session_module import WarmSessionModule

    llm_client = _WarmModuleLLMClient()
    tts = _WarmModuleTTS()
    registry = ModuleRegistry()
    registry.register(_WarmModuleLLM(llm_client))
    registry.register(_WarmModuleVoice(tts))
    module = WarmSessionModule()
    module.build({"warm_sessions": {"enabled": "false"}}, registry)

    await module.start()
    await asyncio.sleep(0.02)

    assert llm_client.calls == []
    assert tts.calls == []
    assert module.health() == {
        "status": "ok",
        "enabled": False,
        "running": False,
        "latency_warm": False,
        "manager_status": "stopped",
        "configured_targets": [],
        "active_targets": [],
        "unavailable_targets": [],
        "degraded_targets": [],
        "targets": {},
    }
    await module.stop()


async def test_warm_session_module_reports_target_degradation_without_failing_readiness() -> None:
    from askme.runtime.module import ModuleRegistry

    from askme.runtime.modules.warm_session_module import WarmSessionModule

    class FailingLLM(_WarmModuleLLMClient):
        async def chat_stream(self, _messages, **kwargs):
            self.calls.append(kwargs)
            raise RuntimeError("provider_down")
            yield "unreachable"

    llm_client = FailingLLM()
    registry = ModuleRegistry()
    registry.register(_WarmModuleLLM(llm_client))
    module = WarmSessionModule()
    module.build(
        {
            "warm_sessions": {
                "enabled": True,
                "shutdown_timeout_seconds": 0.2,
                "llm": {
                    "enabled": True,
                    "refresh_interval_seconds": 60.0,
                    "initial_backoff_seconds": 0.01,
                    "max_backoff_seconds": 0.01,
                    "jitter_ratio": 0.0,
                },
                "tts": {"enabled": False},
            }
        },
        registry,
    )

    await module.start()
    try:
        deadline = asyncio.get_running_loop().time() + 1.0
        while not module.health()["degraded_targets"]:
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("warm-session module did not surface degraded target")
            await asyncio.sleep(0.005)
        health = module.health()

        assert health["status"] == "ok"
        assert health["running"] is True
        assert health["latency_warm"] is False
        assert health["degraded_targets"] == ["llm"]
        assert health["targets"]["llm"]["status"] == "degraded"
    finally:
        await module.stop()


async def test_restart_resets_current_lifecycle_epoch_readiness() -> None:
    target = _RecordingTarget()
    manager = WarmSessionManager(
        [
            WarmSessionBinding(
                name="llm",
                target=target,
                policy=WarmSessionPolicy(
                    startup_delay_seconds=0.02,
                    refresh_interval_seconds=60.0,
                    jitter_ratio=0.0,
                ),
            )
        ]
    )

    await manager.start()
    await asyncio.wait_for(target.called.wait(), timeout=1.0)
    assert manager.snapshot()["targets"]["llm"]["status"] == "warm"
    await manager.stop()

    target.called.clear()
    await manager.start()
    try:
        snapshot = manager.snapshot()["targets"]["llm"]
        assert snapshot["status"] == "scheduled"
        assert snapshot["last_success_age_seconds"] is None
        assert snapshot["provider_session_key"] is None
        assert snapshot["next_attempt_in_seconds"] > 0.0
        await asyncio.wait_for(target.called.wait(), timeout=1.0)
    finally:
        await manager.stop()


class _LifecycleEpochFenceTarget:
    name = "llm"

    def __init__(self) -> None:
        self.calls = 0
        self.first_started = asyncio.Event()
        self.second_started = asyncio.Event()
        self.first_release = asyncio.Event()
        self.second_release = asyncio.Event()
        self.extra_call = asyncio.Event()

    async def ensure_warm(self, *, force_refresh: bool) -> WarmSessionResult:
        self.calls += 1
        call_number = self.calls
        if call_number == 1:
            self.first_started.set()
            release = self.first_release
        elif call_number == 2:
            self.second_started.set()
            release = self.second_release
        else:
            self.extra_call.set()
            return WarmSessionResult(ok=True, status="unexpected")

        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue
        return WarmSessionResult(
            ok=False,
            status="cancelled",
            reason="shutdown",
            neutral=True,
        )

    def cancel(self) -> None:
        return


async def test_detached_old_lifecycle_epoch_cannot_resume_after_manager_restart() -> None:
    target = _LifecycleEpochFenceTarget()
    manager = WarmSessionManager(
        [
            WarmSessionBinding(
                name="llm",
                target=target,
                policy=WarmSessionPolicy(
                    refresh_interval_seconds=0.01,
                    timeout_seconds=60.0,
                    jitter_ratio=0.0,
                    max_attempts_per_hour=100,
                ),
            )
        ],
        shutdown_timeout_seconds=0.01,
    )

    await manager.start()
    await asyncio.wait_for(target.first_started.wait(), timeout=1.0)
    await manager.stop()
    await manager.start()
    await asyncio.wait_for(target.second_started.wait(), timeout=1.0)
    try:
        target.first_release.set()
        await asyncio.sleep(0.05)

        assert target.calls == 2
        assert target.extra_call.is_set() is False
    finally:
        target.second_release.set()
        await manager.stop()


async def test_restart_waits_for_in_progress_stop_to_finish() -> None:
    target = _LifecycleEpochFenceTarget()
    manager = WarmSessionManager(
        [
            WarmSessionBinding(
                name="llm",
                target=target,
                policy=WarmSessionPolicy(
                    refresh_interval_seconds=60.0,
                    timeout_seconds=60.0,
                    jitter_ratio=0.0,
                ),
            )
        ],
        shutdown_timeout_seconds=0.2,
    )

    await manager.start()
    await asyncio.wait_for(target.first_started.wait(), timeout=1.0)
    stop_task = asyncio.create_task(manager.stop())
    await asyncio.sleep(0)
    assert stop_task.done() is False

    restart_task = asyncio.create_task(manager.start())
    await asyncio.sleep(0)
    try:
        assert restart_task.done() is False
        assert target.calls == 1
    finally:
        target.first_release.set()
        await stop_task
        await restart_task
        await asyncio.wait_for(target.second_started.wait(), timeout=1.0)
        target.second_release.set()
        await manager.stop()


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ({"unexpected": 1}, "unknown key"),
        ({"enabled": "maybe"}, "must be a boolean"),
        (
            {"enabled": False, "llm": {"refresh_interva_seconds": 45}},
            "unknown key",
        ),
        (
            {"enabled": True, "shutdown_timeout_seconds": float("nan")},
            "must be a finite number",
        ),
        (
            {"enabled": True, "llm": {"refresh_interval_seconds": 0}},
            "refresh_interval_seconds must be > 0",
        ),
        (
            {"enabled": True, "tts": {"max_attempts_per_hour": "1.5"}},
            "must be an integer",
        ),
    ],
)
def test_warm_session_module_rejects_invalid_product_config(
    section,
    message,
) -> None:
    from askme.runtime.module import ModuleRegistry

    from askme.runtime.modules.warm_session_module import WarmSessionModule

    module = WarmSessionModule()
    with pytest.raises(ValueError, match=message):
        module.build({"warm_sessions": section}, ModuleRegistry())


async def test_snapshot_degrades_when_a_scheduler_exits_unexpectedly(monkeypatch) -> None:
    target = _RecordingTarget()
    manager = WarmSessionManager(
        [
            WarmSessionBinding(
                name="llm",
                target=target,
                policy=WarmSessionPolicy(
                    refresh_interval_seconds=60.0,
                    jitter_ratio=0.0,
                ),
            )
        ]
    )

    def fail_next_delay(*args):
        raise RuntimeError("scheduler bug")

    monkeypatch.setattr(manager, "_next_delay", fail_next_delay)
    await manager.start()
    try:
        await asyncio.wait_for(target.called.wait(), timeout=1.0)
        deadline = asyncio.get_running_loop().time() + 1.0
        snapshot = manager.snapshot()
        while snapshot["status"] != "degraded":
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("dead scheduler was not reflected in health")
            await asyncio.sleep(0.005)
            snapshot = manager.snapshot()

        assert snapshot["running"] is False
        assert snapshot["targets"]["llm"]["status"] == "error"
        assert snapshot["targets"]["llm"]["last_reason"] == "RuntimeError"
    finally:
        await manager.stop()
