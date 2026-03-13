"""Category: Concurrent run() Isolation

Real risks covered:
  - Session state shared between concurrent asyncio coroutines
  - Audio mock answers crossing between runs
  - turn_count bleeding from one run to another
  - OS-thread isolation: two threads share the same orchestrator object

Status BEFORE per-run session isolation: all would FAIL.
Status AFTER: ALL PASS.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

# Shared factory from conftest — single definition, not duplicated per test file
from tests.conftest import make_proactive_orch


# ── Audio helpers ──────────────────────────────────────────────────────────────

class IsolatedAudio:
    def __init__(self, answers=None, label: str = ""):
        self._answers = list(answers or [])
        self._idx = 0
        self.spoken: list[str] = []
        self.label = label

    def drain_buffers(self): ...
    def speak(self, t): self.spoken.append(t)
    def start_playback(self): ...
    def stop_playback(self): ...
    def wait_speaking_done(self): ...

    def listen_loop(self):
        if self._idx < len(self._answers):
            r = self._answers[self._idx]; self._idx += 1; return r
        return None


# ── Two concurrent coroutines ─────────────────────────────────────────────────

class TestTwoConcurrentRuns:

    async def test_two_runs_proceed_independently(self):
        orch = make_proactive_orch()
        r_a, r_b = await asyncio.gather(
            orch.run("web_search", "搜索", IsolatedAudio(["北京天气"])),
            orch.run("web_search", "搜索", IsolatedAudio(["算了"])),
        )
        assert r_a.proceed is True
        assert r_b.proceed is False

    async def test_two_runs_enriched_text_independent(self):
        """Each run's unique answer must appear ONLY in its own enriched_text."""
        orch = make_proactive_orch()
        r_a, r_b = await asyncio.gather(
            orch.run("navigate", "去那里", IsolatedAudio(["仓库A专属答案"])),
            orch.run("navigate", "去那里", IsolatedAudio(["仓库B专属答案"])),
        )
        assert "仓库A专属答案" in r_a.enriched_text
        assert "仓库B专属答案" in r_b.enriched_text
        assert "仓库B专属答案" not in r_a.enriched_text, (
            "Run A must NOT contain B's answer — audio was crossed between runs."
        )
        assert "仓库A专属答案" not in r_b.enriched_text, (
            "Run B must NOT contain A's answer — audio was crossed between runs."
        )

    async def test_audio_mocks_not_crossed(self):
        """Each listen_loop answer must only appear in its own run's enriched_text.

        This verifies content isolation, not just isinstance checks.
        """
        orch = make_proactive_orch()
        r_a, r_b = await asyncio.gather(
            orch.run("web_search", "搜索", IsolatedAudio(["A的独有搜索词"])),
            orch.run("web_search", "搜索", IsolatedAudio(["B的独有搜索词"])),
        )
        assert "A的独有搜索词" in r_a.enriched_text
        assert "B的独有搜索词" in r_b.enriched_text
        assert "B的独有搜索词" not in r_a.enriched_text
        assert "A的独有搜索词" not in r_b.enriched_text

    async def test_interrupted_run_does_not_infect_parallel_run(self):
        orch = make_proactive_orch()

        async def run_b_delayed():
            await asyncio.sleep(0)
            return await orch.run("web_search", "搜索", IsolatedAudio(["明天北京天气"]))

        r_a, r_b = await asyncio.gather(
            orch.run("web_search", "搜索", IsolatedAudio(["算了"])),
            run_b_delayed(),
        )
        assert r_a.proceed is False
        assert r_b.proceed is True

    async def test_estop_in_one_run_does_not_stop_other(self):
        orch = make_proactive_orch()
        r_a, r_b = await asyncio.gather(
            orch.run("navigate", "去那里", IsolatedAudio(["急停"])),
            orch.run("navigate", "去那里", IsolatedAudio(["仓库C"])),
        )
        assert r_a.cancelled_by == "estop"
        assert r_b.proceed is True
        assert "仓库C" in r_b.enriched_text


# ── 20 concurrent coroutines ──────────────────────────────────────────────────

class TestManyConcurrentRuns:

    async def test_20_concurrent_runs_no_state_contamination(self):
        """20 concurrent web_search runs — each with a unique answer."""
        orch = make_proactive_orch()
        n = 20
        answers = [f"搜索词{i:02d}" for i in range(n)]
        results = await asyncio.gather(*[
            orch.run("web_search", "搜索", IsolatedAudio([ans]))
            for ans in answers
        ])
        for i, (result, ans) in enumerate(zip(results, answers)):
            assert result.proceed is True, f"Run {i} failed unexpectedly"
            assert ans in result.enriched_text, (
                f"Run {i}: '{ans}' missing from enriched_text {result.enriched_text!r}"
            )

    async def test_20_mixed_concurrent_runs(self):
        """Even indices succeed; odd indices interrupt — no cross-contamination."""
        orch = make_proactive_orch()
        n = 20
        results = await asyncio.gather(*[
            orch.run(
                "web_search", "搜索",
                IsolatedAudio([f"搜索词{i}"] if i % 2 == 0 else ["算了"])
            )
            for i in range(n)
        ])
        for i, r in enumerate(results):
            if i % 2 == 0:
                assert r.proceed is True, f"Even run {i} should succeed"
                assert f"搜索词{i}" in r.enriched_text
            else:
                assert r.proceed is False, f"Odd run {i} should be interrupted"

    async def test_turn_count_not_shared_across_concurrent_runs(self):
        """Sessions are per-call locals — turn_count must not cross between runs.

        Each run gets [noise, real_answer]. With MAX_TURNS=2:
          - Turn 1: "嗯" (len<2) → retry
          - Turn 2: "真实答案N" → slot filled → proceed

        If turn_count leaked from another session, a run would exhaust MAX_TURNS
        prematurely and proceed without its real answer.
        """
        orch = make_proactive_orch()
        n = 10
        results = await asyncio.gather(*[
            orch.run("web_search", "搜索", IsolatedAudio(["嗯", f"真实答案{i}"]))
            for i in range(n)
        ])
        for i, r in enumerate(results):
            assert r.proceed is True, (
                f"Run {i} must succeed after real answer on turn 2. "
                "Failure suggests turn_count leaked from another session."
            )
            assert f"真实答案{i}" in r.enriched_text, (
                f"Run {i}: real answer '真实答案{i}' missing. Got: {r.enriched_text!r}"
            )

    async def test_cancelled_by_field_independent(self):
        orch = make_proactive_orch()
        r_estop, r_interrupt, r_proceed = await asyncio.gather(
            orch.run("navigate", "去那里", IsolatedAudio(["急停"])),
            orch.run("navigate", "去那里", IsolatedAudio(["算了"])),
            orch.run("navigate", "去那里", IsolatedAudio(["仓库D"])),
        )
        assert r_estop.cancelled_by == "estop"
        assert r_interrupt.cancelled_by == "interrupted"
        assert r_proceed.proceed is True
        assert r_proceed.cancelled_by == ""


# ── Multi-thread isolation ─────────────────────────────────────────────────────

class TestMultiThreadIsolation:
    """Multiple OS threads each call asyncio.run(orch.run(...)) on the same orch.

    asyncio.run() creates a new event loop per thread. Orchestrator has no
    mutable class-level state — sessions are per-run locals. This test detects
    threading.Thread-level data races.
    """

    def test_five_threads_independent_results(self):
        """5 threads, independent event loops, must not contaminate each other.

        Each t.join(timeout=10): if a thread deadlocks, we detect it explicitly
        rather than hanging the entire test suite.
        """
        orch = make_proactive_orch()
        results: dict[int, object] = {}
        errors: list[Exception] = []

        def run_in_thread(i: int) -> None:
            try:
                r = asyncio.run(
                    orch.run("web_search", "搜索", IsolatedAudio([f"线程答案{i}"]))
                )
                results[i] = r
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=run_in_thread, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)  # explicit timeout — detects deadlock

        alive = [t for t in threads if t.is_alive()]
        assert not alive, (
            f"{len(alive)} threads did not finish within 10 s — possible deadlock."
        )
        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 5

        for i in range(5):
            r = results[i]
            assert r.proceed is True, f"Thread {i} must succeed"
            assert f"线程答案{i}" in r.enriched_text, (
                f"Thread {i}: expected '线程答案{i}', got {r.enriched_text!r}."
            )
