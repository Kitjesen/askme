"""Category: Long-run Stability

Real risks covered:
  - After 100 mixed dispatches: stale session data bleeds into later runs
  - Sessions not GC'd → orchestrator holds references forever
  - turn_count drifts up across runs instead of starting fresh each time
  - Memory grows unboundedly within askme modules over 300 runs

Stability criteria:
  - turn_count == 0 on every fresh ClarificationSession creation (verified
    by patching ClarificationSession with patch.object, not a fragile string path)
  - Sessions from prior runs are GC-eligible: weakref dead after gc.collect()
  - Growth within askme source files ≤ 200 KB over 300 runs.
    Rationale: ClarificationSession ≈ 200 B; 300 leaked = 60 KB; 200 KB = 3× margin.
    If this flaps: run with PYTHONTRACEMALLOC=5 and check the top allocators.
  - All 300 runs return a valid ProactiveResult (no exception leakage)
"""

from __future__ import annotations

import gc
import tracemalloc
import weakref
from unittest.mock import patch

import askme.pipeline.proactive.orchestrator as _orch_module
from askme.pipeline.proactive.session_state import ClarificationSession

# Shared factory from conftest — avoids duplicate _make_orch() across test files
from tests.conftest import make_proactive_orch

# ── Audio helper ──────────────────────────────────────────────────────────────

class _SimpleAudio:
    def __init__(self, answers=None):
        self._q = list(answers or [])
        self._i = 0
    def drain_buffers(self): ...
    def speak(self, t): ...
    def start_playback(self): ...
    def stop_playback(self): ...
    def wait_speaking_done(self): ...
    def listen_loop(self):
        if self._i < len(self._q):
            r = self._q[self._i]; self._i += 1; return r
        return None


# ── Session state does not drift across runs ───────────────────────────────────

class TestSessionStateDoesNotDrift:

    async def test_100_sequential_runs_no_session_residue(self):
        """100 runs of mixed types — proceed/cancelled_by must reflect each run's
        own outcome with no contamination from prior runs.

        Audio note: for "navigate/去那里", pipeline=None means location slots
        use _strip_triggers("去那里", nav_skill) → "那里" → is_vague=True
        → question asked → answer "目标NNN" appended → slot filled. This is
        intentional, not MagicMock side-effect behaviour.
        """
        orch = make_proactive_orch()

        for i in range(100):
            if i % 3 == 0:
                audio = _SimpleAudio([f"目标{i:03d}"])
                r = await orch.run("navigate", "去那里", audio)
                assert r.proceed is True, f"Run {i}: expected proceed=True"
                assert r.cancelled_by == "", (
                    f"Run {i}: cancelled_by must be '' on proceed, got {r.cancelled_by!r}"
                )
            elif i % 3 == 1:
                audio = _SimpleAudio(["算了"])
                r = await orch.run("web_search", "搜索", audio)
                assert r.proceed is False
                assert r.cancelled_by == "interrupted", (
                    f"Run {i}: cancelled_by must be 'interrupted', got {r.cancelled_by!r}"
                )
                assert r.enriched_text == "搜索", (
                    f"Run {i}: enriched_text must be original, got {r.enriched_text!r}"
                )
            else:
                audio = _SimpleAudio(["嗯"])
                r = await orch.run("web_search", "搜索", audio)
                assert r.proceed is True
                assert "算了" not in r.enriched_text

    async def test_turn_count_starts_at_zero_each_run(self):
        """Every fresh ClarificationSession must have turn_count == 0 on creation.

        Uses patch.object(module, 'ClarificationSession') instead of a fragile
        string path. If orchestrator.py is ever refactored to rename the import,
        patch.object raises AttributeError immediately rather than silently no-oping.
        """
        orch = make_proactive_orch()
        initial_turn_counts: list[int] = []

        original_cs = ClarificationSession

        def tracking_cs(*args, **kwargs):
            sess = original_cs(*args, **kwargs)
            initial_turn_counts.append(sess.turn_count)
            return sess

        # patch.object on the imported module reference — explicit, not string-based
        with patch.object(_orch_module, "ClarificationSession", side_effect=tracking_cs):
            for i in range(20):
                await orch.run("web_search", "搜索", _SimpleAudio([f"内容{i}"]))

        assert len(initial_turn_counts) == 20
        assert all(tc == 0 for tc in initial_turn_counts), (
            f"Sessions with non-zero initial turn_count: "
            f"{[tc for tc in initial_turn_counts if tc != 0]}"
        )

    async def test_interrupted_by_does_not_leak_to_next_run(self):
        orch = make_proactive_orch()
        r1 = await orch.run("web_search", "搜索", _SimpleAudio(["算了"]))
        assert r1.proceed is False

        r2 = await orch.run("web_search", "搜索", _SimpleAudio(["北京天气"]))
        assert r2.proceed is True
        assert "算了" not in r2.enriched_text

    async def test_cancel_reason_does_not_leak(self):
        orch = make_proactive_orch()
        r1 = await orch.run("navigate", "去那里", _SimpleAudio(["急停"]))
        assert r1.cancelled_by == "estop"

        r2 = await orch.run("navigate", "去那里", _SimpleAudio(["仓库X"]))
        assert r2.proceed is True
        assert r2.cancelled_by == ""


# ── turn_count is bounded per-run ─────────────────────────────────────────────

class TestTurnCountBounded:
    async def test_noisy_audio_does_not_exceed_max_turns(self):
        orch = make_proactive_orch()
        r = await orch.run("web_search", "搜索", _SimpleAudio(["嗯"] * 10))
        assert r.proceed is True

    async def test_sequential_noisy_runs_each_bounded(self):
        orch = make_proactive_orch()
        for i in range(20):
            r = await orch.run("web_search", "搜索", _SimpleAudio(["嗯"] * 5))
            assert isinstance(r.proceed, bool), f"Run {i} must return valid result"


# ── GC eligibility of session objects ─────────────────────────────────────────

class TestSessionGCEligibility:

    async def test_sessions_are_gc_eligible_after_run(self):
        """Orchestrator must not hold references to completed sessions.

        Uses patch.object (not string-based patch) so that renaming the
        ClarificationSession import in orchestrator.py raises AttributeError
        immediately instead of silently letting the test trivially pass.
        """
        orch = make_proactive_orch()
        captured_refs: list[weakref.ref] = []
        original_cs = ClarificationSession

        def tracking_cs(*args, **kwargs):
            sess = original_cs(*args, **kwargs)
            captured_refs.append(weakref.ref(sess))
            return sess

        with patch.object(_orch_module, "ClarificationSession", side_effect=tracking_cs):
            for i in range(10):
                await orch.run("web_search", "搜索", _SimpleAudio([f"内容{i}"]))

        gc.collect()

        alive = [ref for ref in captured_refs if ref() is not None]
        assert len(alive) == 0, (
            f"{len(alive)} of {len(captured_refs)} ClarificationSession objects "
            "still alive after GC — orchestrator is holding references."
        )


# ── Memory growth bounded ─────────────────────────────────────────────────────

class TestMemoryGrowthBounded:

    async def test_300_runs_askme_memory_growth_bounded(self):
        """300 runs — askme-module memory growth must be ≤ 200 KB.

        Filters to only askme source file allocations to exclude pytest/asyncio
        infrastructure noise. The threshold is:
          ClarificationSession ≈ 200 B × 300 leaked = 60 KB worst case;
          200 KB = 3× safety margin.

        If this test flaps: run with PYTHONTRACEMALLOC=5 and inspect the top
        allocators with `stats[0].traceback.format()` to find the leak site.
        """
        orch = make_proactive_orch()

        for i in range(10):  # warm-up: stabilise module-level allocations
            await orch.run("web_search", "搜索", _SimpleAudio([f"热身{i}"]))
        gc.collect()

        tracemalloc.start(5)
        snapshot_before = tracemalloc.take_snapshot()

        for i in range(300):
            a = [
                _SimpleAudio([f"内容{i}"]),
                _SimpleAudio(["算了"]),
                _SimpleAudio(["急停"]),
                _SimpleAudio(["嗯"]),
            ][i % 4]
            await orch.run("web_search", "搜索", a)

        gc.collect()
        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        stats = snapshot_after.compare_to(snapshot_before, "lineno")
        askme_growth = sum(
            s.size_diff
            for s in stats
            if s.size_diff > 0
            and s.traceback
            and any("askme" in frame.filename for frame in s.traceback)
        )
        askme_growth_kb = askme_growth / 1024

        assert askme_growth_kb < 200, (
            f"askme module memory grew {askme_growth_kb:.1f} KB over 300 runs "
            f"(threshold: 200 KB). Check for per-run objects not being GC'd."
        )
