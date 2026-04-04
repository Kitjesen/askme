"""Tests for OTABridgeMetrics and _compute_percentiles."""

from __future__ import annotations

import threading
import time

import pytest

from askme.robot.ota_bridge import OTABridgeMetrics, _compute_percentiles


# ── _compute_percentiles ─────────────────────────────────────────────────────

class TestComputePercentiles:
    def test_empty_returns_none_for_all(self):
        result = _compute_percentiles([])
        assert result == {"p50": None, "p95": None, "p99": None}

    def test_single_value_all_same(self):
        result = _compute_percentiles([100.0])
        assert result["p50"] == 100.0
        assert result["p95"] == 100.0
        assert result["p99"] == 100.0

    def test_two_values_p50_is_midpoint(self):
        result = _compute_percentiles([0.0, 100.0])
        assert result["p50"] == 50.0

    def test_sorted_correctly(self):
        result = _compute_percentiles([300.0, 100.0, 200.0])
        assert result["p50"] == 200.0

    def test_p95_higher_than_p50(self):
        values = list(range(1, 101))  # 1..100
        result = _compute_percentiles(values)
        assert result["p95"] > result["p50"]

    def test_p99_highest(self):
        values = list(range(1, 101))
        result = _compute_percentiles(values)
        assert result["p99"] >= result["p95"] >= result["p50"]


# ── OTABridgeMetrics ─────────────────────────────────────────────────────────

class TestOTABridgeMetricsInit:
    def test_counts_start_at_zero(self):
        m = OTABridgeMetrics()
        snap = m.snapshot()
        assert snap["llm"]["call_count"] == 0
        assert snap["skills"]["run_count"] == 0
        assert snap["conversation_count"] == 0

    def test_uptime_positive(self):
        m = OTABridgeMetrics()
        snap = m.snapshot()
        assert snap["uptime_seconds"] >= 0.0


class TestRecordConversationTurn:
    def test_increments_count(self):
        m = OTABridgeMetrics()
        m.record_conversation_turn()
        m.record_conversation_turn()
        assert m.snapshot()["conversation_count"] == 2


class TestRecordLlmCall:
    def test_call_count_increments(self):
        m = OTABridgeMetrics()
        m.record_llm_call(0.5, success=True)
        assert m.snapshot()["llm"]["call_count"] == 1

    def test_success_count_increments_on_success(self):
        m = OTABridgeMetrics()
        m.record_llm_call(0.5, success=True)
        assert m.snapshot()["llm"]["success_count"] == 1

    def test_failure_count_increments_on_failure(self):
        m = OTABridgeMetrics()
        m.record_llm_call(0.5, success=False)
        assert m.snapshot()["llm"]["failure_count"] == 1

    def test_last_latency_ms_set(self):
        m = OTABridgeMetrics()
        m.record_llm_call(1.0, success=True)
        assert m.snapshot()["llm"]["last_latency_ms"] == 1000.0

    def test_last_mode_and_model_stored(self):
        m = OTABridgeMetrics()
        m.record_llm_call(0.2, success=True, mode="stream", model="claude-sonnet")
        snap = m.snapshot()["llm"]
        assert snap["last_mode"] == "stream"
        assert snap["last_model"] == "claude-sonnet"

    def test_average_latency_calculated(self):
        m = OTABridgeMetrics()
        m.record_llm_call(1.0, success=True)   # 1000ms
        m.record_llm_call(0.5, success=True)   # 500ms
        avg = m.snapshot()["llm"]["average_latency_ms"]
        assert avg == 750.0

    def test_negative_duration_treated_as_zero(self):
        m = OTABridgeMetrics()
        m.record_llm_call(-1.0, success=True)
        assert m.snapshot()["llm"]["last_latency_ms"] == 0.0

    def test_percentiles_populated(self):
        m = OTABridgeMetrics()
        for i in range(10):
            m.record_llm_call(float(i) / 10, success=True)
        snap = m.snapshot()["llm"]
        assert snap["p50_latency_ms"] is not None
        assert snap["p95_latency_ms"] is not None


class TestRecordSkillExecution:
    def test_run_count_increments(self):
        m = OTABridgeMetrics()
        m.record_skill_execution(success=True, skill_name="navigate")
        assert m.snapshot()["skills"]["run_count"] == 1

    def test_success_count_increments(self):
        m = OTABridgeMetrics()
        m.record_skill_execution(success=True, skill_name="patrol")
        assert m.snapshot()["skills"]["success_count"] == 1

    def test_failure_count_increments(self):
        m = OTABridgeMetrics()
        m.record_skill_execution(success=False, skill_name="patrol")
        assert m.snapshot()["skills"]["failure_count"] == 1

    def test_per_skill_stats_tracked(self):
        m = OTABridgeMetrics()
        m.record_skill_execution(success=True, skill_name="navigate", duration_s=0.5)
        m.record_skill_execution(success=False, skill_name="navigate", duration_s=0.3)
        per = m.snapshot()["skills"]["per_skill"]["navigate"]
        assert per["calls"] == 2
        assert per["success"] == 1
        assert per["failure"] == 1
        assert per["avg_ms"] is not None

    def test_success_rate_calculated(self):
        m = OTABridgeMetrics()
        m.record_skill_execution(success=True, skill_name="a")
        m.record_skill_execution(success=True, skill_name="a")
        m.record_skill_execution(success=False, skill_name="a")
        rate = m.snapshot()["skills"]["success_rate"]
        assert abs(rate - 2/3) < 0.01

    def test_no_skill_name_still_counts(self):
        m = OTABridgeMetrics()
        m.record_skill_execution(success=True)
        assert m.snapshot()["skills"]["run_count"] == 1


class TestVoiceState:
    def test_default_voice_state(self):
        m = OTABridgeMetrics()
        vp = m.snapshot()["voice_pipeline"]
        assert vp["mode"] == "text"
        assert vp["enabled"] is False

    def test_update_voice_state(self):
        m = OTABridgeMetrics()
        m.update_voice_state(mode="voice", enabled=True)
        vp = m.snapshot()["voice_pipeline"]
        assert vp["mode"] == "voice"
        assert vp["enabled"] is True

    def test_mark_voice_input(self):
        m = OTABridgeMetrics()
        m.mark_voice_input("hello robot")
        vp = m.snapshot()["voice_pipeline"]
        assert vp["last_input_chars"] == 11
        assert vp["last_input_at"] is not None
        assert vp["last_error"] is None

    def test_mark_voice_error(self):
        m = OTABridgeMetrics()
        m.mark_voice_error("ASR crashed")
        vp = m.snapshot()["voice_pipeline"]
        assert vp["last_error"] == "ASR crashed"
        assert vp["last_error_at"] is not None

    def test_mark_voice_error_clears_on_next_input(self):
        m = OTABridgeMetrics()
        m.mark_voice_error("boom")
        m.mark_voice_input("clear")
        vp = m.snapshot()["voice_pipeline"]
        assert vp["last_error"] is None


class TestReset:
    def test_reset_clears_counts(self):
        m = OTABridgeMetrics()
        m.record_conversation_turn()
        m.record_llm_call(0.5, success=True)
        m.reset()
        snap = m.snapshot()
        assert snap["conversation_count"] == 0
        assert snap["llm"]["call_count"] == 0

    def test_reset_clears_skill_stats(self):
        m = OTABridgeMetrics()
        m.record_skill_execution(success=True, skill_name="navigate")
        m.reset()
        assert m.snapshot()["skills"]["per_skill"] == {}


class TestThreadSafety:
    def test_concurrent_llm_calls(self):
        m = OTABridgeMetrics()
        errors = []

        def worker():
            try:
                for _ in range(50):
                    m.record_llm_call(0.1, success=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert m.snapshot()["llm"]["call_count"] == 200
