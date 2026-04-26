"""Extended tests for runtime_health utilities and edge cases."""

from __future__ import annotations

from askme.robot.runtime_health import (
    RuntimeHealthSnapshot,
    _clean_optional,
    _normalise_skill_names,
    merge_voice_pipeline_status,
)

# ── _clean_optional ──────────────────────────────────────────────────────────

class TestCleanOptional:
    def test_none_returns_none(self):
        assert _clean_optional(None) is None

    def test_empty_string_returns_none(self):
        assert _clean_optional("") is None

    def test_whitespace_only_returns_none(self):
        assert _clean_optional("   ") is None

    def test_normal_string_returned(self):
        assert _clean_optional("claude-haiku") == "claude-haiku"

    def test_strips_whitespace(self):
        assert _clean_optional("  model  ") == "model"

    def test_int_converted_to_string(self):
        assert _clean_optional(42) == "42"

    def test_zero_converted_to_string(self):
        # 0 is falsy but str("0") is truthy
        assert _clean_optional(0) == "0"


# ── _normalise_skill_names ───────────────────────────────────────────────────

class TestNormaliseSkillNames:
    def test_deduplicates_names(self):
        result = _normalise_skill_names(["patrol", "patrol", "navigate"])
        assert result.count("patrol") == 1

    def test_sorted_output(self):
        result = _normalise_skill_names(["zz_skill", "aa_skill", "mm_skill"])
        assert result == sorted(result)

    def test_strips_whitespace(self):
        result = _normalise_skill_names(["  patrol  "])
        assert "patrol" in result

    def test_empty_strings_removed(self):
        result = _normalise_skill_names(["", "  ", "patrol"])
        assert "" not in result
        assert "patrol" in result

    def test_empty_list_returns_empty(self):
        assert _normalise_skill_names([]) == []


# ── merge_voice_pipeline_status ──────────────────────────────────────────────

class TestMergeVoicePipelineStatus:
    def test_both_none_returns_defaults(self):
        result = merge_voice_pipeline_status(None, None)
        assert "pipeline_ok" in result
        assert "mode" in result

    def test_live_overrides_metrics(self):
        metrics = {"pipeline_ok": False, "mode": "text"}
        live = {"pipeline_ok": True}
        result = merge_voice_pipeline_status(live, metrics)
        assert result["pipeline_ok"] is True

    def test_pipeline_ok_false_when_not_ready(self):
        result = merge_voice_pipeline_status(
            {"mode": "voice", "enabled": True, "asr_available": False},
            {},
        )
        assert result["pipeline_ok"] is False

    def test_pipeline_ok_true_in_text_mode(self):
        """In text mode, pipeline_ok only requires output_ready."""
        result = merge_voice_pipeline_status(
            {"mode": "text", "enabled": False, "output_ready": True},
            {},
        )
        assert result["pipeline_ok"] is True

    def test_tts_backend_none_when_absent(self):
        result = merge_voice_pipeline_status({}, {})
        assert result["tts_backend"] is None

    def test_tts_backend_set_from_live(self):
        result = merge_voice_pipeline_status({"tts_backend": "edge"}, {})
        assert result["tts_backend"] == "edge"

    def test_last_input_chars_defaults_zero(self):
        result = merge_voice_pipeline_status({}, {})
        assert result["last_input_chars"] == 0

    def test_last_input_chars_coerced_to_int(self):
        result = merge_voice_pipeline_status({"last_input_chars": "42"}, {})
        assert result["last_input_chars"] == 42


# ── RuntimeHealthSnapshot ────────────────────────────────────────────────────

def _make_snapshot(**overrides) -> RuntimeHealthSnapshot:
    defaults = dict(
        app_name="test-app",
        app_version="1.0.0",
        brain_config={},
        voice_mode=False,
        robot_mode=False,
        metrics_provider=lambda: {},
        active_skill_names_provider=lambda: [],
        # pipeline_ok=True by default so tests don't get spurious degraded
        voice_status_provider=lambda: {"pipeline_ok": True},
        ota_status_provider=lambda: {},
    )
    defaults.update(overrides)
    return RuntimeHealthSnapshot(**defaults)


class TestHealthSnapshot:
    def test_status_ok_by_default(self):
        s = _make_snapshot()
        assert s.health_snapshot()["status"] == "ok"

    def test_service_name_propagated(self):
        s = _make_snapshot(app_name="my-robot")
        assert s.health_snapshot()["service_name"] == "my-robot"

    def test_service_version_propagated(self):
        s = _make_snapshot(app_version="2.5.1")
        assert s.health_snapshot()["service_version"] == "2.5.1"

    def test_active_skills_empty_list(self):
        s = _make_snapshot()
        assert s.health_snapshot()["active_skills"] == []

    def test_total_conversations_zero_when_absent(self):
        s = _make_snapshot()
        snap = s.health_snapshot()
        assert snap["total_conversations"] == 0

    def test_degraded_when_voice_pipeline_unavailable(self):
        s = _make_snapshot(
            voice_status_provider=lambda: {"pipeline_ok": False, "enabled": True,
                                           "output_ready": False},
        )
        snap = s.health_snapshot()
        assert snap["status"] == "degraded"
        assert "voice_pipeline_unavailable" in snap["degraded_reasons"]

    def test_ota_auth_error_causes_degraded(self):
        s = _make_snapshot(
            ota_status_provider=lambda: {"enabled": True, "state": "auth_error"},
            voice_status_provider=lambda: {"pipeline_ok": True},
        )
        snap = s.health_snapshot()
        assert snap["status"] == "degraded"
        assert "ota_bridge_auth_error" in snap["degraded_reasons"]

    def test_ota_disabled_not_degraded(self):
        s = _make_snapshot(
            ota_status_provider=lambda: {"enabled": False, "state": "stopped"},
            voice_status_provider=lambda: {"pipeline_ok": True},
        )
        snap = s.health_snapshot()
        assert snap["status"] == "ok"

    def test_model_name_falls_back_through_chain(self):
        """When last_model missing, use configured_model; else voice_model; else 'unknown'."""
        s = _make_snapshot(
            brain_config={"model": "configured-model"},
            metrics_provider=lambda: {},  # no llm.last_model
        )
        snap = s.health_snapshot()
        assert snap["model_name"] == "configured-model"

    def test_model_name_unknown_when_nothing_set(self):
        s = _make_snapshot()
        snap = s.health_snapshot()
        assert snap["model_name"] == "unknown"


class TestMetricsSnapshot:
    def test_includes_llm_metrics(self):
        s = _make_snapshot(
            metrics_provider=lambda: {"llm": {"last_latency_ms": 100.0}},
        )
        snap = s.metrics_snapshot()
        assert snap["llm"]["last_latency_ms"] == 100.0

    def test_uptime_seconds_defaults_zero(self):
        s = _make_snapshot()
        snap = s.metrics_snapshot()
        assert snap["uptime_seconds"] == 0.0

    def test_voice_mode_robot_mode_propagated(self):
        s = _make_snapshot(voice_mode=True, robot_mode=True)
        snap = s.metrics_snapshot()
        assert snap["voice_mode"] is True
        assert snap["robot_mode"] is True

    def test_active_skill_count_matches_list(self):
        s = _make_snapshot(
            active_skill_names_provider=lambda: ["a", "b", "c"],
        )
        snap = s.metrics_snapshot()
        assert snap["active_skill_count"] == 3
        assert len(snap["active_skills"]) == 3
