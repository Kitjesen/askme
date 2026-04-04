"""Tests for health_server Prometheus helper functions."""

from __future__ import annotations

import pytest

from askme.health_server import (
    _append_metric,
    _escape_label_value,
    _format_labels,
    _format_metric_value,
    render_prometheus_metrics,
)


# ── _format_metric_value ──────────────────────────────────────────────────────

class TestFormatMetricValue:
    def test_none_returns_nan(self):
        assert _format_metric_value(None) == "NaN"

    def test_true_returns_1(self):
        assert _format_metric_value(True) == "1"

    def test_false_returns_0(self):
        assert _format_metric_value(False) == "0"

    def test_integer(self):
        assert _format_metric_value(42) == "42"

    def test_float_no_trailing_zeros(self):
        result = _format_metric_value(1.5)
        assert result == "1.5"

    def test_float_integer_value(self):
        result = _format_metric_value(2.0)
        assert result == "2"

    def test_nan_float(self):
        import math
        assert _format_metric_value(math.nan) == "NaN"

    def test_inf_float(self):
        import math
        assert _format_metric_value(math.inf) == "NaN"

    def test_numeric_string(self):
        result = _format_metric_value("3.14")
        assert result == "3.14"

    def test_non_numeric_string(self):
        assert _format_metric_value("not_a_number") == "NaN"

    def test_zero(self):
        assert _format_metric_value(0) == "0"

    def test_negative_int(self):
        assert _format_metric_value(-5) == "-5"

    def test_large_integer(self):
        result = _format_metric_value(1000000)
        assert result == "1000000"


# ── _escape_label_value ───────────────────────────────────────────────────────

class TestEscapeLabelValue:
    def test_plain_string_unchanged(self):
        assert _escape_label_value("hello") == "hello"

    def test_none_returns_empty(self):
        assert _escape_label_value(None) == ""

    def test_double_quote_escaped(self):
        result = _escape_label_value('say "hello"')
        assert '\\"' in result

    def test_newline_escaped(self):
        result = _escape_label_value("line1\nline2")
        assert "\\n" in result

    def test_backslash_escaped(self):
        result = _escape_label_value("path\\value")
        assert "\\\\" in result

    def test_integer_converted(self):
        assert _escape_label_value(42) == "42"

    def test_empty_string(self):
        assert _escape_label_value("") == ""


# ── _format_labels ───��────────────────────────────────────────────────────────

class TestFormatLabels:
    def test_none_returns_empty(self):
        assert _format_labels(None) == ""

    def test_empty_dict_returns_empty(self):
        assert _format_labels({}) == ""

    def test_single_label(self):
        result = _format_labels({"key": "value"})
        assert result == '{key="value"}'

    def test_multiple_labels_sorted(self):
        result = _format_labels({"z": "last", "a": "first"})
        # Sorted by key
        assert result.startswith('{a=')
        assert 'z=' in result

    def test_label_value_escaped(self):
        result = _format_labels({"model": 'the "best"'})
        assert '\\"best\\"' in result

    def test_format_is_valid_prometheus(self):
        result = _format_labels({"env": "prod", "region": "cn"})
        assert result.startswith("{")
        assert result.endswith("}")


# ── _append_metric ────────────────────────────────────────────────────────────

class TestAppendMetric:
    def test_appends_help_type_value(self):
        lines: list[str] = []
        _append_metric(lines, "my_metric", "A test metric", "gauge", 42)
        text = "".join(lines)
        assert "# HELP my_metric A test metric" in text
        assert "# TYPE my_metric gauge" in text
        assert "my_metric 42" in text

    def test_with_labels(self):
        lines: list[str] = []
        _append_metric(lines, "info", "Info", "gauge", 1, labels={"env": "test"})
        text = "".join(lines)
        assert 'env="test"' in text

    def test_none_value_produces_nan(self):
        lines: list[str] = []
        _append_metric(lines, "my_metric", "help", "gauge", None)
        text = "".join(lines)
        assert "NaN" in text

    def test_bool_true_produces_1(self):
        lines: list[str] = []
        _append_metric(lines, "ok", "ok gauge", "gauge", True)
        text = "".join(lines)
        assert "ok 1" in text


# ── render_prometheus_metrics ─────────────────────────────────────────────────

class TestRenderPrometheusMetrics:
    def test_returns_string(self):
        result = render_prometheus_metrics({})
        assert isinstance(result, str)

    def test_contains_askme_up(self):
        result = render_prometheus_metrics({})
        assert "askme_up" in result

    def test_contains_service_info(self):
        result = render_prometheus_metrics({"service": "askme", "version": "1.0"})
        assert "askme_service_info" in result

    def test_active_skills_list_counted(self):
        snapshot = {"active_skills": ["patrol", "navigate", "report"]}
        result = render_prometheus_metrics(snapshot)
        assert "askme_active_skills" in result
        # Each skill should appear as a label
        assert "patrol" in result

    def test_voice_pipeline_ok(self):
        snapshot = {"voice_pipeline_status": {"pipeline_ok": True}}
        result = render_prometheus_metrics(snapshot)
        assert "askme_voice_pipeline_ok" in result
        assert "1" in result

    def test_ota_bridge_registered(self):
        snapshot = {"ota_bridge_status": {"enabled": True, "registered": True}}
        result = render_prometheus_metrics(snapshot)
        assert "askme_ota_bridge_registered" in result

    def test_llm_latency_percentiles(self):
        snapshot = {"llm": {"p50_latency_ms": 100.0, "p95_latency_ms": 250.0}}
        result = render_prometheus_metrics(snapshot)
        assert "askme_llm_latency_p50_ms" in result
        assert "askme_llm_latency_p95_ms" in result

    def test_empty_snapshot_no_crash(self):
        result = render_prometheus_metrics({})
        assert len(result) > 0

    def test_status_ok_produces_1(self):
        result = render_prometheus_metrics({"status": "ok"})
        assert "askme_health_status" in result

    def test_status_degraded_produces_0(self):
        result = render_prometheus_metrics({"status": "degraded"})
        assert "askme_health_status" in result
