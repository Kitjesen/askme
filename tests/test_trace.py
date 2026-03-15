"""Tests for the pipeline tracing module."""

import time

import pytest

from askme.pipeline.trace import PipelineTracer, Span, Trace, get_tracer


class TestSpan:
    def test_duration_ms(self):
        s = Span(name="test", start=1.0, end=1.5)
        assert s.duration_ms == pytest.approx(500.0)

    def test_incomplete_span(self):
        s = Span(name="test", start=1.0)
        assert s.duration_ms == 0.0
        assert s.is_complete is False

    def test_complete_span(self):
        s = Span(name="test", start=1.0, end=2.0)
        assert s.is_complete is True


class TestTrace:
    def test_total_ms(self):
        t = Trace(id="t1", name="test", start=1.0, end=2.5)
        assert t.total_ms == pytest.approx(1500.0)

    def test_incomplete_trace(self):
        t = Trace(id="t1", name="test", start=1.0)
        assert t.total_ms == 0.0

    def test_summary_format(self):
        t = Trace(id="t1", name="test", start=0.0, end=1.5)
        t.spans = [
            Span(name="asr", start=0.0, end=0.3),
            Span(name="llm", start=0.3, end=1.2),
            Span(name="tts", start=1.2, end=1.5),
        ]
        summary = t.summary()
        assert "asr: 300ms" in summary
        assert "llm: 900ms" in summary
        assert "tts: 300ms" in summary
        assert "1500ms total" in summary

    def test_summary_no_spans(self):
        t = Trace(id="t1", name="test", start=0.0, end=1.0)
        assert "no spans" in t.summary()

    def test_to_dict(self):
        t = Trace(id="t1", name="test", start=0.0, end=1.0)
        t.spans = [Span(name="asr", start=0.0, end=0.3)]
        d = t.to_dict()
        assert d["id"] == "t1"
        assert d["total_ms"] == pytest.approx(1000.0)
        assert len(d["spans"]) == 1
        assert d["spans"][0]["name"] == "asr"
        assert "summary" in d


class TestPipelineTracer:
    def test_start_and_finish_trace(self):
        tracer = PipelineTracer()
        trace = tracer.start_trace("test")
        assert trace.name == "test"
        assert trace.id == "t00001"
        result = tracer.finish_trace()
        assert result is trace
        assert result.total_ms > 0

    def test_finish_without_start(self):
        tracer = PipelineTracer()
        assert tracer.finish_trace() is None

    def test_span_context_manager(self):
        tracer = PipelineTracer()
        tracer.start_trace("test")
        with tracer.span("stage1") as s:
            time.sleep(0.01)
            s.metadata["key"] = "value"
        trace = tracer.finish_trace()
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "stage1"
        assert trace.spans[0].duration_ms > 0
        assert trace.spans[0].metadata["key"] == "value"

    def test_record_span(self):
        tracer = PipelineTracer()
        tracer.start_trace("test")
        tracer.record_span("ttft", 850.0, model="minimax")
        trace = tracer.finish_trace()
        assert len(trace.spans) == 1
        assert trace.spans[0].name == "ttft"
        assert trace.spans[0].duration_ms == pytest.approx(850.0, abs=5)

    def test_span_without_trace_is_noop(self):
        tracer = PipelineTracer()
        with tracer.span("orphan"):
            pass
        # Should not raise

    def test_history_limit(self):
        tracer = PipelineTracer(max_history=3)
        for i in range(5):
            tracer.start_trace(f"t{i}")
            tracer.finish_trace()
        history = tracer.get_history(limit=10)
        assert len(history) == 3

    def test_get_history_most_recent_first(self):
        tracer = PipelineTracer()
        tracer.start_trace("first")
        tracer.finish_trace()
        tracer.start_trace("second")
        tracer.finish_trace()
        history = tracer.get_history()
        assert history[0]["name"] == "second"
        assert history[1]["name"] == "first"

    def test_get_summary(self):
        tracer = PipelineTracer()
        for _ in range(3):
            tracer.start_trace("test")
            with tracer.span("asr"):
                time.sleep(0.001)
            with tracer.span("llm"):
                time.sleep(0.002)
            tracer.finish_trace()
        summary = tracer.get_summary()
        assert summary["count"] == 3
        assert "avg_total_ms" in summary
        assert "p50_total_ms" in summary
        assert "p95_total_ms" in summary
        assert "asr" in summary["stage_avg_ms"]
        assert "llm" in summary["stage_avg_ms"]

    def test_get_summary_empty(self):
        tracer = PipelineTracer()
        summary = tracer.get_summary()
        assert summary["count"] == 0

    def test_current_trace(self):
        tracer = PipelineTracer()
        assert tracer.current_trace is None
        tracer.start_trace("test")
        assert tracer.current_trace is not None
        tracer.finish_trace()
        assert tracer.current_trace is None

    def test_auto_finish_on_new_start(self):
        tracer = PipelineTracer()
        tracer.start_trace("first")
        tracer.start_trace("second")
        # First trace should be auto-finished in history
        history = tracer.get_history()
        assert len(history) == 1
        assert history[0]["name"] == "first"

    def test_trace_ids_increment(self):
        tracer = PipelineTracer()
        t1 = tracer.start_trace("a")
        tracer.finish_trace()
        t2 = tracer.start_trace("b")
        tracer.finish_trace()
        assert t1.id == "t00001"
        assert t2.id == "t00002"


class TestGetTracer:
    def test_returns_singleton(self):
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2

    def test_is_pipeline_tracer(self):
        assert isinstance(get_tracer(), PipelineTracer)
