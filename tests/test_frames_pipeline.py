"""Extended tests for FramePipeline, FrameProcessor, and CancellationToken."""

from __future__ import annotations

import pytest

from askme.pipeline.frames import (
    CancellationToken,
    Frame,
    FramePipeline,
    FrameProcessor,
    InterruptFrame,
    MetricsFrame,
    SystemFrame,
    TranscriptionFrame,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

class CountingProcessor(FrameProcessor):
    def __init__(self):
        super().__init__(name="counter")
        self.frames_received = []

    async def process_frame(self, frame: Frame) -> list[Frame]:
        self.frames_received.append(frame)
        return [frame]


class DoubleProcessor(FrameProcessor):
    def __init__(self):
        super().__init__(name="doubler")

    async def process_frame(self, frame: Frame) -> list[Frame]:
        return [frame, frame]


class DropProcessor(FrameProcessor):
    def __init__(self):
        super().__init__(name="dropper")

    async def process_frame(self, frame: Frame) -> list[Frame]:
        return []


# ── FramePipeline.push_frame ──────────────────────────────────────────────────

class TestPushFrame:
    @pytest.mark.asyncio
    async def test_data_frame_flows_through(self):
        counter = CountingProcessor()
        pipeline = FramePipeline([counter])
        frame = TranscriptionFrame(text="hello")
        result = await pipeline.push_frame(frame)
        assert len(result) == 1
        assert len(counter.frames_received) == 1

    @pytest.mark.asyncio
    async def test_double_processor_emits_twice(self):
        doubler = DoubleProcessor()
        pipeline = FramePipeline([doubler])
        frame = TranscriptionFrame(text="hi")
        result = await pipeline.push_frame(frame)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_drop_processor_returns_empty(self):
        dropper = DropProcessor()
        pipeline = FramePipeline([dropper])
        frame = TranscriptionFrame(text="hi")
        result = await pipeline.push_frame(frame)
        assert result == []

    @pytest.mark.asyncio
    async def test_system_frame_returned_as_is(self):
        pipeline = FramePipeline()
        frame = SystemFrame()
        result = await pipeline.push_frame(frame)
        assert len(result) == 1
        assert result[0] is frame

    @pytest.mark.asyncio
    async def test_interrupt_frame_returns_single_item(self):
        pipeline = FramePipeline()
        interrupt = InterruptFrame()
        result = await pipeline.push_frame(interrupt)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_metrics_frame_collected(self):
        pipeline = FramePipeline()
        metrics = MetricsFrame(stage="test", duration_ms=5.0)
        await pipeline.push_frame(metrics)
        assert len(pipeline.get_metrics()) == 1

    @pytest.mark.asyncio
    async def test_metrics_capped_after_many_pushes(self):
        pipeline = FramePipeline()
        for i in range(250):
            await pipeline.push_frame(MetricsFrame(stage=f"s{i}", duration_ms=1.0))
        # Should be capped (200 initially, pruned to 100)
        assert len(pipeline.get_metrics()) <= 200

    @pytest.mark.asyncio
    async def test_pipeline_skips_interrupted_processor(self):
        counter = CountingProcessor()
        counter._interrupted = True
        pipeline = FramePipeline([counter])
        frame = TranscriptionFrame(text="hi")
        result = await pipeline.push_frame(frame)
        assert result == []

    @pytest.mark.asyncio
    async def test_chained_processors(self):
        counter1 = CountingProcessor()
        counter2 = CountingProcessor()
        pipeline = FramePipeline([counter1, counter2])
        frame = TranscriptionFrame(text="data")
        result = await pipeline.push_frame(frame)
        assert len(result) == 1
        assert len(counter1.frames_received) == 1
        assert len(counter2.frames_received) == 1


# ── FramePipeline.reset ───────────────────────────────────────────────────────

class TestReset:
    @pytest.mark.asyncio
    async def test_reset_clears_interrupt_state(self):
        counter = CountingProcessor()
        pipeline = FramePipeline([counter])
        counter._interrupted = True
        pipeline.reset()
        assert not counter._interrupted

    def test_reset_clears_metrics(self):
        pipeline = FramePipeline()
        pipeline._metrics.append(MetricsFrame(stage="x", duration_ms=1.0))
        pipeline.reset()
        assert pipeline.get_metrics() == []


# ── FramePipeline.add ─────────────────────────────────────────────────────────

class TestAdd:
    def test_add_returns_pipeline(self):
        pipeline = FramePipeline()
        result = pipeline.add(CountingProcessor())
        assert result is pipeline

    def test_add_appends_processor(self):
        pipeline = FramePipeline()
        proc = CountingProcessor()
        pipeline.add(proc)
        assert proc in pipeline.processors


# ── CancellationToken ─────────────────────────────────────────────────────────

class TestCancellationToken:
    def test_not_cancelled_initially(self):
        token = CancellationToken()
        assert token.is_cancelled is False

    def test_cancel_sets_flag(self):
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled is True

    def test_reset_clears_flag(self):
        token = CancellationToken()
        token.cancel()
        token.reset()
        assert token.is_cancelled is False

    @pytest.mark.asyncio
    async def test_wait_returns_true_when_cancelled(self):
        token = CancellationToken()
        token.cancel()
        result = await token.wait(timeout=0.1)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_returns_false_on_timeout(self):
        token = CancellationToken()
        result = await token.wait(timeout=0.01)
        assert result is False

    def test_reset_allows_reuse(self):
        token = CancellationToken()
        token.cancel()
        token.reset()
        assert not token.is_cancelled
