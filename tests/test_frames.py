"""Tests for the Frame-based pipeline abstraction."""

import asyncio

import numpy as np
import pytest

from askme.pipeline.frames import (
    AudioRawFrame,
    CancellationToken,
    DataFrame,
    Frame,
    FramePipeline,
    FrameProcessor,
    IntentFrame,
    InterruptFrame,
    LLMFullResponseFrame,
    LLMStartFrame,
    LLMTextFrame,
    MetricsFrame,
    PassthroughProcessor,
    StartInterruptFrame,
    StopInterruptFrame,
    SystemFrame,
    TranscriptionFrame,
    TTSAudioFrame,
    TTSSpeakFrame,
    VADFrame,
)

# ---------------------------------------------------------------------------
# Frame type hierarchy
# ---------------------------------------------------------------------------

class TestFrameTypes:
    def test_frame_has_id_and_timestamp(self):
        f = Frame()
        assert len(f.id) == 12
        assert f.timestamp > 0

    def test_system_frame_is_frame(self):
        f = SystemFrame()
        assert isinstance(f, Frame)

    def test_data_frame_is_frame(self):
        f = DataFrame()
        assert isinstance(f, Frame)

    def test_interrupt_frame_is_system(self):
        f = InterruptFrame(reason="barge_in")
        assert isinstance(f, SystemFrame)
        assert f.reason == "barge_in"

    def test_audio_raw_frame(self):
        audio = np.zeros(1600, dtype=np.float32)
        f = AudioRawFrame(audio=audio, sample_rate=16000, peak=500)
        assert isinstance(f, DataFrame)
        assert f.sample_rate == 16000
        assert f.peak == 500
        assert len(f.audio) == 1600

    def test_vad_frame(self):
        f = VADFrame(is_speech=True, peak=1200)
        assert f.is_speech is True

    def test_transcription_frame(self):
        f = TranscriptionFrame(text="你好", is_final=True, language="zh")
        assert f.text == "你好"
        assert f.is_final is True

    def test_intent_frame(self):
        f = IntentFrame(intent_type="voice_trigger", skill_name="navigate", user_text="去仓库")
        assert f.skill_name == "navigate"

    def test_llm_frames(self):
        start = LLMStartFrame(model="minimax", ttft_ms=800.0)
        assert start.ttft_ms == 800.0
        chunk = LLMTextFrame(text="你好")
        assert chunk.text == "你好"
        full = LLMFullResponseFrame(text="完整回复", tool_calls=[])
        assert full.text == "完整回复"

    def test_tts_frames(self):
        speak = TTSSpeakFrame(text="说这句话")
        assert speak.text == "说这句话"
        audio = TTSAudioFrame(sample_rate=24000)
        assert audio.sample_rate == 24000

    def test_metrics_frame(self):
        f = MetricsFrame(stage="asr", duration_ms=230.0, metadata={"model": "zipformer"})
        assert isinstance(f, SystemFrame)
        assert f.stage == "asr"

    def test_start_stop_interrupt(self):
        s = StartInterruptFrame()
        assert isinstance(s, SystemFrame)
        e = StopInterruptFrame()
        assert isinstance(e, SystemFrame)


# ---------------------------------------------------------------------------
# Processor and Pipeline
# ---------------------------------------------------------------------------

class DoubleTextProcessor(FrameProcessor):
    """Test processor that doubles TranscriptionFrame text."""

    async def process_frame(self, frame: Frame) -> list[Frame]:
        if isinstance(frame, TranscriptionFrame):
            return [TranscriptionFrame(text=frame.text * 2)]
        return [frame]


class FilterEmptyProcessor(FrameProcessor):
    """Test processor that drops empty TranscriptionFrames."""

    async def process_frame(self, frame: Frame) -> list[Frame]:
        if isinstance(frame, TranscriptionFrame) and not frame.text:
            return []
        return [frame]


class CountingProcessor(FrameProcessor):
    """Counts how many frames it processes and interrupts it receives."""

    def __init__(self):
        super().__init__("counter")
        self.processed = 0
        self.interrupts = 0

    async def process_frame(self, frame: Frame) -> list[Frame]:
        self.processed += 1
        return [frame]

    async def handle_interrupt(self, frame: InterruptFrame) -> None:
        await super().handle_interrupt(frame)
        self.interrupts += 1


class TestPassthroughProcessor:
    @pytest.mark.asyncio
    async def test_passes_frame_through(self):
        proc = PassthroughProcessor()
        f = TranscriptionFrame(text="hello")
        result = await proc.process_frame(f)
        assert len(result) == 1
        assert result[0] is f


class TestFramePipeline:
    @pytest.mark.asyncio
    async def test_empty_pipeline(self):
        pipe = FramePipeline()
        f = TranscriptionFrame(text="hello")
        result = await pipe.push_frame(f)
        assert len(result) == 1
        assert result[0] is f

    @pytest.mark.asyncio
    async def test_single_processor(self):
        pipe = FramePipeline([DoubleTextProcessor()])
        f = TranscriptionFrame(text="AB")
        result = await pipe.push_frame(f)
        assert len(result) == 1
        assert result[0].text == "ABAB"

    @pytest.mark.asyncio
    async def test_chained_processors(self):
        pipe = FramePipeline([
            DoubleTextProcessor(),
            DoubleTextProcessor(),
        ])
        f = TranscriptionFrame(text="X")
        result = await pipe.push_frame(f)
        assert result[0].text == "XXXX"

    @pytest.mark.asyncio
    async def test_filter_drops_frame(self):
        pipe = FramePipeline([FilterEmptyProcessor()])
        result = await pipe.push_frame(TranscriptionFrame(text=""))
        assert result == []

    @pytest.mark.asyncio
    async def test_interrupt_propagates_to_all(self):
        c1 = CountingProcessor()
        c2 = CountingProcessor()
        pipe = FramePipeline([c1, c2])
        interrupt = InterruptFrame(reason="barge_in")
        result = await pipe.push_frame(interrupt)
        assert c1.interrupts == 1
        assert c2.interrupts == 1
        assert len(result) == 1
        assert result[0] is interrupt

    @pytest.mark.asyncio
    async def test_interrupted_processor_skipped(self):
        c = CountingProcessor()
        pipe = FramePipeline([c])
        await pipe.push_frame(InterruptFrame(reason="test"))
        assert c._interrupted is True
        result = await pipe.push_frame(TranscriptionFrame(text="hello"))
        assert result == []
        assert c.processed == 0

    @pytest.mark.asyncio
    async def test_reset_clears_interrupt(self):
        c = CountingProcessor()
        pipe = FramePipeline([c])
        await pipe.push_frame(InterruptFrame(reason="test"))
        pipe.reset()
        assert c._interrupted is False
        result = await pipe.push_frame(TranscriptionFrame(text="hello"))
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_metrics_frame_collected(self):
        pipe = FramePipeline()
        mf = MetricsFrame(stage="test", duration_ms=42.0)
        await pipe.push_frame(mf)
        metrics = pipe.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].stage == "test"

    @pytest.mark.asyncio
    async def test_add_returns_self(self):
        pipe = FramePipeline()
        result = pipe.add(PassthroughProcessor())
        assert result is pipe
        assert len(pipe.processors) == 1


# ---------------------------------------------------------------------------
# CancellationToken
# ---------------------------------------------------------------------------

class TestCancellationToken:
    def test_initial_state(self):
        token = CancellationToken()
        assert token.is_cancelled is False

    def test_cancel(self):
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled is True

    def test_reset(self):
        token = CancellationToken()
        token.cancel()
        token.reset()
        assert token.is_cancelled is False

    @pytest.mark.asyncio
    async def test_wait_returns_true_when_cancelled(self):
        token = CancellationToken()

        async def _cancel_later():
            await asyncio.sleep(0.01)
            token.cancel()

        asyncio.create_task(_cancel_later())
        result = await token.wait(timeout=1.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_wait_returns_false_on_timeout(self):
        token = CancellationToken()
        result = await token.wait(timeout=0.01)
        assert result is False
