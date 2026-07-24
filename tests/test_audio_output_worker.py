"""Contract tests for the resident audio output worker."""

from __future__ import annotations

import threading
import time

import numpy as np


class _RecordingAdapter:
    def __init__(self) -> None:
        self.open_count = 0
        self.close_count = 0
        self.drop_count = 0
        self.writes: list[bytes] = []
        self._lock = threading.Lock()
        self.block_writes = False
        self.first_write = threading.Event()
        self.release_write = threading.Event()

    def open(self) -> None:
        with self._lock:
            self.open_count += 1

    def write(self, pcm: bytes) -> None:
        with self._lock:
            self.writes.append(bytes(pcm))
        self.first_write.set()
        if self.block_writes:
            self.release_write.wait(timeout=1.0)

    def drop(self) -> None:
        with self._lock:
            self.drop_count += 1
        self.release_write.set()

    def close(self) -> None:
        with self._lock:
            self.close_count += 1


def test_two_utterances_share_one_native_output_stream() -> None:
    from askme.voice.output.audio_output_worker import (
        AudioOutputConfig,
        AudioOutputWorker,
    )

    adapter = _RecordingAdapter()
    worker = AudioOutputWorker(
        adapter,
        config=AudioOutputConfig(
            native_sample_rate=48_000,
            channels=2,
            period_ms=10,
            cold_preroll_ms=0,
            warm_leadin_ms=0,
        ),
    )
    try:
        first = worker.submit(
            np.array([0.25, -0.25], dtype=np.float32),
            sample_rate=24_000,
            generation=1,
        )
        second = worker.submit(
            np.array([0.5, -0.5], dtype=np.float32),
            sample_rate=48_000,
            generation=2,
        )

        assert first.wait(timeout=1.0)
        assert second.wait(timeout=1.0)
    finally:
        worker.shutdown()

    assert adapter.open_count == 1
    assert adapter.close_count == 1
    rendered = np.frombuffer(b"".join(adapter.writes), dtype=np.int16).reshape(-1, 2)
    non_silent = rendered[np.any(rendered != 0, axis=1)]
    assert len(non_silent) >= 6
    assert np.array_equal(non_silent[:, 0], non_silent[:, 1])


def test_cancel_drops_current_generation_and_next_utterance_recovers() -> None:
    from askme.voice.output.audio_output_worker import (
        AudioOutputConfig,
        AudioOutputWorker,
    )

    adapter = _RecordingAdapter()
    adapter.block_writes = True
    worker = AudioOutputWorker(
        adapter,
        config=AudioOutputConfig(
            native_sample_rate=48_000,
            channels=2,
            period_ms=10,
            cold_preroll_ms=0,
            warm_leadin_ms=0,
        ),
    )
    try:
        interrupted = worker.submit(
            np.ones(4_800, dtype=np.float32) * 0.2,
            sample_rate=48_000,
            generation=7,
        )
        assert adapter.first_write.wait(timeout=1.0)

        worker.cancel(generation=7)

        assert interrupted.wait(timeout=1.0)
        assert interrupted.cancelled is True

        adapter.block_writes = False
        recovered = worker.submit(
            np.array([0.1, -0.1], dtype=np.float32),
            sample_rate=48_000,
            generation=8,
        )
        assert recovered.wait(timeout=1.0)
        assert recovered.cancelled is False
        assert recovered.error is None
    finally:
        worker.shutdown()

    assert adapter.drop_count == 1
    assert adapter.open_count == 2


def test_drop_failure_marks_active_ticket_cancelled_with_error() -> None:
    from askme.voice.output.audio_output_worker import (
        AudioOutputConfig,
        AudioOutputWorker,
    )

    class _DropFailingAdapter(_RecordingAdapter):
        def drop(self) -> None:
            super().drop()
            raise RuntimeError("drop failed")

    adapter = _DropFailingAdapter()
    adapter.block_writes = True
    worker = AudioOutputWorker(
        adapter,
        config=AudioOutputConfig(
            native_sample_rate=48_000,
            channels=2,
            period_ms=10,
            cold_preroll_ms=0,
            warm_leadin_ms=0,
        ),
    )
    try:
        interrupted = worker.submit(
            np.ones(4_800, dtype=np.float32) * 0.2,
            sample_rate=48_000,
            generation=17,
        )
        assert adapter.first_write.wait(timeout=1.0)

        assert worker.cancel(generation=17) is True
        assert interrupted.wait(timeout=1.0)
        assert interrupted.cancelled is True
        assert isinstance(interrupted.error, RuntimeError)
        assert str(interrupted.error) == "drop failed"
        assert worker.status_snapshot()["last_error"] == (
            "RuntimeError: drop failed"
        )

        deadline = time.monotonic() + 1.0
        while worker.status_snapshot()["queued"] and time.monotonic() < deadline:
            time.sleep(0.005)
        adapter.block_writes = False
        recovered = worker.submit(
            np.array([0.1, -0.1], dtype=np.float32),
            sample_rate=48_000,
            generation=17,
        )
        assert recovered.wait(timeout=1.0)
        assert recovered.cancelled is False
        assert recovered.error is None
        assert worker.status_snapshot()["last_error"] is None
    finally:
        worker.shutdown()


def test_idle_keepalive_feeds_native_silence_without_reopening_stream() -> None:
    from askme.voice.output.audio_output_worker import (
        AudioOutputConfig,
        AudioOutputWorker,
    )

    adapter = _RecordingAdapter()
    worker = AudioOutputWorker(
        adapter,
        config=AudioOutputConfig(
            native_sample_rate=48_000,
            channels=2,
            period_ms=10,
            cold_preroll_ms=0,
            warm_leadin_ms=0,
            idle_keepalive=True,
            warm_hold_seconds=0.08,
        ),
    )
    try:
        time.sleep(0.03)
        assert adapter.open_count == 0
        worker.warm_for()
        assert adapter.first_write.wait(timeout=0.25)
    finally:
        worker.shutdown()

    assert adapter.open_count == 1
    assert adapter.close_count == 1
    first_period = np.frombuffer(adapter.writes[0], dtype=np.int16)
    assert len(first_period) == 480 * 2
    assert np.count_nonzero(first_period) == 0


def test_worker_recovers_after_an_output_stream_failure() -> None:
    from askme.voice.output.audio_output_worker import (
        AudioOutputConfig,
        AudioOutputWorker,
    )

    class _FailOnceAdapter(_RecordingAdapter):
        def __init__(self) -> None:
            super().__init__()
            self._failed = False

        def write(self, pcm: bytes) -> None:
            if not self._failed:
                self._failed = True
                raise BrokenPipeError("device disconnected")
            super().write(pcm)

    adapter = _FailOnceAdapter()
    worker = AudioOutputWorker(
        adapter,
        config=AudioOutputConfig(
            native_sample_rate=48_000,
            channels=2,
            period_ms=10,
            cold_preroll_ms=0,
            warm_leadin_ms=0,
        ),
    )
    try:
        failed = worker.submit(
            np.ones(480, dtype=np.float32) * 0.1,
            sample_rate=48_000,
            generation=21,
        )
        assert failed.wait(timeout=1.0)
        assert isinstance(failed.error, BrokenPipeError)

        recovered = worker.submit(
            np.ones(480, dtype=np.float32) * 0.2,
            sample_rate=48_000,
            generation=22,
        )
        assert recovered.wait(timeout=1.0)
        assert recovered.error is None
    finally:
        worker.shutdown()

    assert adapter.open_count == 2
    assert adapter.drop_count >= 1


class _FakeStdin:
    def __init__(self) -> None:
        self.payloads: list[bytes] = []
        self.flush_count = 0
        self.closed = False

    def write(self, payload: bytes) -> None:
        self.payloads.append(bytes(payload))

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.closed = True


class _FakeAplayProcess:
    def __init__(self) -> None:
        self.stdin = _FakeStdin()
        self.returncode: int | None = None
        self.terminate_count = 0
        self.kill_count = 0
        self.wait_count = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_count += 1
        self.returncode = -15

    def kill(self) -> None:
        self.kill_count += 1
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.wait_count += 1
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def test_aplay_adapter_uses_native_format_and_restarts_only_after_drop() -> None:
    from askme.voice.output.audio_output_worker import (
        AplayOutputAdapter,
        AudioOutputConfig,
    )

    commands: list[list[str]] = []
    processes: list[_FakeAplayProcess] = []

    def create_process(command: list[str], **_kwargs: object) -> _FakeAplayProcess:
        commands.append(list(command))
        process = _FakeAplayProcess()
        processes.append(process)
        return process

    adapter = AplayOutputAdapter(
        device="plughw:1,0",
        config=AudioOutputConfig(
            native_sample_rate=48_000,
            channels=2,
            period_ms=10,
            buffer_ms=80,
        ),
        executable="/usr/bin/aplay",
        process_factory=create_process,
    )

    adapter.open()
    adapter.write(b"\x01\x00\x01\x00")
    adapter.drop()
    adapter.open()
    adapter.close()

    assert len(processes) == 2
    assert processes[0].terminate_count == 1
    assert processes[1].stdin.closed is True
    assert processes[0].stdin.payloads == [b"\x01\x00\x01\x00"]
    command = commands[0]
    assert command[command.index("-r") + 1] == "48000"
    assert command[command.index("-c") + 1] == "2"
    assert "--period-time=10000" in command
    assert "--buffer-time=80000" in command


def test_worker_publishes_native_mono_reference_for_every_speech_period() -> None:
    from askme.voice.output.audio_output_worker import (
        AudioOutputConfig,
        AudioOutputWorker,
    )

    references: list[tuple[np.ndarray, int]] = []
    adapter = _RecordingAdapter()
    worker = AudioOutputWorker(
        adapter,
        config=AudioOutputConfig(
            native_sample_rate=48_000,
            channels=2,
            period_ms=10,
            idle_keepalive=False,
        ),
        render_reference=lambda samples, sample_rate: references.append(
            (samples.copy(), sample_rate)
        ),
    )
    try:
        ticket = worker.submit(
            np.full(960, 0.25, dtype=np.float32),
            sample_rate=48_000,
            generation=11,
        )
        assert ticket.wait(timeout=1.0)
    finally:
        worker.shutdown()

    assert len(references) == 2
    assert {sample_rate for _, sample_rate in references} == {48_000}
    rendered_reference = np.concatenate([samples for samples, _ in references])
    assert rendered_reference.shape == (960,)
    assert np.allclose(rendered_reference, 0.25, atol=1 / 32767)
