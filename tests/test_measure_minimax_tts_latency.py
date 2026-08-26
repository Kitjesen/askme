from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import pytest

from scripts.eval import measure_minimax_tts_latency as measure
from scripts.eval import report_voice_latency


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        self.value += 0.01
        return self.value


class FakeEngine:
    created: list[FakeEngine] = []
    play_calls = 0
    fail_case_id: str | None = None
    backend_result = "minimax"
    provider_exception: BaseException | None = None

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.tts_buffer: deque[list[float]] = deque()
        self.shutdown_called = False
        self.prewarm_calls = 0
        self.generated: list[str] = []
        self._minimax_tts_model = config.get("minimax_tts_model", "speech-test")
        self._minimax_tts_transport = config.get("minimax_tts_transport", "websocket")
        self._minimax_voice_id = config.get("minimax_voice_id", "voice-test")
        self._minimax_sample_rate = int(config.get("minimax_sample_rate", 32000))
        self._sample_rate = int(config.get("sample_rate", 32000))
        self._minimax_audio_format = config.get("minimax_audio_format", "pcm")
        FakeEngine.created.append(self)

    def snapshot(self) -> dict[str, Any]:
        return {
            "sample_rate": self._sample_rate,
            "minimax": {
                "model": self._minimax_tts_model,
                "transport": self._minimax_tts_transport,
                "voice_id": self._minimax_voice_id,
                "sample_rate": self._minimax_sample_rate,
                "format": self._minimax_audio_format,
            },
        }

    def prewarm_provider_session(self) -> dict[str, Any]:
        self.prewarm_calls += 1
        return {"ok": True, "status": "opened" if self.prewarm_calls == 1 else "reused"}

    def _get_generation(self) -> int:
        return 1

    def _generate_audio(self, text: str, generation: int) -> str:
        self.generated.append(text)
        if FakeEngine.provider_exception is not None:
            raise FakeEngine.provider_exception
        case_id = _text_to_case_id(text)
        if FakeEngine.fail_case_id == case_id:
            return self._use_minimax_fallback(text, generation)
        pending: list[list[float]] = []
        state = {"first_flush": True}
        self._commit_minimax_samples_for_generation(
            generation,
            pending,
            state,
            samples=[0.1, 0.2, 0.3],
        )
        self._commit_minimax_samples_for_generation(
            generation,
            pending,
            state,
            samples=None,
            flush=True,
        )
        return FakeEngine.backend_result

    def _use_minimax_fallback(self, text: str, generation: int) -> str:
        return "edge"

    def _commit_minimax_samples_for_generation(self, generation: int, pending: list[Any], state: dict[str, Any], *, samples: Any = None, flush: bool = False) -> bool:
        if samples is not None:
            pending.append(samples)
        if flush:
            self._flush_minimax_pending(pending, state)
        return True

    def _flush_minimax_pending(self, pending: list[Any], state: dict[str, Any]) -> None:
        if pending:
            self.tts_buffer.append(pending.pop(0))

    def speak(self, text: str) -> None:  # pragma: no cover - defensive
        FakeEngine.play_calls += 1
        raise AssertionError("physical playback path must not be used")

    def shutdown(self) -> None:
        self.shutdown_called = True


def _config() -> dict[str, Any]:
    return {
        "voice": {
            "tts": {
                "backend": "minimax",
                "minimax_api_key": "test-key",
                "minimax_tts_model": "speech-test",
                "minimax_tts_transport": "websocket",
                "minimax_voice_id": "voice-test",
                "minimax_sample_rate": 32000,
                "sample_rate": 32000,
                "minimax_audio_format": "pcm",
            }
        }
    }


def _text_to_case_id(text: str) -> str | None:
    for case in measure.load_corpus():
        if case["text"] == text:
            return str(case["case_id"])
    return None


@pytest.fixture(autouse=True)
def reset_fake_engine() -> None:
    FakeEngine.created.clear()
    FakeEngine.play_calls = 0
    FakeEngine.fail_case_id = None
    FakeEngine.backend_result = "minimax"
    FakeEngine.provider_exception = None


def test_load_corpus_has_20_distinct_case_ids() -> None:
    cases = measure.load_corpus()

    assert len(cases) == 20
    assert len({case["case_id"] for case in cases}) == 20


def test_cold_mode_uses_new_engine_per_case_and_never_plays(tmp_path: Path) -> None:
    out = tmp_path / "cold.json"

    payload, ok = measure.run_measurement(
        mode="cold",
        output_path=out,
        config=_config(),
        clock=FakeClock(),
        engine_factory=FakeEngine,
    )

    assert ok is True
    assert payload["schema_version"] == measure.SCHEMA_VERSION
    assert payload["stage"] == "tts"
    assert payload["provider"] == "minimax"
    assert payload["model"] == "speech-test"
    assert payload["transport"] == "websocket"
    assert payload["evidence_type"] == "measured"
    assert payload["corpus_id"] == measure.CORPUS_ID
    assert payload["sample_count"] == 20
    assert len(FakeEngine.created) == 20
    assert all(engine.shutdown_called for engine in FakeEngine.created)
    assert FakeEngine.play_calls == 0
    assert {sample["connection_label"] for sample in payload["samples"]} == {"cold_new_session"}
    assert all(sample["provider_first_pcm_ms"] > 0 for sample in payload["samples"])
    assert all(sample["buffer_commit_ms"] >= sample["provider_first_pcm_ms"] for sample in payload["samples"])
    assert out.exists()


def test_warm_mode_reuses_one_engine_and_labels_opened_then_reused(tmp_path: Path) -> None:
    out = tmp_path / "warm.json"

    payload, ok = measure.run_measurement(
        mode="warm",
        output_path=out,
        config=_config(),
        clock=FakeClock(),
        engine_factory=FakeEngine,
    )

    assert ok is True
    assert len(FakeEngine.created) == 1
    assert FakeEngine.created[0].shutdown_called is True
    labels = [sample["connection_label"] for sample in payload["samples"]]
    assert labels[0] == "warm_opened"
    assert set(labels[1:]) == {"warm_reused"}
    assert all(sample["connection_mode"] == "warm" for sample in payload["samples"])


def test_missing_credentials_fail_fast_without_printing_secret(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

    with pytest.raises(measure.MeasurementError) as exc:
        measure.run_measurement(
            mode="cold",
            output_path=tmp_path / "missing.json",
            config={"voice": {"tts": {"minimax_api_key": ""}}},
            engine_factory=FakeEngine,
        )

    assert "MiniMax TTS credentials are missing" in str(exc.value)
    assert "Bearer" not in str(exc.value)
    assert not FakeEngine.created


def test_failed_sample_has_no_latency_numbers_and_causes_nonzero_status(tmp_path: Path) -> None:
    FakeEngine.fail_case_id = "tts-zh-03"

    payload, ok = measure.run_measurement(
        mode="cold",
        output_path=tmp_path / "failure.json",
        config=_config(),
        clock=FakeClock(),
        engine_factory=FakeEngine,
    )

    assert ok is False
    assert payload["status"] == "failed"
    failed = [sample for sample in payload["samples"] if sample["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["case_id"] == "tts-zh-03"
    assert "provider_first_pcm_ms" not in failed[0]
    assert "buffer_commit_ms" not in failed[0]
    assert "total_synthesis_ms" not in failed[0]
    assert "fallback suppressed" in failed[0]["error"]


def test_failure_artifact_redacts_actual_api_key_value(tmp_path: Path) -> None:
    secret = "sk-test-secret-value"
    config = _config()
    config["voice"]["tts"]["minimax_api_key"] = secret
    FakeEngine.provider_exception = RuntimeError(
        f"Authorization: Bearer {secret}; access_token={secret}"
    )

    payload, ok = measure.run_measurement(
        mode="cold",
        output_path=tmp_path / "redacted.json",
        config=config,
        clock=FakeClock(),
        engine_factory=FakeEngine,
    )

    assert ok is False
    serialized = __import__("json").dumps(payload)
    assert secret not in serialized
    assert "Bearer sk-" not in serialized
    assert "[redacted]" in serialized


def test_main_help_states_not_physical_first_sound(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        measure.build_parser().parse_args(["--help"])

    captured = capsys.readouterr()
    assert "NOT physical first-sound latency" in captured.out


def test_atomic_output_is_valid_experiment_schema(tmp_path: Path) -> None:
    out = tmp_path / "schema.json"

    payload, ok = measure.run_measurement(
        mode="cold",
        output_path=out,
        config=_config(),
        clock=FakeClock(),
        engine_factory=FakeEngine,
    )
    disk = __import__("json").loads(out.read_text(encoding="utf-8"))

    assert ok is True
    assert disk == payload
    assert disk["provider_metadata"]["voice_id"] == "voice-test"
    assert disk["measurement_boundary"] == "client_decoded_provider_pcm_and_tts_buffer_commit_no_physical_playback"


def test_existing_evidence_is_not_overwritten_without_explicit_opt_in(
    tmp_path: Path,
) -> None:
    out = tmp_path / "existing.json"
    out.write_text('{"sentinel": true}\n', encoding="utf-8")

    with pytest.raises(measure.MeasurementError, match="output already exists"):
        measure.run_measurement(
            mode="warm",
            output_path=out,
            config=_config(),
            clock=FakeClock(),
            engine_factory=FakeEngine,
        )

    assert out.read_text(encoding="utf-8") == '{"sentinel": true}\n'


def test_case_delay_is_recorded_and_runs_between_cases(tmp_path: Path) -> None:
    out = tmp_path / "delayed.json"
    sleeps: list[float] = []

    payload, ok = measure.run_measurement(
        mode="cold",
        output_path=out,
        config=_config(),
        clock=FakeClock(),
        sleeper=sleeps.append,
        engine_factory=FakeEngine,
        case_delay_ms=250.0,
    )

    assert ok is True
    assert payload["case_delay_ms"] == 250.0
    assert sleeps == [0.25] * 19
    assert payload["samples"][0]["provider_first_pcm_ms"] == 10.0


def test_explicit_overwrite_replaces_existing_evidence(tmp_path: Path) -> None:
    out = tmp_path / "existing.json"
    out.write_text('{"sentinel": true}\n', encoding="utf-8")

    payload, ok = measure.run_measurement(
        mode="warm",
        output_path=out,
        config=_config(),
        clock=FakeClock(),
        engine_factory=FakeEngine,
        overwrite=True,
    )

    assert ok is True
    assert __import__("json").loads(out.read_text(encoding="utf-8")) == payload


def test_default_output_path_is_unique_and_mode_scoped() -> None:
    first = measure.default_output_path("warm")
    second = measure.default_output_path("warm")

    assert first != second
    assert first.parent == Path("artifacts") / "voice"
    assert first.name.startswith("minimax-tts-warm-")


def test_output_is_accepted_by_unified_latency_report(tmp_path: Path) -> None:
    out = tmp_path / "experiment.json"
    measure.run_measurement(
        mode="cold",
        output_path=out,
        config=_config(),
        clock=FakeClock(),
        engine_factory=FakeEngine,
    )

    report = report_voice_latency.build_report(experiments=[out])

    sources = {source["id"]: source for source in report["sources"]}
    assert sources["latency_experiment_1"]["kind"] == "voice_latency_experiment"
    assert sources["latency_experiment_1"]["evidence_type"] == "measured"
    assert sources["latency_experiment_1"]["status"] == "insufficient_evidence"
    metric = report["stage_metrics"]["tts_provider_first_pcm_ms"][0]
    assert metric["provider"] == "minimax"
    assert metric["sample_count"] == 20
    assert (
        report["stage_metrics"]["tts_physical_first_nonzero_ms"][0]["status"]
        == "insufficient_evidence"
    )
