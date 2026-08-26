from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.eval import measure_volcengine_tts_latency as measure
from scripts.eval import report_voice_latency


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        self.value += 0.01
        return self.value


class FakeClient:
    created: list[FakeClient] = []
    fail_case_id: str | None = None
    empty_case_id: str | None = None
    cancelled_case_id: str | None = None
    failure_message = "provider failed"
    prewarm_failure_call: int | None = None

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.settings = dict(settings)
        self.close_called = False
        self.prewarm_calls = 0
        self.synthesized: list[str] = []
        FakeClient.created.append(self)

    def prewarm(self) -> dict[str, Any]:
        self.prewarm_calls += 1
        if self.prewarm_calls == FakeClient.prewarm_failure_call:
            return {"ok": False, "status": "failed", "reason": "network"}
        return {
            "ok": True,
            "status": "opened" if self.prewarm_calls == 1 else "reused",
        }

    def synthesize(self, text: str, *, on_audio: Any) -> SimpleNamespace:
        self.synthesized.append(text)
        case_id = _text_to_case_id(text)
        if case_id == FakeClient.empty_case_id:
            return SimpleNamespace(status="finished", audio_bytes=0, audio_chunks=0)
        if case_id == FakeClient.fail_case_id:
            on_audio(b"ab")
            raise RuntimeError(FakeClient.failure_message)
        on_audio(b"abcd")
        on_audio(b"ef")
        status = "cancelled" if case_id == FakeClient.cancelled_case_id else "finished"
        return SimpleNamespace(status=status, audio_bytes=6, audio_chunks=2)

    def close(self) -> None:
        self.close_called = True


def _text_to_case_id(text: str) -> str | None:
    for case in measure.load_corpus():
        if case["text"] == text:
            return str(case["case_id"])
    return None


def _config(*, legacy: bool = False) -> dict[str, Any]:
    tts: dict[str, Any] = {
        "volcengine_tts_api_key": "" if legacy else "test-api-key",
        "volcengine_tts_app_id": "legacy-app" if legacy else "",
        "volcengine_tts_access_key": "legacy-access" if legacy else "",
        "volcengine_tts_resource_id": "seed-tts-test",
        "volcengine_tts_speaker": "speaker-test",
        "volcengine_tts_model": "seed-test-model",
        "volcengine_tts_sample_rate": 24000,
        "volcengine_tts_audio_format": "pcm",
        "volcengine_tts_session_timeout_seconds": 12.0,
        "volcengine_tts_ws_url": (
            "wss://user:password@example.test/api/v3/tts?access_token=query-secret"
        ),
    }
    return {"voice": {"tts": tts}}


@pytest.fixture(autouse=True)
def reset_fake_client(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "VOLCENGINE_TTS_API_KEY",
        "VOLCENGINE_TTS_APP_ID",
        "VOLCENGINE_TTS_ACCESS_KEY",
        "VOLCENGINE_TTS_RESOURCE_ID",
        "VOLCENGINE_TTS_SPEAKER",
    ):
        monkeypatch.delenv(name, raising=False)
    FakeClient.created.clear()
    FakeClient.fail_case_id = None
    FakeClient.empty_case_id = None
    FakeClient.cancelled_case_id = None
    FakeClient.failure_message = "provider failed"
    FakeClient.prewarm_failure_call = None


def test_load_corpus_has_exactly_20_distinct_cases() -> None:
    cases = measure.load_corpus()

    assert len(cases) == 20
    assert len({case["case_id"] for case in cases}) == 20


def test_cold_mode_uses_new_client_per_case_and_records_software_boundaries(
    tmp_path: Path,
) -> None:
    out = tmp_path / "cold.json"

    payload, ok = measure.run_measurement(
        mode="cold",
        output_path=out,
        config=_config(),
        clock=FakeClock(),
        client_factory=FakeClient,
    )

    assert ok is True
    assert payload["schema_version"] == measure.SCHEMA_VERSION
    assert payload["provider"] == "volcengine"
    assert payload["model"] == "seed-test-model"
    assert payload["transport"] == "bidirectional_websocket_v3"
    assert payload["evidence_type"] == "measured"
    assert payload["corpus_id"] == measure.CORPUS_ID
    assert payload["sample_count"] == 20
    assert payload["failure_count"] == 0
    assert len(FakeClient.created) == 20
    assert FakeClient.created[0].settings["connect_timeout"] == 10.0
    assert FakeClient.created[0].settings["session_timeout"] == 12.0
    assert all(client.close_called for client in FakeClient.created)
    assert all(client.prewarm_calls == 0 for client in FakeClient.created)
    assert {sample["connection_label"] for sample in payload["samples"]} == {
        "cold_new_connection"
    }
    assert all(sample["audio_chunks"] == 2 for sample in payload["samples"])
    assert all(sample["audio_bytes"] == 6 for sample in payload["samples"])
    assert all(sample["provider_first_pcm_ms"] == 10.0 for sample in payload["samples"])
    assert all(sample["buffer_commit_ms"] == 20.0 for sample in payload["samples"])
    assert all(sample["total_synthesis_ms"] == 30.0 for sample in payload["samples"])
    assert all(sample["provider_status"] == "finished" for sample in payload["samples"])
    assert out.exists()


def test_warm_mode_reuses_one_client_and_prewarmed_connection(tmp_path: Path) -> None:
    payload, ok = measure.run_measurement(
        mode="warm",
        output_path=tmp_path / "warm.json",
        config=_config(),
        clock=FakeClock(),
        client_factory=FakeClient,
    )

    assert ok is True
    assert len(FakeClient.created) == 1
    client = FakeClient.created[0]
    assert client.close_called is True
    assert client.prewarm_calls == 20
    assert len(client.synthesized) == 20
    labels = [sample["connection_label"] for sample in payload["samples"]]
    assert labels[0] == "warm_opened"
    assert set(labels[1:]) == {"warm_reused"}
    assert all(sample["connection_mode"] == "warm" for sample in payload["samples"])


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {
                "volcengine_tts_api_key": "",
                "volcengine_tts_app_id": "",
                "volcengine_tts_access_key": "",
            },
            "credentials are missing",
        ),
        ({"volcengine_tts_resource_id": ""}, "resource is missing"),
        ({"volcengine_tts_speaker": ""}, "speaker is missing"),
    ],
)
def test_required_provider_settings_fail_fast_before_client_creation(
    changes: dict[str, str],
    message: str,
    tmp_path: Path,
) -> None:
    config = _config()
    config["voice"]["tts"].update(changes)
    out = tmp_path / "missing.json"

    with pytest.raises(measure.MeasurementError, match=message):
        measure.run_measurement(
            mode="cold",
            output_path=out,
            config=config,
            client_factory=FakeClient,
        )

    assert not FakeClient.created
    assert not out.exists()


def test_unresolved_model_placeholder_uses_actual_resource_id() -> None:
    config = _config()
    config["voice"]["tts"]["volcengine_tts_model"] = (
        "${VOLCENGINE_TTS_RESOURCE_ID}"
    )

    settings = measure.build_volcengine_settings(config)

    assert settings["model"] == "seed-tts-test"


def test_legacy_app_and_access_key_credentials_are_supported(tmp_path: Path) -> None:
    payload, ok = measure.run_measurement(
        mode="cold",
        output_path=tmp_path / "legacy.json",
        config=_config(legacy=True),
        clock=FakeClock(),
        client_factory=FakeClient,
    )

    assert ok is True
    assert payload["provider_metadata"]["auth_mode"] == "legacy_app_access_key"
    assert "legacy-app" not in json.dumps(payload)
    assert "legacy-access" not in json.dumps(payload)


def test_default_client_factory_adapts_legacy_and_split_timeout_fields() -> None:
    settings = measure.build_volcengine_settings(_config())

    client = measure._default_client_factory(settings)
    try:
        provider_config = client._config
        assert provider_config.timeout == 12.0
        if hasattr(provider_config, "connect_timeout"):
            assert provider_config.connect_timeout == 10.0
        if hasattr(provider_config, "session_timeout"):
            assert provider_config.session_timeout == 12.0
        assert provider_config.extra_req_params == {}
    finally:
        client.close()


def test_partial_provider_failure_preserves_metrics_and_redacts_credentials(
    tmp_path: Path,
) -> None:
    secret = "api-super-secret"
    app_id = "app-super-secret"
    access_key = "access-super-secret"
    config = _config()
    config["voice"]["tts"].update(
        {
            "volcengine_tts_api_key": secret,
            "volcengine_tts_app_id": app_id,
            "volcengine_tts_access_key": access_key,
        }
    )
    FakeClient.fail_case_id = "tts-zh-03"
    FakeClient.failure_message = (
        f"Authorization: Bearer {secret}; X-Api-App-ID={app_id}; "
        f"X-Api-Access-Key: {access_key}; token=url-token"
    )

    payload, ok = measure.run_measurement(
        mode="cold",
        output_path=tmp_path / "failed.json",
        config=config,
        clock=FakeClock(),
        client_factory=FakeClient,
    )

    assert ok is False
    assert payload["status"] == "failed"
    assert payload["failure_count"] == 1
    failed = [sample for sample in payload["samples"] if sample["status"] == "failed"]
    assert len(failed) == 1
    assert failed[0]["case_id"] == "tts-zh-03"
    assert failed[0]["provider_status"] == "exception"
    assert failed[0]["audio_chunks"] == 1
    assert failed[0]["audio_bytes"] == 2
    assert failed[0]["provider_first_pcm_ms"] == 10.0
    assert failed[0]["buffer_commit_ms"] == 20.0
    assert failed[0]["total_synthesis_ms"] == 30.0
    serialized = json.dumps(payload)
    assert secret not in serialized
    assert app_id not in serialized
    assert access_key not in serialized
    assert "url-token" not in serialized
    assert "[redacted]" in serialized


def test_empty_audio_and_cancelled_status_are_failed_evidence(tmp_path: Path) -> None:
    FakeClient.empty_case_id = "tts-zh-01"
    FakeClient.cancelled_case_id = "tts-zh-02"

    payload, ok = measure.run_measurement(
        mode="warm",
        output_path=tmp_path / "invalid-provider-results.json",
        config=_config(),
        clock=FakeClock(),
        client_factory=FakeClient,
    )

    assert ok is False
    assert payload["failure_count"] == 2
    first, second = payload["samples"][:2]
    assert first["status"] == "failed"
    assert first["audio_bytes"] == 0
    assert first["provider_first_pcm_ms"] is None
    assert "without PCM" in first["error"]
    assert second["status"] == "failed"
    assert second["provider_status"] == "cancelled"
    assert "provider status: cancelled" in second["error"]


def test_warm_prewarm_failure_is_recorded_and_later_cases_continue(tmp_path: Path) -> None:
    FakeClient.prewarm_failure_call = 2

    payload, ok = measure.run_measurement(
        mode="warm",
        output_path=tmp_path / "prewarm-failed.json",
        config=_config(),
        clock=FakeClock(),
        client_factory=FakeClient,
    )

    assert ok is False
    assert payload["failure_count"] == 1
    assert payload["samples"][1]["status"] == "failed"
    assert payload["samples"][1]["connection_label"] == "warm_failed"
    assert payload["samples"][2]["status"] == "passed"
    assert len(FakeClient.created[0].synthesized) == 19


def test_warm_client_construction_failures_still_write_redacted_evidence(
    tmp_path: Path,
) -> None:
    calls = 0

    def failing_factory(_settings: Mapping[str, Any]) -> Any:
        nonlocal calls
        calls += 1
        raise RuntimeError("X-Api-Key: test-api-key; connection unavailable")

    out = tmp_path / "constructor-failed.json"
    payload, ok = measure.run_measurement(
        mode="warm",
        output_path=out,
        config=_config(),
        client_factory=failing_factory,
    )

    assert ok is False
    assert calls == 20
    assert payload["failure_count"] == 20
    assert all(sample["status"] == "failed" for sample in payload["samples"])
    assert "test-api-key" not in out.read_text(encoding="utf-8")
    assert "[redacted]" in out.read_text(encoding="utf-8")


def test_case_delay_is_recorded_and_excluded_from_case_latency(tmp_path: Path) -> None:
    sleeps: list[float] = []

    payload, ok = measure.run_measurement(
        mode="cold",
        output_path=tmp_path / "delay.json",
        config=_config(),
        clock=FakeClock(),
        sleeper=sleeps.append,
        client_factory=FakeClient,
        case_delay_ms=4500.0,
    )

    assert ok is True
    assert payload["case_delay_ms"] == 4500.0
    assert sleeps == [4.5] * 19
    assert payload["samples"][0]["provider_first_pcm_ms"] == 10.0


def test_existing_evidence_is_rejected_before_network_work(tmp_path: Path) -> None:
    out = tmp_path / "existing.json"
    out.write_text('{"sentinel": true}\n', encoding="utf-8")

    with pytest.raises(measure.MeasurementError, match="output already exists"):
        measure.run_measurement(
            mode="cold",
            output_path=out,
            config=_config(),
            client_factory=FakeClient,
        )

    assert not FakeClient.created
    assert out.read_text(encoding="utf-8") == '{"sentinel": true}\n'


def test_explicit_overwrite_atomically_replaces_existing_evidence(tmp_path: Path) -> None:
    out = tmp_path / "existing.json"
    out.write_text('{"sentinel": true}\n', encoding="utf-8")

    payload, ok = measure.run_measurement(
        mode="warm",
        output_path=out,
        config=_config(),
        clock=FakeClock(),
        client_factory=FakeClient,
        overwrite=True,
    )

    assert ok is True
    assert json.loads(out.read_text(encoding="utf-8")) == payload
    assert not list(tmp_path.glob(".*.tmp"))


def test_no_clobber_publish_rejects_target_created_after_initial_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    out = tmp_path / "raced.json"
    sentinel = '{"created_by": "competing_collector"}\n'

    def competing_link(_source: Any, destination: Any) -> None:
        Path(destination).write_text(sentinel, encoding="utf-8")
        raise FileExistsError(str(destination))

    monkeypatch.setattr(measure.os, "link", competing_link)

    with pytest.raises(measure.MeasurementError, match="output already exists"):
        measure.atomic_write_json(out, {"new": "evidence"})

    assert out.read_text(encoding="utf-8") == sentinel
    assert not list(tmp_path.glob(".*.tmp"))


def test_artifact_metadata_strips_endpoint_credentials_and_query(tmp_path: Path) -> None:
    payload, _ = measure.run_measurement(
        mode="cold",
        output_path=tmp_path / "safe-metadata.json",
        config=_config(),
        clock=FakeClock(),
        client_factory=FakeClient,
    )

    metadata = payload["provider_metadata"]
    assert metadata["endpoint"] == "wss://example.test/api/v3/tts"
    serialized = json.dumps(payload)
    assert "password" not in serialized
    assert "query-secret" not in serialized
    assert "test-api-key" not in serialized


def test_default_output_path_is_unique_and_provider_scoped() -> None:
    first = measure.default_output_path("warm")
    second = measure.default_output_path("warm")

    assert first != second
    assert first.parent == Path("artifacts") / "voice"
    assert first.name.startswith("volcengine-tts-warm-")


def test_help_explicitly_says_measurement_is_not_physical_first_sound(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        measure.build_parser().parse_args(["--help"])

    assert "NOT physical first-sound latency" in capsys.readouterr().out


def test_failed_measurement_causes_nonzero_main_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        measure,
        "run_measurement",
        lambda **_kwargs: ({"status": "failed"}, False),
    )

    exit_code = measure.main(
        ["--mode", "cold", "--out", str(tmp_path / "failure.json")]
    )

    assert exit_code == 2


def test_output_is_accepted_by_unified_latency_report(tmp_path: Path) -> None:
    out = tmp_path / "experiment.json"
    measure.run_measurement(
        mode="cold",
        output_path=out,
        config=_config(),
        clock=FakeClock(),
        client_factory=FakeClient,
    )

    report = report_voice_latency.build_report(experiments=[out])

    source = next(
        item
        for item in report["sources"]
        if item["kind"] == "voice_latency_experiment"
    )
    assert source["kind"] == "voice_latency_experiment"
    assert source["evidence_type"] == "measured"
    assert source["status"] == "insufficient_evidence"
    metric = report["stage_metrics"]["tts_provider_first_pcm_ms"][0]
    assert metric["provider"] == "volcengine"
    assert metric["sample_count"] == 20
    assert metric["status"] == "insufficient_evidence"
    assert (
        report["stage_metrics"]["tts_physical_first_nonzero_ms"][0]["status"]
        == "insufficient_evidence"
    )


def test_output_is_not_provider_decision_ready_with_only_20_cases(
    tmp_path: Path,
) -> None:
    volc_path = tmp_path / "volcengine.json"
    volc_payload, _ = measure.run_measurement(
        mode="cold",
        output_path=volc_path,
        config=_config(),
        clock=FakeClock(),
        client_factory=FakeClient,
    )
    minimax_payload = json.loads(json.dumps(volc_payload))
    minimax_payload.update(
        {
            "experiment_id": "tts-minimax-comparison",
            "provider": "minimax",
            "model": "speech-test",
            "transport": "websocket",
        }
    )
    for sample in minimax_payload["samples"]:
        sample["provider_first_pcm_ms"] = 50.0
        sample["buffer_commit_ms"] = 60.0
    minimax_path = tmp_path / "minimax.json"
    minimax_path.write_text(
        json.dumps(minimax_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    report = report_voice_latency.build_report(
        experiments=[minimax_path, volc_path]
    )

    decision = report["provider_decisions"]["tts"]
    assert decision["status"] == "insufficient_evidence"
    assert decision["corpus_id"] == measure.CORPUS_ID
    assert decision["winner"] is None
    assert "at least 100 cases" in decision["reason"]
