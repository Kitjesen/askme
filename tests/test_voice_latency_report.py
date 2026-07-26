from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from askme.voice.diagnostics.full_duplex_hardware import evaluate_hardware_run
from askme.voice.diagnostics.hardware_audio_capture import (
    build_instrumented_trial_evidence,
    build_manual_trial_evidence,
)
from scripts.eval import report_voice_latency


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _stage_experiment(
    *,
    experiment_id: str,
    stage: str,
    provider: str,
    model: str,
    corpus_id: str = "voice-zh-20-v1",
    sample_count: int = 100,
    offset_ms: float = 0.0,
    evidence_type: str = "measured",
) -> dict:
    samples: list[dict] = []
    for index in range(sample_count):
        sample: dict = {"case_id": f"case-{index:02d}"}
        if stage == "asr":
            sample["endpoint_ms"] = 140.0 + offset_ms + index
        elif stage == "llm":
            sample["first_content_ms"] = 100.0 + offset_ms + index
            sample["first_semantic_clause_ms"] = 380.0 + offset_ms + index
        elif stage == "tts":
            sample["provider_first_pcm_ms"] = 300.0 + offset_ms + index
            sample["buffer_commit_ms"] = 340.0 + offset_ms + index
            sample["physical_first_nonzero_ms"] = 390.0 + offset_ms + index
        elif stage == "barge_in":
            sample["physical_stop_ms"] = 120.0 + offset_ms + index
        samples.append(sample)
    return {
        "schema_version": "askme.voice_latency_experiment.v1",
        "experiment_id": experiment_id,
        "stage": stage,
        "provider": provider,
        "model": model,
        "transport": "websocket",
        "corpus_id": corpus_id,
        "evidence_type": evidence_type,
        "samples": samples,
    }


def _hardware_status() -> dict:
    now = datetime.now(UTC).isoformat()
    return {
        "status": "ok",
        "snapshot_at": now,
        "voice_pipeline_status": {
            "pipeline_ok": True,
            "recorded_at": now,
            "media": {
                "full_duplex": {
                    "enabled": True,
                    "reason": "verified_echo_control",
                    "echo_control": "hardware",
                    "aec_backend": "hardware",
                }
            },
        },
    }


def _physical_trial_evidence(*, scenario: str, latency_ms: float) -> dict:
    capture_role = (
        "isolated_speaker_monitor" if scenario == "human_overlap" else "room_acoustic_monitor"
    )
    return build_instrumented_trial_evidence(
        evidence_kind="physical_acoustic",
        method="rms_threshold_v2",
        capture={
            "source_label": "microphone",
            "source_evidence_kind": "physical_acoustic",
            "instrumented": True,
            "device_id": "speaker-probe",
            "stream_id": "speaker-stream",
            "channel": 0,
            "clock_id": "capture-clock",
            "role": capture_role,
            "isolated_from_reference": scenario == "human_overlap",
        },
        reference={
            "event": "human_speech_onset" if scenario == "human_overlap" else "speech_end",
            "instrumented": True,
            "device_id": "speech-reference",
            "stream_id": "speech-stream",
            "channel": 0,
            "clock_id": "capture-clock",
        },
        reference_timestamp_s=100.0,
        event_timestamp_s=100.0 + latency_ms / 1000.0,
        calibration={
            "performed": True,
            "source_label": "microphone",
            "source_evidence_kind": "physical_acoustic",
            "sample_rate_hz": 48_000,
            "valid_frame_count": 200,
            "threshold": 0.02,
        },
        dropped_frames=0,
        clock_id="capture-clock",
    )


def _passing_hardware_payload(*, summary: dict) -> dict:
    status = _hardware_status()
    overlap_summary = summary["human_overlap"]["speaker_stop_latency_ms"]
    stop_latency = float(overlap_summary["p95"])
    response_summary = summary.get("assistant_response", {}).get(
        "speech_end_to_physical_first_sound_ms",
        {"p95": 900.0},
    )
    first_sound_latency = float(response_summary["p95"])
    return evaluate_hardware_run(
        config={
            "voice": {
                "full_duplex": {
                    "enabled": True,
                    "echo_control": "hardware",
                    "echo_control_verified": True,
                }
            }
        },
        metadata={
            "operating_system": "Windows",
            "python_version": "3.11",
            "room": "lab-a",
            "audio_device": "speakerphone-a",
            "audio_driver": "WASAPI",
            "input_device_id": "mic-1",
            "output_device_id": "speaker-1",
            "input_sample_rate_hz": 16_000,
            "output_sample_rate_hz": 32_000,
            "aec_backend": "hardware",
        },
        speaker_only_trials=[
            {
                "false_barge_in": False,
                "runtime_status": status,
                **build_manual_trial_evidence(
                    method="manual_observation",
                    reference_event="speaker_only_false_barge_in",
                    observed_timestamp_s=10.0 + index,
                ),
            }
            for index in range(100)
        ],
        overlap_trials=[
            {
                "detected": True,
                "speaker_stop_latency_ms": stop_latency,
                "runtime_status": status,
                **_physical_trial_evidence(
                    scenario="human_overlap",
                    latency_ms=stop_latency,
                ),
            }
            for _ in range(100)
        ],
        response_trials=[
            {
                "heard": True,
                "audio_class": "semantic",
                "speech_end_to_first_sound_ms": first_sound_latency,
                "speech_end_to_first_semantic_audio_ms": first_sound_latency,
                "runtime_status": status,
                **_physical_trial_evidence(
                    scenario="assistant_response",
                    latency_ms=first_sound_latency,
                ),
            }
            for _ in range(100)
        ],
        require_response_trials=True,
    )


def test_report_classifies_fast_path_route_as_measured(tmp_path: Path) -> None:
    source = _write_json(
        tmp_path / "fast-path.json",
        {
            "status": "passed",
            "benchmark": {
                "route_ms": {"count": 3, "p50": 0.04, "p95": 0.08, "max": 0.1},
                "cached_pcm_queue_ms": {
                    "count": 3,
                    "p50": 0.05,
                    "p95": 0.09,
                    "max": 0.12,
                },
            },
        },
    )

    report = report_voice_latency.build_report(fast_path=source)

    metric = report["metrics"]["route_ms"]
    assert metric["evidence_type"] == "measured"
    assert metric["measurement_scope"] == "process_microbenchmark"
    assert metric["p95_ms"] == 0.08


def test_report_classifies_fast_path_first_pcm_as_projected(tmp_path: Path) -> None:
    source = _write_json(
        tmp_path / "fast-path.json",
        {
            "status": "passed",
            "benchmark": {
                "projected_speech_end_to_first_pcm_ms": {
                    "p50": 750.0,
                    "p95": 751.0,
                    "measured_on_device": False,
                },
            },
        },
    )

    report = report_voice_latency.build_report(fast_path=source)

    metric = report["metrics"]["speech_end_to_first_pcm_ms"]
    assert metric["evidence_type"] == "projected"
    assert metric["measurement_scope"] == "computed_budget"
    assert report["checks"]["required_measured_e2e_present"] is False
    assert report["status"] == "insufficient_evidence"


def test_report_blocks_pass_when_required_e2e_metric_is_projected(tmp_path: Path) -> None:
    source = _write_json(
        tmp_path / "fast-path.json",
        {
            "status": "passed",
            "benchmark": {
                "projected_speech_end_to_first_pcm_ms": {
                    "p50": 700.0,
                    "p95": 800.0,
                    "measured_on_device": False,
                },
            },
        },
    )

    report = report_voice_latency.build_report(fast_path=source)

    assert report["status"] == "insufficient_evidence"
    assert (
        "speech_end_to_physical_first_semantic_audio_ms"
        in report["evidence_summary"]["missing_required_metrics"]
    )


def test_report_rejects_legacy_entry_latency_as_product_grade_instrumented(
    tmp_path: Path,
) -> None:
    source = _write_json(
        tmp_path / "hardware.json",
        {
            "target": "askme-full-duplex-target-hardware",
            "status": "passed",
            "latency_source": "entry",
            "summary": {
                "human_overlap": {
                    "count": 20,
                    "detected": 20,
                    "speaker_stop_latency_ms": {
                        "count": 20,
                        "p50": 90.0,
                        "p95": 180.0,
                        "p99": 220.0,
                    },
                },
            },
        },
    )

    report = report_voice_latency.build_report(hardware=source)

    source_record = next(item for item in report["sources"] if item["id"] == "full_duplex_hardware")
    assert source_record["status"] == "invalid"
    assert "barge_in_to_speaker_stop_ms" not in report["metrics"]
    assert report["checks"]["required_measured_e2e_present"] is False


def test_report_accepts_physical_semantic_audio_as_required_measured_evidence(
    tmp_path: Path,
) -> None:
    source = _write_json(
        tmp_path / "hardware.json",
        _passing_hardware_payload(
            summary={
                "human_overlap": {
                    "count": 20,
                    "detected": 20,
                    "speaker_stop_latency_ms": {
                        "count": 20,
                        "p50": 90.0,
                        "p95": 180.0,
                        "p99": 220.0,
                    },
                },
                "assistant_response": {
                    "count": 20,
                    "heard": 20,
                    "speech_end_to_physical_first_sound_ms": {
                        "count": 20,
                        "p50": 700.0,
                        "p95": 900.0,
                        "p99": 1050.0,
                    },
                },
            }
        ),
    )

    report = report_voice_latency.build_report(hardware=source)

    metric = report["metrics"]["speech_end_to_physical_first_semantic_audio_ms"]
    assert metric["evidence_type"] == "measured"
    assert (
        metric["measurement_scope"]
        == "target_hardware_physical_acoustic_instrumented_semantic"
    )
    assert metric["p95_ms"] == 900.0
    assert report["checks"]["required_measured_e2e_present"] is True
    assert report["status"] == "passed"


def test_ack_first_sound_cannot_satisfy_physical_semantic_audio_gate(
    tmp_path: Path,
) -> None:
    payload = _passing_hardware_payload(
        summary={
            "human_overlap": {
                "speaker_stop_latency_ms": {"p95": 180.0},
            },
            "assistant_response": {
                "speech_end_to_physical_first_sound_ms": {"p95": 300.0},
            },
        }
    )
    for trial in payload["trials"]["assistant_response"]:
        trial["audio_class"] = "ack"
    source = _write_json(tmp_path / "ack-only-hardware.json", payload)

    report = report_voice_latency.build_report(hardware=source)

    hardware_source = next(
        item for item in report["sources"] if item["id"] == "full_duplex_hardware"
    )
    assert hardware_source["status"] == "invalid"
    assert "semantic-audio" in hardware_source["error"]
    assert report["checks"]["required_measured_e2e_present"] is False
    assert (
        "speech_end_to_physical_first_semantic_audio_ms"
        in report["evidence_summary"]["missing_required_metrics"]
    )


def test_required_measured_latency_needs_one_hundred_samples_and_a_p95(
    tmp_path: Path,
) -> None:
    source = _write_json(
        tmp_path / "hardware.json",
        {
            "status": "passed",
            "latency_source": "entry",
            "summary": {
                "human_overlap": {
                    "count": 19,
                    "speaker_stop_latency_ms": {"count": 19, "p50": 90.0},
                },
                "assistant_response": {
                    "count": 19,
                    "speech_end_to_physical_first_sound_ms": {
                        "count": 19,
                        "p50": 700.0,
                    },
                },
            },
        },
    )

    report = report_voice_latency.build_report(hardware=source)

    assert report["status"] == "insufficient_evidence"
    assert set(report["evidence_summary"]["missing_required_metrics"]) == {
        "barge_in_to_physical_speaker_stop_ms",
        "speech_end_to_physical_first_semantic_audio_ms",
    }


def test_failed_hardware_source_cannot_pass_on_latency_metrics_alone(
    tmp_path: Path,
) -> None:
    payload = _passing_hardware_payload(
        summary={
            "human_overlap": {
                "speaker_stop_latency_ms": {"p95": 180.0},
            },
            "assistant_response": {
                "speech_end_to_physical_first_sound_ms": {"p95": 900.0},
            },
        }
    )
    payload["status"] = "failed"
    payload["failed_checks"] = ["speaker_only_no_false_barge_in"]
    source = _write_json(
        tmp_path / "hardware.json",
        payload,
    )

    report = report_voice_latency.build_report(hardware=source)

    assert report["checks"]["required_measured_e2e_present"] is True
    assert report["checks"]["hardware_full_duplex_passed"] is False
    assert report["status"] == "failed"


def test_report_keeps_legacy_stopwatch_latency_manual_and_non_product_grade(
    tmp_path: Path,
) -> None:
    source = _write_json(
        tmp_path / "hardware.json",
        {
            "status": "passed",
            "latency_source": "stopwatch",
            "summary": {
                "human_overlap": {
                    "count": 20,
                    "speaker_stop_latency_ms": {
                        "count": 20,
                        "p50": 120.0,
                        "p95": 210.0,
                        "p99": 260.0,
                    },
                },
            },
        },
    )

    report = report_voice_latency.build_report(hardware=source)

    metric = report["metrics"]["manual_barge_in_to_speaker_stop_ms"]
    assert metric["evidence_type"] == "manual"
    assert metric["measurement_scope"] == "legacy_target_hardware_unproven"
    assert "schema-v2" in " ".join(metric["limitations"])
    assert report["checks"]["required_measured_e2e_present"] is False


def test_report_never_uses_simulated_scenario_latency_for_product_pass(
    tmp_path: Path,
) -> None:
    source = _write_json(
        tmp_path / "scenario.json",
        {
            "status": "passed",
            "metrics": {
                "first_useful_response_latency_ms": 900.0,
                "tts_first_audio_ms": 300.0,
            },
        },
    )

    report = report_voice_latency.build_report(scenario=source)

    assert (
        report["metrics"]["scenario_first_useful_response_latency_ms"]["evidence_type"]
        == "simulated"
    )
    assert report["checks"]["no_simulated_metric_used_for_pass"] is True
    assert report["status"] == "insufficient_evidence"


def test_report_marks_missing_required_source_as_insufficient_evidence(
    tmp_path: Path,
) -> None:
    report = report_voice_latency.build_report(hardware=tmp_path / "missing.json")

    assert report["status"] == "insufficient_evidence"
    hardware_source = next(
        source for source in report["sources"] if source["id"] == "full_duplex_hardware"
    )
    assert hardware_source["status"] == "missing"
    assert (
        "barge_in_to_physical_speaker_stop_ms"
        in report["evidence_summary"]["missing_required_metrics"]
    )


def test_cli_writes_schema_versioned_json(tmp_path: Path) -> None:
    output = tmp_path / "report.json"

    code = report_voice_latency.main(["--out", str(output)])

    assert code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "askme.voice_latency_report.v1"


def test_cli_exits_nonzero_when_required_measured_metrics_missing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"

    assert report_voice_latency.main(["--out", str(output)]) == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "insufficient_evidence"


def test_cli_can_emit_diagnostic_report_without_failing_on_optional_sources(
    tmp_path: Path,
) -> None:
    hardware = _write_json(
        tmp_path / "hardware.json",
        {
            "status": "passed",
            "latency_source": "entry",
            "summary": {
                "human_overlap": {
                    "count": 20,
                    "speaker_stop_latency_ms": {
                        "count": 20,
                        "p50": 100.0,
                        "p95": 180.0,
                        "p99": 240.0,
                    },
                },
            },
        },
    )
    output = tmp_path / "report.json"

    code = report_voice_latency.main(
        [
            "--hardware",
            str(hardware),
            "--online-smoke",
            str(tmp_path / "none.json"),
            "--out",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert code == 1
    assert payload["checks"]["online_provider_smoke_passed"] is False
    assert payload["metrics"]["manual_barge_in_to_speaker_stop_ms"]["status"] == "passed"


def test_report_normalizes_all_decision_grade_stage_metrics(tmp_path: Path) -> None:
    paths = [
        _write_json(
            tmp_path / f"{stage}.json",
            _stage_experiment(
                experiment_id=f"{stage}-run",
                stage=stage,
                provider={
                    "asr": "volcengine",
                    "llm": "deepseek",
                    "tts": "minimax",
                    "barge_in": "askme-local",
                }[stage],
                model=f"{stage}-model",
            ),
        )
        for stage in ("asr", "llm", "tts", "barge_in")
    ]

    report = report_voice_latency.build_report(experiments=paths)

    assert set(report["stage_metrics"]) == {
        "asr_endpoint_ms",
        "llm_first_content_ms",
        "llm_first_semantic_clause_ms",
        "tts_provider_first_pcm_ms",
        "tts_buffer_commit_ms",
        "tts_physical_first_nonzero_ms",
        "barge_in_physical_stop_ms",
    }
    tts_metric = report["stage_metrics"]["tts_provider_first_pcm_ms"][0]
    assert tts_metric["provider"] == "minimax"
    assert tts_metric["model"] == "tts-model"
    assert tts_metric["transport"] == "websocket"
    assert tts_metric["sample_count"] == 100
    assert tts_metric["evidence_type"] == "measured"
    assert tts_metric["p50_ms"] is not None
    assert tts_metric["p95_ms"] is not None
    assert tts_metric["status"] == "sufficient_evidence"
    assert report["checks"]["all_supplied_experiments_decision_grade"] is True


def test_experiment_with_less_than_one_hundred_samples_withholds_p95(
    tmp_path: Path,
) -> None:
    experiment = _write_json(
        tmp_path / "small-tts.json",
        _stage_experiment(
            experiment_id="small-minimax",
            stage="tts",
            provider="minimax",
            model="speech-2.8-turbo",
            sample_count=99,
        ),
    )

    report = report_voice_latency.build_report(experiments=[experiment])

    metric = report["stage_metrics"]["tts_provider_first_pcm_ms"][0]
    assert metric["sample_count"] == 99
    assert metric["p50_ms"] is not None
    assert metric["p95_ms"] is None
    assert metric["status"] == "insufficient_evidence"
    assert report["experiments"][0]["status"] == "insufficient_evidence"
    assert report["checks"]["all_supplied_experiments_decision_grade"] is False
    assert report["status"] == "insufficient_evidence"


def test_experiment_withholds_p99_until_three_hundred_samples(
    tmp_path: Path,
) -> None:
    below = _write_json(
        tmp_path / "tts-299.json",
        _stage_experiment(
            experiment_id="tts-299",
            stage="tts",
            provider="minimax",
            model="speech-2.8-turbo",
            sample_count=299,
        ),
    )
    enough = _write_json(
        tmp_path / "tts-300.json",
        _stage_experiment(
            experiment_id="tts-300",
            stage="tts",
            provider="minimax",
            model="speech-2.8-turbo",
            sample_count=300,
        ),
    )

    report = report_voice_latency.build_report(experiments=[below, enough])
    metrics = {
        metric["experiment_id"]: metric
        for metric in report["stage_metrics"]["tts_provider_first_pcm_ms"]
    }

    assert metrics["tts-299"]["p95_ms"] is not None
    assert metrics["tts-299"]["p99_ms"] is None
    assert metrics["tts-300"]["p99_ms"] is not None


def test_tts_same_corpus_comparison_selects_lower_provider_first_pcm_p95(
    tmp_path: Path,
) -> None:
    minimax = _write_json(
        tmp_path / "minimax.json",
        _stage_experiment(
            experiment_id="tts-minimax",
            stage="tts",
            provider="MiniMax",
            model="speech-2.8-turbo",
            offset_ms=100.0,
        ),
    )
    volc = _write_json(
        tmp_path / "volc.json",
        _stage_experiment(
            experiment_id="tts-volc",
            stage="tts",
            provider="VolcEngine",
            model="seed-tts-2.0",
        ),
    )

    report = report_voice_latency.build_report(experiments=[minimax, volc])

    decision = report["provider_decisions"]["tts"]
    assert decision["status"] == "decision_ready"
    assert decision["corpus_id"] == "voice-zh-20-v1"
    assert decision["decision_metric"] == "tts_provider_first_pcm_ms"
    assert decision["winner"]["provider"] == "VolcEngine"
    assert decision["winner"]["model"] == "seed-tts-2.0"
    assert decision["winner"]["sample_count"] == 100
    assert decision["decision_scope"] == "latency_only"


def test_tts_comparison_refuses_different_case_sets_or_projected_data(
    tmp_path: Path,
) -> None:
    minimax_payload = _stage_experiment(
        experiment_id="tts-minimax",
        stage="tts",
        provider="minimax",
        model="speech-2.8-turbo",
    )
    volc_payload = _stage_experiment(
        experiment_id="tts-volc",
        stage="tts",
        provider="volcengine",
        model="seed-tts-2.0",
        evidence_type="projected",
    )
    volc_payload["samples"][0]["case_id"] = "different-case"
    paths = [
        _write_json(tmp_path / "minimax.json", minimax_payload),
        _write_json(tmp_path / "volc.json", volc_payload),
    ]

    report = report_voice_latency.build_report(experiments=paths)

    decision = report["provider_decisions"]["tts"]
    assert decision["status"] == "insufficient_evidence"
    assert decision["winner"] is None
    assert "measured" in decision["reason"] or "case" in decision["reason"]


def test_cli_accepts_repeatable_offline_experiment_inputs(tmp_path: Path) -> None:
    experiment = _write_json(
        tmp_path / "asr.json",
        _stage_experiment(
            experiment_id="asr-volc",
            stage="asr",
            provider="volcengine",
            model="seed-asr",
        ),
    )
    output = tmp_path / "report.json"

    code = report_voice_latency.main(["--experiment", str(experiment), "--out", str(output)])

    assert code == 1  # Full product pass still requires the hardware evidence.
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["experiments"][0]["experiment_id"] == "asr-volc"
    assert payload["stage_metrics"]["asr_endpoint_ms"][0]["p95_ms"] is not None


def test_skeletal_hardware_summary_cannot_claim_product_pass(tmp_path: Path) -> None:
    hardware = _write_json(
        tmp_path / "skeletal-hardware.json",
        {
            "status": "passed",
            "latency_source": "entry",
            "summary": {
                "human_overlap": {
                    "count": 20,
                    "speaker_stop_latency_ms": {
                        "count": 20,
                        "p50": 100.0,
                        "p95": 180.0,
                        "p99": 220.0,
                    },
                },
                "assistant_response": {
                    "count": 20,
                    "speech_end_to_physical_first_sound_ms": {
                        "count": 20,
                        "p50": 700.0,
                        "p95": 900.0,
                        "p99": 1050.0,
                    },
                },
            },
        },
    )

    report = report_voice_latency.build_report(hardware=hardware)

    source = next(item for item in report["sources"] if item["id"] == "full_duplex_hardware")
    assert source["status"] == "invalid"
    assert report["checks"]["required_measured_e2e_present"] is False
    assert report["status"] == "insufficient_evidence"


def test_invalid_supplied_optional_source_blocks_product_pass(tmp_path: Path) -> None:
    hardware = _write_json(
        tmp_path / "hardware.json",
        _passing_hardware_payload(
            summary={
                "human_overlap": {
                    "count": 20,
                    "detected": 20,
                    "speaker_stop_latency_ms": {
                        "count": 20,
                        "p50": 100.0,
                        "p95": 180.0,
                        "p99": 220.0,
                    },
                },
                "assistant_response": {
                    "count": 20,
                    "heard": 20,
                    "speech_end_to_physical_first_sound_ms": {
                        "count": 20,
                        "p50": 700.0,
                        "p95": 900.0,
                        "p99": 1050.0,
                    },
                },
            }
        ),
    )
    broken_fast_path = tmp_path / "broken-fast-path.json"
    broken_fast_path.write_text("{not-json", encoding="utf-8")

    report = report_voice_latency.build_report(
        hardware=hardware,
        fast_path=broken_fast_path,
    )

    assert report["checks"]["required_measured_e2e_present"] is True
    assert report["checks"]["hardware_full_duplex_passed"] is True
    assert report["status"] == "insufficient_evidence"
