"""Evidence report for target-hardware full-duplex voice acceptance."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from askme.voice.diagnostics.hardware_audio_capture import EVIDENCE_KINDS

HARDWARE_REPORT_SCHEMA_VERSION = "askme.full_duplex_hardware.v2"
TRIAL_EVIDENCE_FIELDS = (
    "evidence_kind",
    "method",
    "capture",
    "reference",
    "monotonic_timestamps",
    "calibration",
    "dropped_frames",
)
MANUAL_METHODS = frozenset(
    {
        "entry",
        "manual",
        "manual_entry",
        "manual_observation",
        "manual_stopwatch",
        "stopwatch",
    }
)
PHYSICAL_STOP_CAPTURE_ROLE = "isolated_speaker_monitor"
PHYSICAL_FIRST_SOUND_CAPTURE_ROLES = frozenset(
    {"isolated_speaker_monitor", "room_acoustic_monitor"}
)

MIN_TRIALS_PER_SCENARIO = 20
MIN_OVERLAP_DETECTION_RATE = 0.95
MAX_SPEAKER_ONLY_FALSE_TRIGGER_RATE = 0.0
MAX_SPEAKER_STOP_P95_MS = 250.0
MAX_SPEAKER_STOP_P99_MS = 400.0
MAX_PHYSICAL_FIRST_SOUND_P95_MS = 1_200.0
MAX_PHYSICAL_FIRST_SOUND_P99_MS = 1_800.0
MAX_RUNTIME_STATUS_AGE_SECONDS = 15.0
MAX_RUNTIME_STATUS_FUTURE_SKEW_SECONDS = 5.0
EXPECTED_NATIVE_AEC_BACKENDS = frozenset({"webrtc-apm-v2.1"})
REQUIRED_METADATA_FIELDS = (
    "operating_system",
    "python_version",
    "room",
    "audio_device",
    "audio_driver",
    "input_device_id",
    "output_device_id",
    "input_sample_rate_hz",
    "output_sample_rate_hz",
    "aec_backend",
)


def evaluate_hardware_run(
    *,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    speaker_only_trials: Sequence[Mapping[str, Any]],
    overlap_trials: Sequence[Mapping[str, Any]],
    response_trials: Sequence[Mapping[str, Any]] = (),
    require_response_trials: bool = False,
) -> dict[str, Any]:
    """Build a fail-closed acoustic acceptance report.

    Every trial must include a fresh ``runtime_status`` snapshot. Hardware and
    operating-system echo control additionally require the exact deployment
    profile to opt in with ``echo_control_verified: true``. Native AEC proves
    readiness through its active runtime status instead of that config flag.
    """

    speaker_trials = [dict(trial) for trial in speaker_only_trials]
    human_trials = [dict(trial) for trial in overlap_trials]
    assistant_trials = [dict(trial) for trial in response_trials]
    runtime_statuses = [
        trial.get("runtime_status") for trial in [*speaker_trials, *human_trials, *assistant_trials]
    ]
    runtime_failures = _runtime_failures(runtime_statuses)
    echo_evidence = _echo_control_evidence(config, runtime_statuses)

    trial_schema_failures = _trial_schema_failures(
        {
            "speaker_only": speaker_trials,
            "human_overlap": human_trials,
            "assistant_response": assistant_trials,
        }
    )

    speaker_passed = sum(1 for trial in speaker_trials if trial.get("false_barge_in") is False)
    false_triggers = sum(1 for trial in speaker_trials if trial.get("false_barge_in") is True)
    speaker_count = len(speaker_trials)
    speaker_results_complete = speaker_passed + false_triggers == speaker_count
    speaker_pass_rate = speaker_passed / speaker_count if speaker_count else 0.0
    false_trigger_rate = false_triggers / speaker_count if speaker_count else 1.0

    overlap_count = len(human_trials)
    overlap_results_complete = (
        sum(1 for trial in human_trials if isinstance(trial.get("detected"), bool)) == overlap_count
    )
    overlap_evidence = _classify_latency_trials(
        human_trials,
        scenario="human_overlap",
        outcome_field="detected",
        latency_field="speaker_stop_latency_ms",
    )
    physical_overlap = overlap_evidence["physical_acoustic"]
    render_overlap = overlap_evidence["render_chain"]
    manual_overlap = overlap_evidence["manual"]
    physical_overlap_count = int(physical_overlap["count"])
    detected = int(physical_overlap["positive"])
    detection_rate = detected / physical_overlap_count if physical_overlap_count else 0.0
    latencies = list(physical_overlap["latencies"])
    latency_summary = _latency_summary(latencies)
    render_latency_summary = _latency_summary(render_overlap["latencies"])
    manual_latency_summary = _latency_summary(manual_overlap["latencies"])
    p95_latency = latency_summary["p95"]
    p99_latency = latency_summary["p99"]

    response_count = len(assistant_trials)
    response_results_complete = (
        sum(1 for trial in assistant_trials if isinstance(trial.get("heard"), bool))
        == response_count
    )
    response_evidence = _classify_latency_trials(
        assistant_trials,
        scenario="assistant_response",
        outcome_field="heard",
        latency_field="speech_end_to_first_sound_ms",
    )
    physical_response = response_evidence["physical_acoustic"]
    render_response = response_evidence["render_chain"]
    manual_response = response_evidence["manual"]
    physical_response_count = int(physical_response["count"])
    response_heard = int(physical_response["positive"])
    response_latencies = list(physical_response["latencies"])
    response_latency_summary = _latency_summary(response_latencies)
    render_response_latency_summary = _latency_summary(render_response["latencies"])
    manual_response_latency_summary = _latency_summary(manual_response["latencies"])
    response_p95 = response_latency_summary["p95"]
    response_p99 = response_latency_summary["p99"]
    metadata_payload = dict(metadata)
    metadata_missing = [
        field
        for field in REQUIRED_METADATA_FIELDS
        if not _metadata_value_present(field, metadata_payload)
    ]
    provenance_failures = [
        *overlap_evidence["failures"],
        *response_evidence["failures"],
    ]
    evidence_failures = [*trial_schema_failures, *provenance_failures]
    invalid_physical_stop = any(
        failure.get("scenario") == "human_overlap"
        and failure.get("evidence_kind") == "physical_acoustic"
        for failure in provenance_failures
    )
    invalid_physical_first_sound = any(
        failure.get("scenario") == "assistant_response"
        and failure.get("evidence_kind") == "physical_acoustic"
        for failure in provenance_failures
    )

    checks = {
        "trial_evidence_schema_v2": not trial_schema_failures,
        "instrumented_evidence_provenance": not provenance_failures,
        "hardware_metadata_complete": not metadata_missing,
        "speaker_only_sample_count": speaker_count >= MIN_TRIALS_PER_SCENARIO,
        "speaker_only_results_complete": speaker_results_complete,
        "speaker_only_no_false_barge_in": (
            speaker_count >= MIN_TRIALS_PER_SCENARIO
            and false_trigger_rate <= MAX_SPEAKER_ONLY_FALSE_TRIGGER_RATE
        ),
        "human_overlap_sample_count": overlap_count >= MIN_TRIALS_PER_SCENARIO,
        "human_overlap_results_complete": overlap_results_complete,
        "physical_speaker_stop_sample_count": (physical_overlap_count >= MIN_TRIALS_PER_SCENARIO),
        "physical_speaker_stop_provenance": (
            physical_overlap_count > 0 and not invalid_physical_stop
        ),
        "human_overlap_detection_rate": (
            physical_overlap_count >= MIN_TRIALS_PER_SCENARIO
            and detection_rate >= MIN_OVERLAP_DETECTION_RATE
        ),
        "human_overlap_latency_complete": bool(detected) and len(latencies) == detected,
        "speaker_stop_latency_p95": (
            p95_latency is not None and p95_latency <= MAX_SPEAKER_STOP_P95_MS
        ),
        "speaker_stop_latency_p99": (
            p99_latency is not None and p99_latency <= MAX_SPEAKER_STOP_P99_MS
        ),
        "echo_control_proven": bool(echo_evidence["proven"]),
        "runtime_remained_full_duplex": not runtime_failures,
    }
    # Kept in the signature for compatibility with existing callers.  Schema
    # v2 always requires physical first-sound evidence for a passing report.
    _ = require_response_trials
    checks.update(
        {
            "assistant_response_sample_count": (response_count >= MIN_TRIALS_PER_SCENARIO),
            "assistant_response_results_complete": response_results_complete,
            "physical_first_sound_sample_count": (
                physical_response_count >= MIN_TRIALS_PER_SCENARIO
            ),
            "physical_first_sound_provenance": (
                physical_response_count > 0 and not invalid_physical_first_sound
            ),
            "assistant_response_all_heard": (
                physical_response_count >= MIN_TRIALS_PER_SCENARIO
                and response_heard == physical_response_count
            ),
            "assistant_response_latency_complete": (
                bool(response_heard) and len(response_latencies) == response_heard
            ),
            "physical_first_sound_latency_p95": (
                response_p95 is not None and response_p95 <= MAX_PHYSICAL_FIRST_SOUND_P95_MS
            ),
            "physical_first_sound_latency_p99": (
                response_p99 is not None and response_p99 <= MAX_PHYSICAL_FIRST_SOUND_P99_MS
            ),
        }
    )
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": HARDWARE_REPORT_SCHEMA_VERSION,
        "target": "askme-full-duplex-target-hardware",
        "status": "passed" if not failed_checks else "failed",
        "generated_at": datetime.now(UTC).isoformat(),
        "metadata": metadata_payload,
        "metadata_missing": metadata_missing,
        "thresholds": {
            "minimum_trials_per_scenario": MIN_TRIALS_PER_SCENARIO,
            "maximum_speaker_only_false_trigger_rate": (MAX_SPEAKER_ONLY_FALSE_TRIGGER_RATE),
            "minimum_human_overlap_detection_rate": MIN_OVERLAP_DETECTION_RATE,
            "speaker_stop_latency_p95_ms": MAX_SPEAKER_STOP_P95_MS,
            "speaker_stop_latency_p99_ms": MAX_SPEAKER_STOP_P99_MS,
            "physical_first_sound_latency_p95_ms": MAX_PHYSICAL_FIRST_SOUND_P95_MS,
            "physical_first_sound_latency_p99_ms": MAX_PHYSICAL_FIRST_SOUND_P99_MS,
            "maximum_runtime_status_age_seconds": MAX_RUNTIME_STATUS_AGE_SECONDS,
            "maximum_runtime_status_future_skew_seconds": (MAX_RUNTIME_STATUS_FUTURE_SKEW_SECONDS),
        },
        "echo_control_evidence": echo_evidence,
        "runtime_failures": runtime_failures,
        "evidence_failures": evidence_failures,
        "checks": checks,
        "failed_checks": failed_checks,
        "summary": {
            "speaker_only": {
                "count": speaker_count,
                "passed": speaker_passed,
                "false_barge_ins": false_triggers,
                "pass_rate": round(speaker_pass_rate, 4),
                "false_trigger_rate": round(false_trigger_rate, 4),
            },
            "human_overlap": {
                "count": overlap_count,
                "detected": detected,
                "detection_rate": round(detection_rate, 4),
                "speaker_stop_latency_ms": latency_summary,
                "physical_speaker_stop_latency_ms": latency_summary,
                "render_chain_speaker_stop_latency_ms": render_latency_summary,
                "manual_speaker_stop_latency_ms": manual_latency_summary,
                "speaker_stop_by_evidence_kind": _evidence_breakdown(
                    overlap_evidence,
                    latency_key="speaker_stop_latency_ms",
                ),
            },
            "assistant_response": {
                "count": response_count,
                "heard": response_heard,
                "speech_end_to_physical_first_sound_ms": response_latency_summary,
                "speech_end_to_render_chain_first_sound_ms": (render_response_latency_summary),
                "speech_end_to_manual_first_sound_ms": (manual_response_latency_summary),
                "first_sound_by_evidence_kind": _evidence_breakdown(
                    response_evidence,
                    latency_key="speech_end_to_first_sound_ms",
                ),
            },
        },
        "trials": {
            "speaker_only": speaker_trials,
            "human_overlap": human_trials,
            "assistant_response": assistant_trials,
        },
        "limitations": [
            "This report is valid only for the recorded device, driver, room, and profile.",
            "Manual entry and stopwatch trials are diagnostic only and never product-grade instrumented evidence.",
            "Render-chain timing is reported separately and cannot prove physical acoustic onset or stop.",
            "A shared room microphone cannot isolate speaker stop while overlapping human speech is present.",
            "Unit tests and loopback devices do not prove acoustic readiness.",
        ],
    }


def trial_evidence_schema_error(trial: Mapping[str, Any]) -> str | None:
    """Return the first schema-v2 trial-envelope error, if any."""

    missing = [field for field in TRIAL_EVIDENCE_FIELDS if field not in trial]
    if missing:
        return "missing_fields:" + ",".join(missing)
    evidence_kind = str(trial.get("evidence_kind") or "").strip().lower()
    if evidence_kind not in EVIDENCE_KINDS:
        return "evidence_kind_invalid"
    method = str(trial.get("method") or "").strip().lower()
    if not method:
        return "method_missing"
    if evidence_kind == "manual" and method not in MANUAL_METHODS:
        return "manual_method_invalid"
    if evidence_kind != "manual" and method in MANUAL_METHODS:
        return "instrumented_method_is_manual"
    for field in ("capture", "reference", "monotonic_timestamps", "calibration"):
        if not isinstance(trial.get(field), Mapping):
            return f"{field}_missing"
    dropped_frames = trial.get("dropped_frames")
    if evidence_kind == "manual":
        if dropped_frames is not None:
            return "manual_dropped_frames_must_be_unknown"
    elif (
        isinstance(dropped_frames, bool)
        or not isinstance(dropped_frames, int)
        or dropped_frames < 0
    ):
        return "dropped_frames_invalid"
    return None


def physical_acoustic_provenance_error(
    trial: Mapping[str, Any],
    *,
    scenario: str,
) -> str | None:
    """Reject physical gate evidence unless its capture provenance is complete.

    ``human_overlap`` deliberately requires an isolated speaker monitor.  A
    normal room microphone observes a mixture of the human and the speaker and
    therefore cannot prove when the speaker stopped during overlap.
    """

    return _instrumented_provenance_error(
        trial,
        scenario=scenario,
        evidence_kind="physical_acoustic",
    )


def render_chain_provenance_error(
    trial: Mapping[str, Any],
    *,
    scenario: str,
) -> str | None:
    """Validate render-chain timing while keeping it outside physical gates."""

    return _instrumented_provenance_error(
        trial,
        scenario=scenario,
        evidence_kind="render_chain",
    )


def _instrumented_provenance_error(
    trial: Mapping[str, Any],
    *,
    scenario: str,
    evidence_kind: str,
) -> str | None:
    schema_error = trial_evidence_schema_error(trial)
    if schema_error is not None:
        return schema_error
    if trial.get("evidence_kind") != evidence_kind:
        return f"evidence_kind_not_{evidence_kind}"
    if trial.get("dropped_frames") != 0:
        return "dropped_frames_nonzero"

    capture = trial["capture"]
    reference = trial["reference"]
    timestamps = trial["monotonic_timestamps"]
    calibration = trial["calibration"]
    assert isinstance(capture, Mapping)
    assert isinstance(reference, Mapping)
    assert isinstance(timestamps, Mapping)
    assert isinstance(calibration, Mapping)

    if capture.get("instrumented") is not True:
        return "capture_not_instrumented"
    if reference.get("instrumented") is not True:
        return "reference_not_instrumented"
    capture_kind = capture.get("source_evidence_kind", capture.get("evidence_kind"))
    if capture_kind != evidence_kind:
        return "capture_evidence_kind_mismatch"
    calibration_kind = calibration.get("source_evidence_kind", calibration.get("evidence_kind"))
    if calibration_kind != evidence_kind:
        return "calibration_evidence_kind_mismatch"
    if calibration.get("performed") is not True:
        return "calibration_not_performed"
    for field in ("device_id", "stream_id", "channel", "clock_id", "source_label", "role"):
        if not _value_present(capture.get(field)):
            return f"capture_{field}_missing"
    for field in ("device_id", "stream_id", "channel", "clock_id", "event"):
        if not _value_present(reference.get(field)):
            return f"reference_{field}_missing"
    if calibration.get("source_label") != capture.get("source_label"):
        return "calibration_capture_source_mismatch"
    if not _is_finite_positive(calibration.get("sample_rate_hz")):
        return "calibration_sample_rate_invalid"
    if not _is_positive_int(calibration.get("valid_frame_count")):
        return "calibration_frame_count_invalid"
    if not _is_finite_positive(calibration.get("threshold")):
        return "calibration_threshold_invalid"

    clock_id = str(timestamps.get("clock_id") or "").strip()
    if not clock_id:
        return "monotonic_clock_id_missing"
    if clock_id != str(capture.get("clock_id")) or clock_id != str(reference.get("clock_id")):
        return "monotonic_clock_mismatch"
    reference_s = _finite_number(timestamps.get("reference_s"))
    if reference_s is None:
        return "reference_timestamp_invalid"

    if scenario == "human_overlap":
        outcome_field = "detected"
        latency_field = "speaker_stop_latency_ms"
        expected_reference_event = "human_speech_onset"
        if evidence_kind == "physical_acoustic":
            if capture.get("role") != PHYSICAL_STOP_CAPTURE_ROLE:
                return "speaker_stop_requires_isolated_monitor"
            if capture.get("isolated_from_reference") is not True:
                return "speaker_stop_capture_not_isolated"
            capture_identity = (
                str(capture.get("device_id")),
                str(capture.get("stream_id")),
                str(capture.get("channel")),
            )
            reference_identity = (
                str(reference.get("device_id")),
                str(reference.get("stream_id")),
                str(reference.get("channel")),
            )
            if capture_identity == reference_identity:
                return "speaker_stop_reference_uses_same_capture_channel"
        elif capture.get("role") not in {"render_loopback", "render_monitor"}:
            return "render_chain_capture_role_invalid"
    elif scenario == "assistant_response":
        outcome_field = "heard"
        latency_field = "speech_end_to_first_sound_ms"
        expected_reference_event = "speech_end"
        if evidence_kind == "physical_acoustic":
            if capture.get("role") not in PHYSICAL_FIRST_SOUND_CAPTURE_ROLES:
                return "first_sound_capture_role_invalid"
        elif capture.get("role") not in {"render_loopback", "render_monitor"}:
            return "render_chain_capture_role_invalid"
    else:
        return "scenario_invalid"

    if reference.get("event") != expected_reference_event:
        return "reference_event_mismatch"
    outcome = trial.get(outcome_field)
    if not isinstance(outcome, bool):
        return f"{outcome_field}_missing"
    event_s = _finite_number(timestamps.get("event_s"))
    latency_ms = _finite_number(trial.get(latency_field))
    if not outcome:
        if event_s is not None or latency_ms is not None:
            return "negative_outcome_has_latency"
        return None
    if event_s is None or event_s < reference_s:
        return "event_timestamp_invalid"
    if latency_ms is None or latency_ms < 0.0:
        return f"{latency_field}_invalid"
    derived_latency_ms = (event_s - reference_s) * 1000.0
    if abs(derived_latency_ms - latency_ms) > 1.0:
        return "latency_timestamp_mismatch"
    return None


def _trial_schema_failures(
    scenarios: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for scenario, trials in scenarios.items():
        for index, trial in enumerate(trials, start=1):
            reason = trial_evidence_schema_error(trial)
            if reason is not None:
                failures.append(
                    {
                        "scenario": scenario,
                        "trial": index,
                        "evidence_kind": trial.get("evidence_kind"),
                        "reason": reason,
                    }
                )
    return failures


def _classify_latency_trials(
    trials: Sequence[Mapping[str, Any]],
    *,
    scenario: str,
    outcome_field: str,
    latency_field: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        kind: {"count": 0, "positive": 0, "latencies": []} for kind in EVIDENCE_KINDS
    }
    failures: list[dict[str, Any]] = []
    for index, trial in enumerate(trials, start=1):
        if trial_evidence_schema_error(trial) is not None:
            continue
        kind = str(trial["evidence_kind"])
        provenance_error: str | None = None
        if kind == "physical_acoustic":
            provenance_error = physical_acoustic_provenance_error(
                trial,
                scenario=scenario,
            )
        elif kind == "render_chain":
            provenance_error = render_chain_provenance_error(
                trial,
                scenario=scenario,
            )
        if provenance_error is not None:
            failures.append(
                {
                    "scenario": scenario,
                    "trial": index,
                    "evidence_kind": kind,
                    "reason": provenance_error,
                }
            )
            continue
        bucket = result[kind]
        bucket["count"] += 1
        if trial.get(outcome_field) is True:
            bucket["positive"] += 1
            if _is_finite_nonnegative(trial.get(latency_field)):
                bucket["latencies"].append(float(trial[latency_field]))
    result["failures"] = failures
    return result


def _evidence_breakdown(
    evidence: Mapping[str, Any],
    *,
    latency_key: str,
) -> dict[str, Any]:
    return {
        kind: {
            "count": int(evidence[kind]["count"]),
            "positive": int(evidence[kind]["positive"]),
            latency_key: _latency_summary(evidence[kind]["latencies"]),
        }
        for kind in sorted(EVIDENCE_KINDS)
    }


def write_hardware_report(report: Mapping[str, Any], path: Path) -> None:
    """Write a report without exposing a partially written target file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def preflight_hardware_run(
    *,
    config: Mapping[str, Any],
    runtime_status: Mapping[str, Any],
) -> dict[str, Any]:
    """Check that operator trials can start without bypassing fail-closed gates."""

    runtime_ready, runtime_reason = runtime_readiness(runtime_status)
    evidence = _echo_control_evidence(config, [runtime_status])
    errors: list[str] = []
    if not runtime_ready:
        errors.append(runtime_reason)
    if not evidence["proven"]:
        errors.append("echo_control_unproven")
    return {
        "status": "ready" if not errors else "failed",
        "runtime_ready": runtime_ready,
        "runtime_reason": runtime_reason,
        "echo_control_proven": bool(evidence["proven"]),
        "echo_control_evidence": evidence,
        "errors": errors,
    }


def runtime_readiness(snapshot: Mapping[str, Any]) -> tuple[bool, str]:
    """Return whether a runtime snapshot still proves active full duplex."""

    top_status = snapshot.get("status")
    if top_status != "ok":
        if top_status is None or not str(top_status).strip():
            return False, "runtime_status_missing"
        return False, f"runtime_status_{str(top_status).strip().lower()}"

    now = datetime.now(UTC)
    timestamp_reason = _timestamp_failure_reason(
        snapshot.get("snapshot_at"),
        field="snapshot_at",
        now=now,
    )
    if timestamp_reason is not None:
        return False, timestamp_reason

    voice = snapshot.get("voice_pipeline_status")
    if not isinstance(voice, Mapping):
        return False, "voice_pipeline_status_missing"
    if voice.get("pipeline_ok") is not True:
        if "pipeline_ok" not in voice:
            return False, "voice_pipeline_health_missing"
        return False, "voice_pipeline_degraded"

    timestamp_reason = _timestamp_failure_reason(
        voice.get("recorded_at"),
        field="voice_recorded_at",
        now=now,
    )
    if timestamp_reason is not None:
        return False, timestamp_reason

    media = voice.get("media", voice)
    if not isinstance(media, Mapping):
        return False, "media_status_missing"
    full_duplex = media.get("full_duplex", media)
    if not isinstance(full_duplex, Mapping):
        return False, "full_duplex_status_missing"
    if full_duplex.get("enabled") is not True:
        reason = str(full_duplex.get("reason", "disabled") or "disabled")
        return False, f"full_duplex_{reason}"
    if str(full_duplex.get("echo_control", "none") or "none") == "none":
        return False, "echo_control_none"
    return True, "ready"


def _runtime_failures(statuses: Sequence[Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for index, status in enumerate(statuses, start=1):
        if not isinstance(status, Mapping):
            failures.append({"snapshot": index, "reason": "runtime_status_missing"})
            continue
        ready, reason = runtime_readiness(status)
        if not ready:
            failures.append({"snapshot": index, "reason": reason})
    return failures


def _echo_control_evidence(
    config: Mapping[str, Any],
    statuses: Sequence[Any],
) -> dict[str, Any]:
    voice = config.get("voice", {})
    voice_cfg = voice if isinstance(voice, Mapping) else {}
    full_duplex = voice_cfg.get("full_duplex", {})
    cfg = full_duplex if isinstance(full_duplex, Mapping) else {}
    configured_mode = str(cfg.get("echo_control", "auto") or "auto").lower()
    verified_flag = cfg.get("echo_control_verified") is True
    runtime_modes: set[str] = set()
    runtime_reasons: set[str] = set()
    runtime_backends: set[str] = set()
    full_status_count = 0
    for status in statuses:
        full_status = _full_duplex_status(status)
        if full_status is None:
            continue
        full_status_count += 1
        runtime_modes.add(str(full_status.get("echo_control", "none") or "none"))
        runtime_reasons.add(str(full_status.get("reason", "unknown") or "unknown"))
        runtime_backends.add(str(full_status.get("aec_backend", "unknown") or "unknown"))

    one_runtime_mode = len(runtime_modes) == 1
    all_statuses_complete = full_status_count == len(statuses) and bool(statuses)
    runtime_mode = next(iter(runtime_modes), "none")
    if runtime_mode in {"hardware", "system"}:
        proven = (
            bool(cfg.get("enabled"))
            and configured_mode == runtime_mode
            and verified_flag
            and runtime_reasons == {"verified_echo_control"}
            and runtime_backends == {runtime_mode}
        )
        proof = (
            "verified_deployment_profile"
            if proven
            else "verification_flag_or_runtime_backend_mismatch"
        )
    elif runtime_mode == "native":
        proven = (
            bool(cfg.get("enabled"))
            and configured_mode in {"auto", "native"}
            and runtime_reasons == {"native_aec_ready"}
            and runtime_backends == EXPECTED_NATIVE_AEC_BACKENDS
        )
        proof = "active_native_aec" if proven else "native_aec_not_active"
    else:
        proven = False
        proof = "runtime_echo_control_missing_or_inconsistent"
    proven = bool(proven and one_runtime_mode and all_statuses_complete)
    return {
        "proven": proven,
        "proof": proof,
        "configured_mode": configured_mode,
        "echo_control_verified": verified_flag,
        "runtime_modes": sorted(runtime_modes),
        "runtime_reasons": sorted(runtime_reasons),
        "runtime_backends": sorted(runtime_backends),
    }


def _full_duplex_status(status: Any) -> Mapping[str, Any] | None:
    if not isinstance(status, Mapping):
        return None
    voice = status.get("voice_pipeline_status", status)
    if not isinstance(voice, Mapping):
        return None
    media = voice.get("media", voice)
    if not isinstance(media, Mapping):
        return None
    full_duplex = media.get("full_duplex", media)
    return full_duplex if isinstance(full_duplex, Mapping) else None


def _latency_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(ordered),
        "p50": round(_percentile(ordered, 0.50), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
        "p99": round(_percentile(ordered, 0.99), 3),
        "max": round(ordered[-1], 3),
    }


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower]) * (1.0 - weight) + float(ordered[upper]) * weight


def _is_finite_nonnegative(value: Any) -> bool:
    number = _finite_number(value)
    return number is not None and number >= 0.0


def _is_finite_positive(value: Any) -> bool:
    number = _finite_number(value)
    return number is not None and number > 0.0


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_positive_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _value_present(value: Any) -> bool:
    return value is not None and bool(str(value).strip())


def _timestamp_failure_reason(
    value: Any,
    *,
    field: str,
    now: datetime,
) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return f"{field}_missing"
    timestamp = _parse_health_timestamp(value)
    if timestamp is None:
        return f"{field}_invalid"
    age_seconds = (now - timestamp).total_seconds()
    if age_seconds > MAX_RUNTIME_STATUS_AGE_SECONDS:
        return f"{field}_stale"
    if age_seconds < -MAX_RUNTIME_STATUS_FUTURE_SKEW_SECONDS:
        return f"{field}_future"
    return None


def _parse_health_timestamp(value: Any) -> datetime | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _datetime_from_epoch(float(value))
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            return _datetime_from_epoch(float(text))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _datetime_from_epoch(value: float) -> datetime | None:
    if not math.isfinite(value):
        return None
    try:
        return datetime.fromtimestamp(value, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _metadata_value_present(field: str, metadata: Mapping[str, Any]) -> bool:
    value = metadata.get(field)
    if field in {"input_sample_rate_hz", "output_sample_rate_hz"}:
        if value is None:
            return False
        try:
            sample_rate = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(sample_rate) and sample_rate > 0.0
    return value is not None and bool(str(value).strip())
