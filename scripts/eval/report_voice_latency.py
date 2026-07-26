"""Normalize voice latency evidence into a product-readiness report.

This command does not run hardware or online measurements.  It records what is
actually proven by existing reports and refuses to mark product readiness as
passed when required evidence is only projected or simulated.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from askme.voice.diagnostics.full_duplex_hardware import (
    HARDWARE_REPORT_SCHEMA_VERSION,
    physical_acoustic_provenance_error,
    render_chain_provenance_error,
    trial_evidence_schema_error,
)

SCHEMA_VERSION = "askme.voice_latency_report.v1"
EVIDENCE_TYPES = frozenset({"manual", "measured", "projected", "simulated"})
REQUIRED_MEASURED_METRICS = (
    "barge_in_to_physical_speaker_stop_ms",
    "speech_end_to_physical_first_semantic_audio_ms",
)
MIN_P95_SAMPLES = 100
MIN_P99_SAMPLES = 300
# Compatibility name for callers importing the old constant.
MIN_REQUIRED_SAMPLES = MIN_P95_SAMPLES
EXPERIMENT_SCHEMA_VERSION = "askme.voice_latency_experiment.v1"
HARDWARE_REPORT_TARGET = "askme-full-duplex-target-hardware"
HARDWARE_REQUIRED_CHECKS = frozenset(
    {
        "hardware_metadata_complete",
        "trial_evidence_schema_v2",
        "instrumented_evidence_provenance",
        "speaker_only_sample_count",
        "speaker_only_results_complete",
        "speaker_only_no_false_barge_in",
        "human_overlap_sample_count",
        "human_overlap_results_complete",
        "physical_speaker_stop_sample_count",
        "physical_speaker_stop_provenance",
        "human_overlap_detection_rate",
        "human_overlap_latency_complete",
        "speaker_stop_latency_p95",
        "speaker_stop_latency_p99",
        "echo_control_proven",
        "runtime_remained_full_duplex",
        "assistant_response_sample_count",
        "assistant_response_results_complete",
        "physical_first_sound_sample_count",
        "physical_first_sound_provenance",
        "assistant_response_all_heard",
        "assistant_response_latency_complete",
        "physical_first_sound_latency_p95",
        "physical_first_sound_latency_p99",
    }
)
HARDWARE_REQUIRED_METADATA = frozenset(
    {
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
    }
)
STAGE_METRIC_FIELDS: dict[str, dict[str, tuple[str, ...]]] = {
    "asr": {
        "asr_endpoint_ms": ("endpoint_ms", "asr_endpoint_ms"),
    },
    "llm": {
        "llm_first_content_ms": ("first_content_ms", "llm_first_content_ms"),
        "llm_first_semantic_clause_ms": (
            "first_semantic_clause_ms",
            "llm_first_semantic_clause_ms",
        ),
    },
    "tts": {
        "tts_provider_first_pcm_ms": (
            "provider_first_pcm_ms",
            "tts_provider_first_pcm_ms",
        ),
        "tts_buffer_commit_ms": ("buffer_commit_ms", "tts_buffer_commit_ms"),
        "tts_physical_first_nonzero_ms": (
            "physical_first_nonzero_ms",
            "tts_physical_first_nonzero_ms",
        ),
    },
    "barge_in": {
        "barge_in_physical_stop_ms": (
            "physical_stop_ms",
            "barge_in_physical_stop_ms",
        ),
    },
}


def build_report(
    *,
    fast_path: Path | None = None,
    hardware: Path | None = None,
    online_smoke: Path | None = None,
    voice_health: Path | None = None,
    scenario: Path | None = None,
    experiments: Sequence[Path] | None = None,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    checks = {
        "no_simulated_metric_used_for_pass": True,
        "required_measured_e2e_present": False,
        "hardware_full_duplex_passed": False,
        "online_provider_smoke_passed": False,
        "all_supplied_experiments_decision_grade": True,
        "tts_provider_comparison_ready": True,
    }
    limitations: list[str] = []

    _merge_source(
        sources,
        metrics,
        source_id="fast_path",
        kind="fast_path_benchmark",
        path=fast_path,
        optional=True,
        loader=_load_fast_path,
    )
    hardware_status = _merge_source(
        sources,
        metrics,
        source_id="full_duplex_hardware",
        kind="full_duplex_hardware",
        path=hardware,
        optional=False,
        loader=_load_hardware,
    )
    checks["hardware_full_duplex_passed"] = hardware_status == "passed"
    online_status = _merge_source(
        sources,
        metrics,
        source_id="online_smoke",
        kind="online_provider_smoke",
        path=online_smoke,
        optional=True,
        loader=_load_online_smoke,
    )
    checks["online_provider_smoke_passed"] = online_status == "passed"
    _merge_source(
        sources,
        metrics,
        source_id="voice_health",
        kind="voice_health_snapshot",
        path=voice_health,
        optional=True,
        loader=_load_voice_health,
    )
    _merge_source(
        sources,
        metrics,
        source_id="scenario",
        kind="offline_scenario_simulation",
        path=scenario,
        optional=True,
        loader=_load_scenario,
    )

    experiment_records, stage_metrics, experiment_sources_valid = _load_experiment_sources(
        sources, experiments or ()
    )
    experiments_requested = bool(experiments)
    all_experiments_decision_grade = bool(experiment_records) and all(
        experiment["status"] == "sufficient_evidence" for experiment in experiment_records
    )
    if not experiments_requested:
        all_experiments_decision_grade = True
    elif not experiment_sources_valid:
        all_experiments_decision_grade = False
    checks["all_supplied_experiments_decision_grade"] = all_experiments_decision_grade

    tts_decision = _build_tts_provider_decision(experiment_records)
    tts_comparison_required = tts_decision["status"] != "not_requested"
    checks["tts_provider_comparison_ready"] = (
        not tts_comparison_required or tts_decision["status"] == "decision_ready"
    )

    _append_legacy_stage_metrics(stage_metrics, metrics)

    for name, metric in metrics.items():
        _validate_metric(name, metric)

    missing_required = [
        name
        for name in REQUIRED_MEASURED_METRICS
        if not _required_metric_is_measured(metrics.get(name))
    ]
    checks["required_measured_e2e_present"] = not missing_required
    failed_required = [
        name
        for name in REQUIRED_MEASURED_METRICS
        if (required_metric := metrics.get(name)) is not None
        and required_metric.get("status") == "failed"
    ]
    required_source_failures = [
        source
        for source in sources
        if source.get("status") == "failed" and not source.get("optional", False)
    ]
    required_source_errors = [
        source
        for source in sources
        if source.get("status") not in {"passed", "failed"} and not source.get("optional", False)
    ]
    supplied_source_errors = [
        source
        for source in sources
        if source.get("path") is not None and source.get("status") in {"missing", "invalid"}
    ]
    if failed_required or required_source_failures:
        status = "failed"
    elif (
        missing_required
        or required_source_errors
        or supplied_source_errors
        or not all_experiments_decision_grade
        or not checks["tts_provider_comparison_ready"]
    ):
        status = "insufficient_evidence"
    else:
        status = "passed"

    counts = _evidence_counts(metrics)
    experiment_counts = _experiment_evidence_counts(stage_metrics)
    if any(metric["evidence_type"] == "projected" for metric in metrics.values()):
        limitations.append(
            "Projected latency is useful for optimization only, not product acceptance."
        )
    if any(metric["evidence_type"] == "simulated" for metric in metrics.values()):
        limitations.append(
            "Simulated scenario latency cannot prove real user-perceived responsiveness."
        )
    if any(metric["evidence_type"] == "manual" for metric in metrics.values()):
        limitations.append(
            "Manual entry and stopwatch timing are diagnostic, not product-grade instrumented evidence."
        )
    if any(experiment["evidence_type"] == "projected" for experiment in experiment_records):
        limitations.append("Projected stage experiments are not decision-grade measured evidence.")
    if any(experiment["evidence_type"] == "simulated" for experiment in experiment_records):
        limitations.append("Simulated stage experiments cannot select a production provider.")

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "profile": dict(
            profile
            or {
                "robot": "local_speakerphone",
                "mode": "hybrid",
                "config_path": "config.board.yaml",
            }
        ),
        "evidence_summary": {
            "manual_metrics": counts["manual"] + experiment_counts["manual"],
            "measured_metrics": counts["measured"] + experiment_counts["measured"],
            "projected_metrics": (counts["projected"] + experiment_counts["projected"]),
            "simulated_metrics": (counts["simulated"] + experiment_counts["simulated"]),
            "experiment_metrics": experiment_counts,
            "missing_required_metrics": missing_required,
        },
        "sources": sources,
        "metrics": metrics,
        "experiments": experiment_records,
        "stage_metrics": stage_metrics,
        "provider_decisions": {"tts": tts_decision},
        "checks": checks,
        "limitations": limitations,
    }


def _load_experiment_sources(
    sources: list[dict[str, Any]],
    paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], bool]:
    experiments: list[dict[str, Any]] = []
    stage_metrics: dict[str, list[dict[str, Any]]] = {}
    all_sources_valid = True
    for path_index, path in enumerate(paths, start=1):
        source_id = f"latency_experiment_{path_index}"
        source = {
            "id": source_id,
            "kind": "voice_latency_experiment",
            "path": str(path),
            "optional": True,
        }
        if not path.exists():
            source.update(
                {
                    "status": "missing",
                    "evidence_type": None,
                }
            )
            sources.append(source)
            all_sources_valid = False
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_experiments = _experiment_payloads(payload)
            parsed = [
                _parse_latency_experiment(
                    raw,
                    source_id=source_id,
                    source_path=path,
                )
                for raw in raw_experiments
            ]
        except Exception as exc:
            source.update(
                {
                    "status": "invalid",
                    "evidence_type": None,
                    "error": str(exc),
                }
            )
            sources.append(source)
            all_sources_valid = False
            continue

        experiments.extend(parsed)
        for experiment in parsed:
            for metric_name, metric in experiment["metrics"].items():
                stage_metrics.setdefault(metric_name, []).append(dict(metric))
        evidence_types = {item["evidence_type"] for item in parsed}
        source.update(
            {
                "status": (
                    "sufficient_evidence"
                    if all(item["status"] == "sufficient_evidence" for item in parsed)
                    else "insufficient_evidence"
                ),
                "evidence_type": (
                    next(iter(evidence_types)) if len(evidence_types) == 1 else "mixed"
                ),
                "experiment_ids": [item["experiment_id"] for item in parsed],
                "generated_at": (
                    payload.get("generated_at") if isinstance(payload, Mapping) else None
                ),
            }
        )
        sources.append(source)
    return experiments, stage_metrics, all_sources_valid


def _experiment_payloads(payload: Any) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("latency experiment JSON must be an object")
    nested = payload.get("experiments")
    if nested is None:
        return [payload]
    if not isinstance(nested, list) or not nested:
        raise ValueError("experiments must be a non-empty list")
    if not all(isinstance(item, Mapping) for item in nested):
        raise ValueError("every experiments item must be an object")
    inherited_schema = payload.get("schema_version")
    return [
        {"schema_version": inherited_schema, **dict(item)} if "schema_version" not in item else item
        for item in nested
    ]


def _parse_latency_experiment(
    payload: Mapping[str, Any],
    *,
    source_id: str,
    source_path: Path,
) -> dict[str, Any]:
    schema_version = str(payload.get("schema_version") or "").strip()
    if schema_version != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported experiment schema_version {schema_version!r}; "
            f"expected {EXPERIMENT_SCHEMA_VERSION!r}"
        )
    experiment_id = _required_text(payload, "experiment_id")
    stage = _required_text(payload, "stage").lower().replace("-", "_")
    if stage not in STAGE_METRIC_FIELDS:
        raise ValueError(f"{experiment_id}: unsupported stage {stage!r}")
    provider = _required_text(payload, "provider")
    model = _required_text(payload, "model")
    transport = _required_text(payload, "transport")
    evidence_type = _required_text(payload, "evidence_type").lower()
    if evidence_type not in EVIDENCE_TYPES:
        raise ValueError(f"{experiment_id}: invalid evidence_type {evidence_type!r}")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError(f"{experiment_id}: samples must be a non-empty list")
    if not all(isinstance(sample, Mapping) for sample in raw_samples):
        raise ValueError(f"{experiment_id}: every sample must be an object")
    declared_count = _int(payload.get("sample_count"))
    if "sample_count" in payload and declared_count is None:
        raise ValueError(f"{experiment_id}: sample_count must be a non-negative integer")
    if declared_count is not None and declared_count != len(raw_samples):
        raise ValueError(
            f"{experiment_id}: sample_count={declared_count} does not match "
            f"samples={len(raw_samples)}"
        )

    corpus_id = str(payload.get("corpus_id") or "").strip() or None
    case_ids = [str(sample.get("case_id") or "").strip() for sample in raw_samples]
    metrics: dict[str, dict[str, Any]] = {}
    for metric_name, aliases in STAGE_METRIC_FIELDS[stage].items():
        values = [
            value
            for sample in raw_samples
            if (value := _sample_latency(sample, aliases)) is not None
        ]
        metrics[metric_name] = _experiment_metric(
            experiment_id=experiment_id,
            stage=stage,
            provider=provider,
            model=model,
            transport=transport,
            evidence_type=evidence_type,
            corpus_id=corpus_id,
            metric_name=metric_name,
            values=values,
            experiment_sample_count=len(raw_samples),
        )

    status = (
        "sufficient_evidence"
        if all(metric["status"] == "sufficient_evidence" for metric in metrics.values())
        else "insufficient_evidence"
    )
    limitations = list(_string_list(payload.get("limitations")))
    if evidence_type != "measured":
        limitations.append("Only measured experiments are decision-grade.")
    if len(raw_samples) < MIN_P95_SAMPLES:
        limitations.append(
            f"p95 is withheld until at least {MIN_P95_SAMPLES} valid samples exist."
        )
    if len(raw_samples) < MIN_P99_SAMPLES:
        limitations.append(
            f"p99 is withheld until at least {MIN_P99_SAMPLES} valid samples exist."
        )
    return {
        "schema_version": schema_version,
        "experiment_id": experiment_id,
        "stage": stage,
        "provider": provider,
        "model": model,
        "transport": transport,
        "sample_count": len(raw_samples),
        "evidence_type": evidence_type,
        "corpus_id": corpus_id,
        "case_ids": case_ids,
        "status": status,
        "source_id": source_id,
        "source_path": str(source_path),
        "metrics": metrics,
        "limitations": limitations,
    }


def _experiment_metric(
    *,
    experiment_id: str,
    stage: str,
    provider: str,
    model: str,
    transport: str,
    evidence_type: str,
    corpus_id: str | None,
    metric_name: str,
    values: Sequence[float],
    experiment_sample_count: int,
) -> dict[str, Any]:
    ordered = sorted(values)
    valid_count = len(ordered)
    distribution_grade = valid_count >= MIN_P95_SAMPLES
    decision_grade = evidence_type == "measured" and distribution_grade
    limitations: list[str] = []
    if valid_count != experiment_sample_count:
        limitations.append(
            f"{experiment_sample_count - valid_count} samples lack a valid {metric_name} value."
        )
    if valid_count < MIN_P95_SAMPLES:
        limitations.append(f"p95 requires at least {MIN_P95_SAMPLES} valid samples.")
    if valid_count < MIN_P99_SAMPLES:
        limitations.append(f"p99 requires at least {MIN_P99_SAMPLES} valid samples.")
    if evidence_type != "measured":
        limitations.append("Projected or simulated evidence cannot select a provider.")
    return {
        "experiment_id": experiment_id,
        "stage": stage,
        "metric": metric_name,
        "provider": provider,
        "model": model,
        "transport": transport,
        "sample_count": valid_count,
        "experiment_sample_count": experiment_sample_count,
        "evidence_type": evidence_type,
        "corpus_id": corpus_id,
        "p50_ms": _percentile(ordered, 0.50) if ordered else None,
        "p95_ms": _percentile(ordered, 0.95) if distribution_grade else None,
        "p99_ms": (
            _percentile(ordered, 0.99)
            if valid_count >= MIN_P99_SAMPLES
            else None
        ),
        "status": ("sufficient_evidence" if decision_grade else "insufficient_evidence"),
        "limitations": limitations,
    }


def _build_tts_provider_decision(
    experiments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tts_experiments = [experiment for experiment in experiments if experiment.get("stage") == "tts"]
    minimax = [
        item for item in tts_experiments if _provider_family(item.get("provider")) == "minimax"
    ]
    volc = [
        item for item in tts_experiments if _provider_family(item.get("provider")) == "volcengine"
    ]
    base: dict[str, Any] = {
        "status": "not_requested",
        "decision_scope": "latency_only",
        "decision_metric": "tts_provider_first_pcm_ms",
        "corpus_id": None,
        "winner": None,
        "candidates": [],
        "reason": "MiniMax and VolcEngine experiments were not both supplied.",
        "limitations": [
            "Latency alone does not prove voice quality, intelligibility, or provider reliability."
        ],
    }
    if not minimax or not volc:
        return base

    common_corpora = sorted(
        {str(item.get("corpus_id")) for item in minimax if item.get("corpus_id")}
        & {str(item.get("corpus_id")) for item in volc if item.get("corpus_id")}
    )
    base["status"] = "insufficient_evidence"
    if not common_corpora:
        base["reason"] = "No MiniMax and VolcEngine experiments share a corpus_id."
        return base
    if len(common_corpora) != 1:
        base["reason"] = "Multiple common corpora are ambiguous; compare one corpus per report."
        return base
    corpus_id = common_corpora[0]
    minimax_matches = [item for item in minimax if item.get("corpus_id") == corpus_id]
    volc_matches = [item for item in volc if item.get("corpus_id") == corpus_id]
    if len(minimax_matches) != 1 or len(volc_matches) != 1:
        base["reason"] = "Exactly one experiment per provider and corpus is required."
        return base
    candidates = [minimax_matches[0], volc_matches[0]]
    base["corpus_id"] = corpus_id
    if any(item.get("evidence_type") != "measured" for item in candidates):
        base["reason"] = "All provider comparison candidates must be measured evidence."
        return base
    case_sets = [set(_string_list(item.get("case_ids"))) for item in candidates]
    if (
        any(len(case_set) < MIN_P95_SAMPLES for case_set in case_sets)
        or len(case_sets[0]) != int(candidates[0].get("sample_count") or 0)
        or len(case_sets[1]) != int(candidates[1].get("sample_count") or 0)
        or case_sets[0] != case_sets[1]
    ):
        base["reason"] = (
            "Provider comparison requires the same distinct case_id set with at least "
            f"{MIN_P95_SAMPLES} cases."
        )
        return base

    candidate_summaries: list[dict[str, Any]] = []
    for item in candidates:
        metrics = _mapping(item.get("metrics"))
        metric = _mapping(metrics.get("tts_provider_first_pcm_ms"))
        p95_ms = _number(metric.get("p95_ms"))
        if metric.get("status") != "sufficient_evidence" or p95_ms is None:
            base["reason"] = (
                "Each provider needs at least "
                f"{MIN_P95_SAMPLES} measured provider-first-PCM samples."
            )
            return base
        candidate_summaries.append(
            {
                "experiment_id": item.get("experiment_id"),
                "provider": item.get("provider"),
                "model": item.get("model"),
                "transport": item.get("transport"),
                "sample_count": metric.get("sample_count"),
                "evidence_type": item.get("evidence_type"),
                "p50_ms": metric.get("p50_ms"),
                "p95_ms": p95_ms,
            }
        )
    base["candidates"] = candidate_summaries
    ordered = sorted(candidate_summaries, key=lambda item: float(item["p95_ms"]))
    if math.isclose(float(ordered[0]["p95_ms"]), float(ordered[1]["p95_ms"])):
        base["status"] = "decision_ready"
        base["reason"] = "The providers tie on provider-first-PCM p95."
        return base
    base["status"] = "decision_ready"
    base["winner"] = dict(ordered[0])
    base["reason"] = "Lowest same-corpus provider-first-PCM p95 latency."
    return base


def _append_legacy_stage_metrics(
    stage_metrics: dict[str, list[dict[str, Any]]],
    metrics: Mapping[str, Mapping[str, Any]],
) -> None:
    physical_stop = metrics.get("barge_in_to_speaker_stop_ms")
    if physical_stop is None:
        return
    sample_count = int(physical_stop.get("count") or 0)
    decision_grade = (
        physical_stop.get("evidence_type") == "measured"
        and physical_stop.get("p95_ms") is not None
        and sample_count >= MIN_P95_SAMPLES
    )
    stage_metrics.setdefault("barge_in_physical_stop_ms", []).append(
        {
            **dict(physical_stop),
            "experiment_id": None,
            "stage": "barge_in",
            "metric": "barge_in_physical_stop_ms",
            "provider": None,
            "model": None,
            "transport": "physical_speaker",
            "sample_count": sample_count,
            "p95_ms": physical_stop.get("p95_ms") if decision_grade else None,
            "p99_ms": physical_stop.get("p99_ms") if decision_grade else None,
            "status": ("sufficient_evidence" if decision_grade else "insufficient_evidence"),
        }
    )


def _merge_source(
    sources: list[dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
    *,
    source_id: str,
    kind: str,
    path: Path | None,
    optional: bool,
    loader: Any,
) -> str:
    if path is None:
        sources.append(
            {
                "id": source_id,
                "kind": kind,
                "path": None,
                "status": "not_provided",
                "optional": optional,
                "evidence_type": None,
            }
        )
        return "not_provided"
    if not path.exists():
        sources.append(
            {
                "id": source_id,
                "kind": kind,
                "path": str(path),
                "status": "missing",
                "optional": optional,
                "evidence_type": None,
            }
        )
        return "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("top-level JSON must be an object")
        source_metrics = loader(payload)
    except Exception as exc:
        sources.append(
            {
                "id": source_id,
                "kind": kind,
                "path": str(path),
                "status": "invalid",
                "optional": optional,
                "evidence_type": None,
                "error": str(exc),
            }
        )
        return "invalid"
    metrics.update(source_metrics)
    status = (
        "passed" if payload.get("status") == "passed" else str(payload.get("status") or "loaded")
    )
    source_error: str | None = None
    if kind == "full_duplex_hardware" and status == "passed":
        source_error = _hardware_contract_error(payload)
        if source_error is not None:
            status = "invalid"
    source_record = {
        "id": source_id,
        "kind": kind,
        "path": str(path),
        "status": status,
        "optional": optional,
        "evidence_type": _source_evidence_type(source_metrics),
        "generated_at": payload.get("generated_at"),
    }
    if source_error is not None:
        source_record["error"] = source_error
    sources.append(source_record)
    return status


def _hardware_contract_error(payload: Mapping[str, Any]) -> str | None:
    if payload.get("schema_version") != HARDWARE_REPORT_SCHEMA_VERSION:
        return (
            "hardware schema_version must be "
            f"{HARDWARE_REPORT_SCHEMA_VERSION!r}; legacy/manual summaries are diagnostic only"
        )
    if payload.get("target") != HARDWARE_REPORT_TARGET:
        return f"hardware target must be {HARDWARE_REPORT_TARGET!r}"
    if payload.get("failed_checks") != []:
        return "hardware failed_checks must be an empty list"
    if payload.get("runtime_failures") != []:
        return "hardware runtime_failures must be an empty list"
    if payload.get("evidence_failures") != []:
        return "hardware evidence_failures must be an empty list"
    if payload.get("metadata_missing") != []:
        return "hardware metadata_missing must be an empty list"

    metadata = _mapping(payload.get("metadata"))
    missing_metadata = sorted(
        field
        for field in HARDWARE_REQUIRED_METADATA
        if not _hardware_metadata_present(field, metadata.get(field))
    )
    if missing_metadata:
        return f"hardware metadata is incomplete: {', '.join(missing_metadata)}"

    checks = _mapping(payload.get("checks"))
    missing_or_failed_checks = sorted(
        check for check in HARDWARE_REQUIRED_CHECKS if checks.get(check) is not True
    )
    if missing_or_failed_checks:
        return "hardware acceptance checks are missing or false: " + ", ".join(
            missing_or_failed_checks
        )
    echo_evidence = _mapping(payload.get("echo_control_evidence"))
    if echo_evidence.get("proven") is not True:
        return "hardware echo_control_evidence.proven must be true"

    trial_error = _hardware_trial_contract_error(payload)
    if trial_error is not None:
        return trial_error

    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        return "hardware schema v2 requires a summary object"
    overlap = _mapping(summary.get("human_overlap"))
    response = _mapping(summary.get("assistant_response"))
    for field in (
        "physical_speaker_stop_latency_ms",
        "render_chain_speaker_stop_latency_ms",
    ):
        if not isinstance(overlap.get(field), Mapping):
            return f"hardware schema v2 summary is missing human_overlap.{field}"
    for field in (
        "speech_end_to_physical_first_sound_ms",
        "speech_end_to_render_chain_first_sound_ms",
    ):
        if not isinstance(response.get(field), Mapping):
            return f"hardware schema v2 summary is missing assistant_response.{field}"
    return None


def _hardware_trial_contract_error(payload: Mapping[str, Any]) -> str | None:
    raw_trials = payload.get("trials")
    if not isinstance(raw_trials, Mapping):
        return "hardware schema v2 requires raw trials"
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for scenario in ("speaker_only", "human_overlap", "assistant_response"):
        raw_group = raw_trials.get(scenario)
        if not isinstance(raw_group, Sequence) or isinstance(raw_group, (str, bytes)):
            return f"hardware trials.{scenario} must be a list"
        group: list[Mapping[str, Any]] = []
        for index, item in enumerate(raw_group, start=1):
            if not isinstance(item, Mapping):
                return f"hardware {scenario} trial {index} must be an object"
            schema_error = trial_evidence_schema_error(item)
            if schema_error is not None:
                return f"hardware {scenario} trial {index} invalid: {schema_error}"
            kind = item.get("evidence_kind")
            if kind == "physical_acoustic":
                provenance_error = (
                    physical_acoustic_provenance_error(
                        item,
                        scenario=scenario,
                    )
                    if scenario != "speaker_only"
                    else None
                )
            elif kind == "render_chain":
                provenance_error = (
                    render_chain_provenance_error(
                        item,
                        scenario=scenario,
                    )
                    if scenario != "speaker_only"
                    else None
                )
            else:
                provenance_error = None
            if provenance_error is not None:
                return f"hardware {scenario} trial {index} provenance invalid: {provenance_error}"
            group.append(item)
        groups[scenario] = group

    speaker_trials = groups["speaker_only"]
    if len(speaker_trials) < MIN_P95_SAMPLES:
        return (
            "hardware schema v2 requires at least "
            f"{MIN_P95_SAMPLES} speaker_only trials for p95"
        )
    if any(trial.get("false_barge_in") is not False for trial in speaker_trials):
        return "hardware speaker_only raw trials contain a false or missing outcome"

    physical_overlap = [
        trial
        for trial in groups["human_overlap"]
        if trial.get("evidence_kind") == "physical_acoustic"
    ]
    if len(physical_overlap) < MIN_P95_SAMPLES:
        return (
            "hardware schema v2 requires "
            f"{MIN_P95_SAMPLES} physical_acoustic speaker-stop trials for p95"
        )
    detected_overlap = [trial for trial in physical_overlap if trial.get("detected") is True]
    if len(detected_overlap) < math.ceil(len(physical_overlap) * 0.95):
        return "hardware physical_acoustic speaker-stop trials do not prove 95% detection"
    stop_latencies = sorted(float(trial["speaker_stop_latency_ms"]) for trial in detected_overlap)
    if _percentile(stop_latencies, 0.95) > 250.0:
        return "hardware physical_acoustic speaker-stop p95 exceeds 250ms"
    if (
        len(stop_latencies) >= MIN_P99_SAMPLES
        and _percentile(stop_latencies, 0.99) > 400.0
    ):
        return "hardware physical_acoustic speaker-stop p99 exceeds 400ms"

    physical_semantic_response = [
        trial
        for trial in groups["assistant_response"]
        if trial.get("evidence_kind") == "physical_acoustic"
        and trial.get("audio_class") == "semantic"
    ]
    if len(physical_semantic_response) < MIN_P95_SAMPLES:
        return (
            "hardware schema v2 requires "
            f"{MIN_P95_SAMPLES} physical_acoustic semantic-audio trials for p95"
        )
    if any(trial.get("heard") is not True for trial in physical_semantic_response):
        return "hardware physical semantic-audio trials are not fully heard"
    if any(
        _number(trial.get("speech_end_to_first_semantic_audio_ms")) is None
        for trial in physical_semantic_response
    ):
        return "hardware semantic-audio trials are missing semantic latency"
    response_latencies = sorted(
        float(trial["speech_end_to_first_semantic_audio_ms"])
        for trial in physical_semantic_response
    )
    if _percentile(response_latencies, 0.95) > 1_200.0:
        return "hardware physical semantic-audio p95 exceeds 1200ms"
    if (
        len(response_latencies) >= MIN_P99_SAMPLES
        and _percentile(response_latencies, 0.99) > 1_800.0
    ):
        return "hardware physical semantic-audio p99 exceeds 1800ms"
    return None


def _hardware_metadata_present(field: str, value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if field in {"input_sample_rate_hz", "output_sample_rate_hz"}:
        number = _number(value)
        return number is not None and number > 0.0
    return bool(str(value or "").strip())


def _load_fast_path(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    benchmark = _mapping(payload.get("benchmark"))
    metrics: dict[str, dict[str, Any]] = {}
    if route := _summary_metric(
        benchmark.get("route_ms"),
        evidence_type="measured",
        measurement_scope="process_microbenchmark",
        source_id="fast_path",
        threshold_p95_ms=5.0,
    ):
        metrics["route_ms"] = route
    if cache := _summary_metric(
        benchmark.get("cached_pcm_queue_ms"),
        evidence_type="measured",
        measurement_scope="process_microbenchmark",
        source_id="fast_path",
        threshold_p95_ms=15.0,
    ):
        metrics["cached_pcm_queue_ms"] = cache
    projected = _mapping(benchmark.get("projected_speech_end_to_first_pcm_ms"))
    if projected:
        evidence = "measured" if projected.get("measured_on_device") is True else "projected"
        metrics["speech_end_to_first_pcm_ms"] = _metric(
            evidence_type=evidence,
            measurement_scope=(
                "target_hardware_instrumented" if evidence == "measured" else "computed_budget"
            ),
            source_id="fast_path",
            p50_ms=_number(projected.get("p50")),
            p95_ms=_number(projected.get("p95")),
            threshold_p95_ms=900.0,
            limitations=[] if evidence == "measured" else ["not physical speaker output"],
        )
    return metrics


def _load_hardware(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    if payload.get("schema_version") != HARDWARE_REPORT_SCHEMA_VERSION:
        return _load_legacy_hardware_as_manual(payload)

    stop_values = _hardware_latency_values(
        payload,
        scenario="human_overlap",
        evidence_kind="physical_acoustic",
        outcome_field="detected",
        latency_field="speaker_stop_latency_ms",
    )
    if stop_values:
        physical_stop_metric = _hardware_metric_from_values(
            stop_values,
            evidence_type="measured",
            measurement_scope="target_hardware_physical_acoustic_instrumented",
            threshold_p95_ms=250.0,
            threshold_p99_ms=400.0,
        )
        metrics["barge_in_to_speaker_stop_ms"] = dict(physical_stop_metric)
        metrics["barge_in_to_physical_speaker_stop_ms"] = physical_stop_metric
    render_stop_values = _hardware_latency_values(
        payload,
        scenario="human_overlap",
        evidence_kind="render_chain",
        outcome_field="detected",
        latency_field="speaker_stop_latency_ms",
    )
    if render_stop_values:
        metrics["barge_in_to_render_chain_stop_ms"] = _hardware_metric_from_values(
            render_stop_values,
            evidence_type="measured",
            measurement_scope="target_hardware_render_chain_instrumented",
            limitations=["render-chain stop does not prove physical speaker silence"],
        )
    manual_stop_values = _hardware_latency_values(
        payload,
        scenario="human_overlap",
        evidence_kind="manual",
        outcome_field="detected",
        latency_field="speaker_stop_latency_ms",
    )
    if manual_stop_values:
        metrics["manual_barge_in_to_speaker_stop_ms"] = _hardware_metric_from_values(
            manual_stop_values,
            evidence_type="manual",
            measurement_scope="target_hardware_manual",
            limitations=["operator entry/reaction time; not product-grade instrumented"],
        )

    first_sound_values = _hardware_latency_values(
        payload,
        scenario="assistant_response",
        evidence_kind="physical_acoustic",
        outcome_field="heard",
        latency_field="speech_end_to_first_sound_ms",
    )
    if first_sound_values:
        metrics["speech_end_to_physical_first_sound_ms"] = _hardware_metric_from_values(
            first_sound_values,
            evidence_type="measured",
            measurement_scope="target_hardware_physical_acoustic_instrumented",
            threshold_p95_ms=1_200.0,
            threshold_p99_ms=1_800.0,
        )
    semantic_sound_values = _hardware_latency_values(
        payload,
        scenario="assistant_response",
        evidence_kind="physical_acoustic",
        outcome_field="heard",
        latency_field="speech_end_to_first_semantic_audio_ms",
        required_audio_class="semantic",
    )
    if semantic_sound_values:
        metrics["speech_end_to_physical_first_semantic_audio_ms"] = (
            _hardware_metric_from_values(
                semantic_sound_values,
                evidence_type="measured",
                measurement_scope=(
                    "target_hardware_physical_acoustic_instrumented_semantic"
                ),
                threshold_p95_ms=1_200.0,
                threshold_p99_ms=1_800.0,
            )
        )
    render_first_sound_values = _hardware_latency_values(
        payload,
        scenario="assistant_response",
        evidence_kind="render_chain",
        outcome_field="heard",
        latency_field="speech_end_to_first_sound_ms",
    )
    if render_first_sound_values:
        metrics["speech_end_to_render_chain_first_sound_ms"] = _hardware_metric_from_values(
            render_first_sound_values,
            evidence_type="measured",
            measurement_scope="target_hardware_render_chain_instrumented",
            limitations=["render-chain onset does not prove physical acoustic first sound"],
        )
    manual_first_sound_values = _hardware_latency_values(
        payload,
        scenario="assistant_response",
        evidence_kind="manual",
        outcome_field="heard",
        latency_field="speech_end_to_first_sound_ms",
    )
    if manual_first_sound_values:
        metrics["manual_speech_end_to_first_sound_ms"] = _hardware_metric_from_values(
            manual_first_sound_values,
            evidence_type="manual",
            measurement_scope="target_hardware_manual",
            limitations=["operator entry/reaction time; not product-grade instrumented"],
        )
    return metrics


def _load_legacy_hardware_as_manual(
    payload: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    summary = _mapping(payload.get("summary"))
    human_overlap = _mapping(summary.get("human_overlap"))
    stop = _mapping(human_overlap.get("speaker_stop_latency_ms"))
    assistant_response = _mapping(summary.get("assistant_response"))
    first_sound = _mapping(assistant_response.get("speech_end_to_physical_first_sound_ms"))
    metrics: dict[str, dict[str, Any]] = {}
    limitation = ["legacy v1/operator timing has no schema-v2 capture provenance"]
    if stop:
        metrics["manual_barge_in_to_speaker_stop_ms"] = _metric(
            evidence_type="manual",
            measurement_scope="legacy_target_hardware_unproven",
            source_id="full_duplex_hardware",
            p50_ms=_number(stop.get("p50")),
            p95_ms=_number(stop.get("p95")),
            p99_ms=_number(stop.get("p99")),
            count=_int(stop.get("count") or human_overlap.get("count")),
            limitations=limitation,
        )
    if first_sound:
        metrics["manual_speech_end_to_first_sound_ms"] = _metric(
            evidence_type="manual",
            measurement_scope="legacy_target_hardware_unproven",
            source_id="full_duplex_hardware",
            p50_ms=_number(first_sound.get("p50")),
            p95_ms=_number(first_sound.get("p95")),
            p99_ms=_number(first_sound.get("p99")),
            count=_int(first_sound.get("count") or assistant_response.get("count")),
            limitations=limitation,
        )
    return metrics


def _hardware_latency_values(
    payload: Mapping[str, Any],
    *,
    scenario: str,
    evidence_kind: str,
    outcome_field: str,
    latency_field: str,
    required_audio_class: str | None = None,
) -> list[float]:
    trials = _mapping(payload.get("trials")).get(scenario)
    if not isinstance(trials, Sequence) or isinstance(trials, (str, bytes)):
        return []
    values: list[float] = []
    for trial in trials:
        if not isinstance(trial, Mapping) or trial.get("evidence_kind") != evidence_kind:
            continue
        if (
            required_audio_class is not None
            and trial.get("audio_class") != required_audio_class
        ):
            continue
        if trial_evidence_schema_error(trial) is not None:
            continue
        if evidence_kind == "physical_acoustic":
            if physical_acoustic_provenance_error(trial, scenario=scenario) is not None:
                continue
        elif evidence_kind == "render_chain":
            if render_chain_provenance_error(trial, scenario=scenario) is not None:
                continue
        if trial.get(outcome_field) is not True:
            continue
        value = _number(trial.get(latency_field))
        if value is not None and value >= 0.0:
            values.append(value)
    return values


def _hardware_metric_from_values(
    values: Sequence[float],
    *,
    evidence_type: str,
    measurement_scope: str,
    threshold_p95_ms: float | None = None,
    threshold_p99_ms: float | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    ordered = sorted(values)
    metric_limitations = list(limitations or [])
    p95_ms = (
        _percentile(ordered, 0.95)
        if len(ordered) >= MIN_P95_SAMPLES
        else None
    )
    p99_ms = (
        _percentile(ordered, 0.99)
        if len(ordered) >= MIN_P99_SAMPLES
        else None
    )
    if p95_ms is None:
        metric_limitations.append(
            f"p95 requires at least {MIN_P95_SAMPLES} valid samples"
        )
    if p99_ms is None:
        metric_limitations.append(
            f"p99 requires at least {MIN_P99_SAMPLES} valid samples"
        )
    return _metric(
        evidence_type=evidence_type,
        measurement_scope=measurement_scope,
        source_id="full_duplex_hardware",
        p50_ms=_percentile(ordered, 0.50),
        p95_ms=p95_ms,
        p99_ms=p99_ms,
        count=len(ordered),
        threshold_p95_ms=threshold_p95_ms,
        threshold_p99_ms=threshold_p99_ms,
        limitations=metric_limitations,
    )


def _load_online_smoke(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    checks = payload.get("checks")
    if isinstance(checks, Mapping):
        items = ((str(name), item) for name, item in checks.items())
    else:
        items = ((str(index), item) for index, item in enumerate(checks or []))
    for name, item in items:
        if not isinstance(item, Mapping):
            continue
        latency = _number(item.get("latency_ms"))
        if latency is None:
            continue
        metrics[f"online_{_safe_metric_name(name)}_latency_ms"] = _metric(
            evidence_type="measured",
            measurement_scope="provider_network_single_run",
            source_id="online_smoke",
            p50_ms=latency,
            p95_ms=latency,
            count=1,
            limitations=["single run; not p95 evidence"],
        )
    return metrics


def _load_voice_health(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    voice_turn = _mapping(payload.get("voice_turn"))
    summary = _mapping(voice_turn.get("latency_summary"))
    buckets = _mapping(summary.get("buckets"))
    metrics: dict[str, dict[str, Any]] = {}
    for name, bucket in buckets.items():
        if not isinstance(bucket, Mapping):
            continue
        count = _int(bucket.get("count")) or 0
        if "physical" in str(name):
            valid_provenance_count = (
                _int(bucket.get("physical_provenance_valid_count")) or 0
            )
            if valid_provenance_count != count or count == 0:
                # Runtime values without verified acoustic provenance are not
                # measurements and must not enter a measured report.
                continue
        metric_name = f"turn_trace_{_safe_metric_name(str(name))}"
        derived = str(name).startswith(("speech_end_to_", "barge_in_to_"))
        metrics[metric_name] = _metric(
            evidence_type="measured",
            measurement_scope=(
                "runtime_turn_trace_event_delta"
                if derived
                else "runtime_turn_trace_legacy_listen_offset"
            ),
            source_id="voice_health",
            p50_ms=_number(bucket.get("p50_ms")),
            p95_ms=_number(bucket.get("p95_ms")),
            p99_ms=_number(bucket.get("p99_ms")),
            count=count,
            limitations=[
                "runtime snapshot only; verify source freshness separately",
                *(
                    []
                    if derived
                    else ["legacy bucket is an offset from listen start"]
                ),
            ],
        )
    return metrics


def _load_scenario(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    source_metrics = _mapping(payload.get("metrics"))
    metrics: dict[str, dict[str, Any]] = {}
    for name, value in source_metrics.items():
        latency = _number(value)
        if latency is None:
            continue
        metrics[f"scenario_{_safe_metric_name(str(name))}"] = _metric(
            evidence_type="simulated",
            measurement_scope="scenario_simulation",
            source_id="scenario",
            p50_ms=latency,
            p95_ms=latency,
            limitations=["offline deterministic scenario; no physical audio path"],
        )
    return metrics


def _summary_metric(
    value: Any,
    *,
    evidence_type: str,
    measurement_scope: str,
    source_id: str,
    threshold_p95_ms: float | None = None,
) -> dict[str, Any] | None:
    summary = _mapping(value)
    if not summary:
        return None
    return _metric(
        evidence_type=evidence_type,
        measurement_scope=measurement_scope,
        source_id=source_id,
        p50_ms=_number(summary.get("p50")),
        p95_ms=_number(summary.get("p95")),
        p99_ms=_number(summary.get("p99")),
        count=_int(summary.get("count")),
        threshold_p95_ms=threshold_p95_ms,
    )


def _metric(
    *,
    evidence_type: str,
    measurement_scope: str,
    source_id: str,
    p50_ms: float | None = None,
    p95_ms: float | None = None,
    p99_ms: float | None = None,
    count: int | None = None,
    threshold_p95_ms: float | None = None,
    threshold_p99_ms: float | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    failed = (
        threshold_p95_ms is not None and p95_ms is not None and p95_ms > threshold_p95_ms
    ) or (threshold_p99_ms is not None and p99_ms is not None and p99_ms > threshold_p99_ms)
    return {
        "evidence_type": evidence_type,
        "measurement_scope": measurement_scope,
        "source_id": source_id,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "count": count,
        "threshold_p95_ms": threshold_p95_ms,
        "threshold_p99_ms": threshold_p99_ms,
        "status": "failed" if failed else "passed",
        "limitations": limitations or [],
    }


def _validate_metric(name: str, metric: Mapping[str, Any]) -> None:
    evidence_type = metric.get("evidence_type")
    if evidence_type not in EVIDENCE_TYPES:
        raise ValueError(f"{name}: invalid evidence_type {evidence_type!r}")
    if not str(metric.get("measurement_scope") or "").strip():
        raise ValueError(f"{name}: missing measurement_scope")


def _required_metric_is_measured(metric: Mapping[str, Any] | None) -> bool:
    return bool(
        metric
        and metric.get("evidence_type") == "measured"
        and metric.get("status") == "passed"
        and metric.get("p95_ms") is not None
        and int(metric.get("count") or 0) >= MIN_REQUIRED_SAMPLES
    )


def _evidence_counts(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    return {
        evidence: sum(1 for metric in metrics.values() if metric.get("evidence_type") == evidence)
        for evidence in sorted(EVIDENCE_TYPES)
    }


def _experiment_evidence_counts(
    stage_metrics: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, int]:
    experiment_metrics = [
        metric
        for metrics in stage_metrics.values()
        for metric in metrics
        if metric.get("experiment_id") is not None
    ]
    return {
        evidence: sum(1 for metric in experiment_metrics if metric.get("evidence_type") == evidence)
        for evidence in sorted(EVIDENCE_TYPES)
    }


def _source_evidence_type(metrics: Mapping[str, Mapping[str, Any]]) -> str | None:
    types = {metric.get("evidence_type") for metric in metrics.values()}
    types.discard(None)
    if not types:
        return None
    if len(types) == 1:
        return str(next(iter(types)))
    return "mixed"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _required_text(payload: Mapping[str, Any], field: str) -> str:
    value = str(payload.get(field) or "").strip()
    if not value:
        raise ValueError(f"missing required experiment field {field!r}")
    return value


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _sample_latency(sample: Mapping[str, Any], aliases: Sequence[str]) -> float | None:
    for alias in aliases:
        if alias not in sample:
            continue
        raw_value = sample.get(alias)
        if isinstance(raw_value, bool):
            return None
        value = _number(raw_value)
        if value is not None and value >= 0.0:
            return value
        return None
    return None


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    if not ordered:
        raise ValueError("cannot calculate a percentile from an empty sample")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(float(ordered[lower]), 3)
    weight = position - lower
    value = float(ordered[lower]) * (1.0 - weight) + float(ordered[upper]) * weight
    return round(value, 3)


def _provider_family(value: Any) -> str | None:
    normalized = "".join(
        character for character in str(value or "").strip().lower() if character.isalnum()
    )
    if "minimax" in normalized:
        return "minimax"
    if "volc" in normalized or "doubao" in normalized:
        return "volcengine"
    return None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _safe_metric_name(value: str) -> str:
    return (
        "".join(character if character.isalnum() else "_" for character in value).strip("_").lower()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast-path", type=Path)
    parser.add_argument("--hardware", type=Path)
    parser.add_argument("--online-smoke", type=Path)
    parser.add_argument("--voice-health", type=Path)
    parser.add_argument("--scenario", type=Path)
    parser.add_argument(
        "--experiment",
        action="append",
        type=Path,
        default=[],
        help=(
            "offline askme.voice_latency_experiment.v1 JSON; repeat for "
            "same-corpus provider comparisons"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_report(
        fast_path=args.fast_path,
        hardware=args.hardware,
        online_smoke=args.online_smoke,
        voice_health=args.voice_health,
        scenario=args.scenario,
        experiments=args.experiment,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.out)}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
