"""Persistent, fail-closed state machine for collaborative voice hardware tests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import threading
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol
from uuid import uuid4

from askme.voice.diagnostics.full_duplex_hardware import (
    HARDWARE_REPORT_SCHEMA_VERSION,
    MIN_TRIALS_PER_SCENARIO,
    evaluate_hardware_run,
)
from askme.voice.diagnostics.hardware_audio_capture import build_manual_trial_evidence
from askme.voice.lab.audio_backend import SoundDeviceVoiceLabBackend, VoiceLabAudioBackend

VOICE_LAB_SCHEMA_VERSION = "askme.voice_lab.v1"
SCENARIOS = ("speaker_only", "human_overlap", "assistant_response")
QUALITY_VALUES = frozenset({"clear", "clipped", "choppy", "unintelligible"})
_RUN_ID_PATTERN = re.compile(r"^vlab_[A-Za-z0-9_-]{8,80}$")
_IDEMPOTENCY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_MAX_TIMELINE_EVENTS = 256
_MAX_EVIDENCE_STRING_LENGTH = 512
_MAX_EVIDENCE_CONTAINER_ITEMS = 64
_MAX_EVIDENCE_DEPTH = 4
_MAX_EVIDENCE_NODES = 2_048
_FORBIDDEN_EVIDENCE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "audio_bytes",
        "authorization",
        "credential",
        "credentials",
        "echo_control_verified",
        "password",
        "pcm",
        "product_gate_usable",
        "prompt",
        "raw_audio",
        "raw_text",
        "refresh_token",
        "samples",
        "secret",
        "text",
        "token",
        "transcript",
        "waveform",
    }
)
_FORBIDDEN_EVIDENCE_KEY_PARTS = frozenset(
    {
        "apikey",
        "authorization",
        "credential",
        "credentials",
        "password",
        "pcm",
        "prompt",
        "secret",
        "text",
        "token",
        "transcript",
        "waveform",
    }
)


class VoiceLabError(RuntimeError):
    """Base class for controlled Voice Lab failures."""


class VoiceLabValidationError(VoiceLabError, ValueError):
    """The caller supplied an invalid request."""


class VoiceLabNotFound(VoiceLabError):
    """A requested run does not exist."""


class VoiceLabConflict(VoiceLabError):
    """An idempotency key or optimistic version conflicted."""


class VoiceLabStateError(VoiceLabError):
    """The requested transition is invalid for the current run state."""


class VoiceLabEvidenceUnavailable(VoiceLabError):
    """The trusted runtime evidence adapter is absent or failed."""


class VoiceLabTrialEvidenceProvider(Protocol):
    """Trusted server adapter that executes one active Voice Lab attempt."""

    def __call__(
        self,
        *,
        correlation_id: str,
        run_context: Mapping[str, Any],
        scenario_context: Mapping[str, Any],
        device_context: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class VoiceLabService:
    """Own TestRun/Trial persistence while exposing a compact operator contract."""

    def __init__(
        self,
        root: str | Path,
        *,
        audio_backend: VoiceLabAudioBackend | None = None,
        trial_evidence_provider: VoiceLabTrialEvidenceProvider | None = None,
        now: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._root = Path(root)
        self._runs_root = self._root / "runs"
        self._idempotency_path = self._root / "create-idempotency.json"
        self._audio = audio_backend or SoundDeviceVoiceLabBackend()
        self._trial_evidence_provider = trial_evidence_provider
        self._now = now or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.monotonic
        self._lock = threading.RLock()

    def list_devices(self) -> dict[str, Any]:
        payload = dict(self._audio.inventory())
        payload["capabilities"] = dict(self._audio.capabilities())
        payload["evidence_policy"] = {
            "manual_marks_are_diagnostic_only": True,
            "render_loopback_is_not_physical_acoustic_evidence": True,
            "physical_overlap_stop_requires_isolated_speaker_monitor": True,
            "minimum_trials_per_scenario": MIN_TRIALS_PER_SCENARIO,
        }
        return payload

    def create_run(
        self,
        body: Mapping[str, Any],
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
        normalized = _normalize_create_body(body)
        key = _validate_idempotency_key(idempotency_key)
        body_hash = _payload_hash({"action": "create", "body": normalized})
        with self._lock:
            index = self._load_create_index()
            previous = index.get(key)
            if isinstance(previous, Mapping):
                if previous.get("body_hash") != body_hash:
                    raise VoiceLabConflict("idempotency key was already used with another body")
                return self.get_run(str(previous.get("run_id") or ""))

            run_id = self._new_run_id()
            timestamp = self._timestamp()
            capabilities = dict(self._audio.capabilities())
            blocked_reasons = _capability_blocked_reasons(capabilities)
            run: dict[str, Any] = {
                "schema_version": VOICE_LAB_SCHEMA_VERSION,
                "hardware_report_schema_version": HARDWARE_REPORT_SCHEMA_VERSION,
                "run_id": run_id,
                "version": 1,
                "status": "needs_device_check",
                "operator_id": normalized["operator_id"],
                "room": normalized["room"],
                "no_ros2": True,
                "device_binding": normalized["device_binding"],
                "plan": {scenario: MIN_TRIALS_PER_SCENARIO for scenario in SCENARIOS},
                "capabilities": capabilities,
                "product_gate_possible": not blocked_reasons,
                "product_gate_blocked_reasons": blocked_reasons,
                "device_check": {"status": "pending"},
                "calibration": {"status": "pending"},
                "trials": [],
                "active_trial": None,
                "invalidated_trials": [],
                "operations": {},
                "manual_diagnostic_complete": False,
                "product_gate": {
                    "status": "not_evaluated",
                    "reason": "instrumented physical evidence has not been collected",
                },
                "created_at": timestamp,
                "updated_at": timestamp,
            }
            self._refresh_derived(run)
            self._save_run(run)
            index[key] = {"body_hash": body_hash, "run_id": run_id}
            self._save_create_index(index)
            return self._public_run(run)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            return self._public_run(self._load_run(run_id))

    def check_devices(
        self,
        run_id: str,
        *,
        expected_version: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        request = {"run_id": run_id, "action": "device_check"}
        with self._lock:
            run = self._load_run(run_id)
            if self._operation_replayed(run, idempotency_key, request):
                return self._public_run(run)
            self._require_version(run, expected_version)
            if run["status"] not in {"needs_device_check", "blocked"}:
                raise VoiceLabStateError("device check is not allowed in the current state")
            result = self._audio.run_device_check(
                run_dir=self._artifact_dir(run_id),
                device_binding=deepcopy(run["device_binding"]),
            )
            run["device_check"] = self._device_check_payload(run_id, result)
            if result.get("status") == "ok":
                run["status"] = "needs_calibration"
                run["device_check"]["blocking_reason"] = None
            else:
                run["status"] = "blocked"
                run["device_check"]["blocking_reason"] = (
                    result.get("failure_reason") or result.get("error") or "device_check_failed"
                )
            return self._commit(run, idempotency_key, request)

    def calibrate(
        self,
        run_id: str,
        body: Mapping[str, Any],
        *,
        expected_version: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        duration_s = _finite_float(body.get("duration_s", 2.0), "duration_s")
        if not 0.5 <= duration_s <= 10.0:
            raise VoiceLabValidationError("duration_s must be between 0.5 and 10")
        request = {"run_id": run_id, "action": "calibrate", "duration_s": duration_s}
        with self._lock:
            run = self._load_run(run_id)
            if self._operation_replayed(run, idempotency_key, request):
                return self._public_run(run)
            self._require_version(run, expected_version)
            if run["status"] != "needs_calibration" or run["device_check"].get("status") != "ok":
                raise VoiceLabStateError("a successful device check is required before calibration")
            result = self._audio.calibrate_microphone(
                device_binding=deepcopy(run["device_binding"]),
                duration_s=duration_s,
            )
            run["calibration"] = deepcopy(result)
            run["calibration"]["captured_at"] = self._timestamp()
            if result.get("status") == "ok" and isinstance(result.get("calibration"), Mapping):
                run["status"] = "running"
            else:
                run["status"] = "blocked"
                run["calibration"]["blocking_reason"] = (
                    result.get("error") or "microphone_calibration_failed"
                )
            return self._commit(run, idempotency_key, request)

    def begin_trial(
        self,
        run_id: str,
        *,
        expected_version: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Start the one server-owned attempt that may be completed next."""

        request = {"run_id": run_id, "action": "begin_trial"}
        with self._lock:
            run = self._load_run(run_id)
            if self._operation_replayed(run, idempotency_key, request):
                return self._public_run(run)
            self._require_version(run, expected_version)
            if run["status"] != "running":
                raise VoiceLabStateError("trial attempts require a calibrated running test")
            if isinstance(run.get("active_trial"), Mapping):
                raise VoiceLabStateError("an active trial attempt already exists")
            next_trial = self._next_trial(run)
            if next_trial.get("action") != "trial":
                raise VoiceLabStateError("no trial attempt is available to start")
            run["active_trial"] = {
                "attempt_id": f"vat_{uuid4().hex}",
                "scenario": next_trial["scenario"],
                "ordinal": next_trial["ordinal"],
                "started_at": self._timestamp(),
            }
            return self._commit(run, idempotency_key, request)

    def submit_trial(
        self,
        run_id: str,
        body: Mapping[str, Any],
        *,
        expected_version: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        normalized = _normalize_trial_body(body)
        request = {"run_id": run_id, "action": "trial", "body": normalized}
        with self._lock:
            run = self._load_run(run_id)
            if self._operation_replayed(run, idempotency_key, request):
                return self._public_run(run)
            self._require_version(run, expected_version)
            if run["status"] != "running":
                raise VoiceLabStateError("trials require a calibrated running test")
            active_trial = run.get("active_trial")
            if not isinstance(active_trial, Mapping):
                raise VoiceLabStateError("a server-started trial attempt is required")
            if normalized["attempt_id"] != active_trial.get("attempt_id"):
                raise VoiceLabStateError("submission does not match the active trial attempt")
            expected = (active_trial.get("scenario"), active_trial.get("ordinal"))
            supplied = (normalized["scenario"], normalized["ordinal"])
            if expected != supplied:
                raise VoiceLabStateError(
                    "submission does not match the active trial attempt: "
                    f"expected {expected[0]} #{expected[1]}, not {supplied[0]} #{supplied[1]}"
                )
            observed = self._monotonic()
            evidence = build_manual_trial_evidence(
                method="manual_observation",
                reference_event=_reference_event(normalized["scenario"]),
                observed_timestamp_s=observed,
            )
            trial = {
                "trial_id": f"{run_id}-{normalized['scenario']}-{normalized['ordinal']:02d}",
                "attempt_id": normalized["attempt_id"],
                "scenario": normalized["scenario"],
                "ordinal": normalized["ordinal"],
                **evidence,
                "operator_mark": {
                    "operator_id": run["operator_id"],
                    "quality": normalized["quality"],
                    "notes": normalized["notes"],
                    "recorded_at": self._timestamp(),
                },
                "product_gate_usable": False,
                "product_gate_reason": "manual evidence is diagnostic only",
            }
            turn_evidence = active_trial.get("turn_evidence")
            if isinstance(turn_evidence, Mapping):
                trial["turn_evidence"] = deepcopy(dict(turn_evidence))
                trial["product_gate_reason"] = (
                    "runtime execution evidence alone does not prove physical product gates"
                )
            if normalized["scenario"] == "speaker_only":
                trial["false_barge_in"] = normalized["false_barge_in"]
            elif normalized["scenario"] == "human_overlap":
                trial["detected"] = normalized["detected"]
            else:
                trial["heard"] = normalized["heard"]
            run["trials"].append(trial)
            run["active_trial"] = None
            self._refresh_derived(run)
            return self._commit(run, idempotency_key, request)

    def execute_trial(
        self,
        run_id: str,
        attempt_id: str,
        *,
        expected_version: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Execute one active attempt through the trusted runtime evidence adapter."""

        normalized_attempt_id = _bounded_text(attempt_id, "attempt_id", maximum=128)
        request = {
            "run_id": run_id,
            "action": "execute_trial",
            "attempt_id": normalized_attempt_id,
        }
        with self._lock:
            run = self._load_run(run_id)
            if self._operation_replayed(run, idempotency_key, request):
                return self._public_run(run)
            self._require_version(run, expected_version)
            if run["status"] != "running":
                raise VoiceLabStateError("trial execution requires a calibrated running test")
            active_trial = run.get("active_trial")
            if not isinstance(active_trial, Mapping):
                raise VoiceLabStateError("an active trial attempt is required for execution")
            if active_trial.get("attempt_id") != normalized_attempt_id:
                raise VoiceLabStateError("execution does not match the active trial attempt")
            if isinstance(active_trial.get("turn_evidence"), Mapping):
                raise VoiceLabStateError("the active trial attempt has already been executed")
            provider = self._trial_evidence_provider
            if provider is None:
                raise VoiceLabEvidenceUnavailable("trial evidence provider is unavailable")

            run_context = _immutable_context(
                {
                    "run_id": run["run_id"],
                    "schema_version": run["schema_version"],
                    "operator_id": run["operator_id"],
                    "room": run["room"],
                    "version": run["version"],
                    "no_ros2": run["no_ros2"],
                }
            )
            scenario_context = _immutable_context(
                {
                    "attempt_id": active_trial["attempt_id"],
                    "scenario": active_trial["scenario"],
                    "ordinal": active_trial["ordinal"],
                    "started_at": active_trial["started_at"],
                }
            )
            device_context = _immutable_context(run["device_binding"])
            try:
                supplied_evidence = provider(
                    correlation_id=normalized_attempt_id,
                    run_context=run_context,
                    scenario_context=scenario_context,
                    device_context=device_context,
                )
            except Exception as exc:
                raise VoiceLabEvidenceUnavailable(
                    f"trial evidence provider failed: {type(exc).__name__}"
                ) from exc

            turn_evidence = _normalize_turn_evidence(
                supplied_evidence,
                expected_correlation_id=normalized_attempt_id,
                captured_at=self._timestamp(),
            )
            updated_trial = deepcopy(dict(active_trial))
            updated_trial["turn_evidence"] = turn_evidence
            run["active_trial"] = updated_trial
            return self._commit(run, idempotency_key, request)

    def pause(
        self,
        run_id: str,
        *,
        expected_version: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        request = {"run_id": run_id, "action": "pause"}
        with self._lock:
            run = self._load_run(run_id)
            if self._operation_replayed(run, idempotency_key, request):
                return self._public_run(run)
            self._require_version(run, expected_version)
            if run["status"] not in {
                "needs_device_check",
                "needs_calibration",
                "running",
                "blocked",
            }:
                raise VoiceLabStateError("run cannot be paused in the current state")
            self._invalidate_active_trial(run, reason="run_paused")
            run["status"] = "paused"
            return self._commit(run, idempotency_key, request)

    def resume(
        self,
        run_id: str,
        *,
        expected_version: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        request = {"run_id": run_id, "action": "resume"}
        with self._lock:
            run = self._load_run(run_id)
            if self._operation_replayed(run, idempotency_key, request):
                return self._public_run(run)
            self._require_version(run, expected_version)
            if run["status"] != "paused":
                raise VoiceLabStateError("only a paused run can be resumed")
            run["status"] = "needs_device_check"
            run["device_check"] = {"status": "stale", "reason": "run_resumed"}
            run["calibration"] = {"status": "stale", "reason": "run_resumed"}
            return self._commit(run, idempotency_key, request)

    def abort(
        self,
        run_id: str,
        *,
        expected_version: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        request = {"run_id": run_id, "action": "abort"}
        with self._lock:
            run = self._load_run(run_id)
            if self._operation_replayed(run, idempotency_key, request):
                return self._public_run(run)
            self._require_version(run, expected_version)
            if run["status"] in {"completed", "aborted"}:
                raise VoiceLabStateError("terminal run cannot be aborted")
            self._invalidate_active_trial(run, reason="run_aborted")
            run["status"] = "aborted"
            run["aborted_at"] = self._timestamp()
            return self._commit(run, idempotency_key, request)

    def generate_report(
        self,
        run_id: str,
        *,
        expected_version: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        request = {"run_id": run_id, "action": "report"}
        with self._lock:
            run = self._load_run(run_id)
            if self._operation_replayed(run, idempotency_key, request):
                return self._public_run(run)
            self._require_version(run, expected_version)
            if run["status"] == "aborted":
                raise VoiceLabStateError("aborted run cannot generate a final report")
            if (
                run["status"] != "ready_for_report"
                or run.get("manual_diagnostic_complete") is not True
                or isinstance(run.get("active_trial"), Mapping)
            ):
                raise VoiceLabStateError(
                    "report requires a completed trial plan with no active trial attempt"
                )
            trials = list(run["trials"])
            report = evaluate_hardware_run(
                config=self._report_config(run),
                metadata=self._report_metadata(run),
                speaker_only_trials=[t for t in trials if t.get("scenario") == "speaker_only"],
                overlap_trials=[t for t in trials if t.get("scenario") == "human_overlap"],
                response_trials=[
                    t for t in trials if t.get("scenario") == "assistant_response"
                ],
                require_response_trials=True,
            )
            artifact_path = self._artifact_dir(run_id) / "hardware-report.json"
            self._atomic_write_json(artifact_path, report)
            run["product_gate"] = {
                "status": report["status"],
                "reason": (
                    "strict physical evidence requirements passed"
                    if report["status"] == "passed"
                    else "diagnostic completed; physical evidence requirements are not satisfied"
                ),
                "artifact": f"artifacts/voice-lab/runs/{run_id}/hardware-report.json",
                "report": report,
            }
            self._refresh_derived(run)
            if run["manual_diagnostic_complete"]:
                run["status"] = "completed"
                run["completed_at"] = self._timestamp()
            return self._commit(run, idempotency_key, request)

    def _report_config(self, run: Mapping[str, Any]) -> dict[str, Any]:
        binding = run["device_binding"]
        aec_backend = str(binding.get("aec_backend") or "none")
        return {
            "voice": {
                "full_duplex": {
                    "enabled": aec_backend != "none",
                    "echo_control": aec_backend,
                    "echo_control_verified": False,
                }
            }
        }

    def _report_metadata(self, run: Mapping[str, Any]) -> dict[str, Any]:
        binding = run["device_binding"]
        return {
            "operating_system": platform.platform(),
            "python_version": platform.python_version(),
            "room": run["room"],
            "audio_device": binding["audio_device"],
            "audio_driver": binding["audio_driver"],
            "input_device_id": str(binding["input_device_id"]),
            "output_device_id": str(binding["output_device_id"]),
            "input_sample_rate_hz": int(binding["input_sample_rate_hz"]),
            "output_sample_rate_hz": int(binding["output_sample_rate_hz"]),
            "aec_backend": str(binding.get("aec_backend") or "none"),
        }

    def _device_check_payload(self, run_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
        payload = deepcopy(dict(result))
        if payload.pop("wav_out", None):
            payload["artifact"] = f"artifacts/voice-lab/runs/{run_id}/device-check.wav"
        payload["captured_at"] = self._timestamp()
        payload["evidence_kind"] = "diagnostic_room_loopback"
        payload["product_gate_usable"] = False
        return payload

    def _commit(
        self,
        run: dict[str, Any],
        idempotency_key: str,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        key = _validate_idempotency_key(idempotency_key)
        run["version"] = int(run["version"]) + 1
        run["updated_at"] = self._timestamp()
        run.setdefault("operations", {})[key] = {
            "body_hash": _payload_hash(request),
            "version": run["version"],
        }
        self._refresh_derived(run)
        self._save_run(run)
        return self._public_run(run)

    def _operation_replayed(
        self,
        run: Mapping[str, Any],
        idempotency_key: str,
        request: Mapping[str, Any],
    ) -> bool:
        key = _validate_idempotency_key(idempotency_key)
        previous = run.get("operations", {}).get(key)
        if not isinstance(previous, Mapping):
            return False
        if previous.get("body_hash") != _payload_hash(request):
            raise VoiceLabConflict("idempotency key was already used with another operation")
        return True

    def _require_version(self, run: Mapping[str, Any], expected_version: int) -> None:
        if isinstance(expected_version, bool) or not isinstance(expected_version, int):
            raise VoiceLabValidationError("expected version must be an integer")
        if int(run["version"]) != expected_version:
            raise VoiceLabConflict(
                f"version conflict: expected {expected_version}, current {run['version']}"
            )

    def _refresh_derived(self, run: dict[str, Any]) -> None:
        counts = {
            scenario: sum(1 for trial in run["trials"] if trial.get("scenario") == scenario)
            for scenario in SCENARIOS
        }
        total = sum(counts.values())
        required_total = sum(int(run["plan"][scenario]) for scenario in SCENARIOS)
        run["progress"] = {
            **counts,
            "total": total,
            "required_total": required_total,
            "percent": round((total / required_total) * 100, 1) if required_total else 0.0,
        }
        run["manual_diagnostic_complete"] = all(
            counts[scenario] >= int(run["plan"][scenario]) for scenario in SCENARIOS
        )
        if run["status"] == "running" and run["manual_diagnostic_complete"]:
            run["status"] = "ready_for_report"
        run["next_action"] = self._next_action(run)

    def _next_action(self, run: Mapping[str, Any]) -> dict[str, Any]:
        status = str(run["status"])
        if status in {"needs_device_check", "blocked"}:
            return {"action": "device_check"}
        if status == "needs_calibration":
            return {"action": "calibration"}
        if status == "running":
            active_trial = run.get("active_trial")
            if isinstance(active_trial, Mapping):
                return {
                    "action": "trial_active",
                    "attempt_id": active_trial.get("attempt_id"),
                    "scenario": active_trial.get("scenario"),
                    "ordinal": active_trial.get("ordinal"),
                }
            return self._next_trial(run)
        if status == "ready_for_report":
            return {"action": "report"}
        if status == "paused":
            return {"action": "resume"}
        if status == "completed":
            return {"action": "view_report"}
        return {"action": "none"}

    def _next_trial(self, run: Mapping[str, Any]) -> dict[str, Any]:
        trials = run.get("trials", [])
        for scenario in SCENARIOS:
            count = sum(1 for trial in trials if trial.get("scenario") == scenario)
            required = int(run["plan"][scenario])
            if count < required:
                return {"action": "trial", "scenario": scenario, "ordinal": count + 1}
        return {"action": "report"}

    def _new_run_id(self) -> str:
        stamp = self._now().strftime("%Y%m%dT%H%M%S")
        return f"vlab_{stamp}_{uuid4().hex[:8]}"

    def _invalidate_active_trial(self, run: dict[str, Any], *, reason: str) -> None:
        active_trial = run.get("active_trial")
        if not isinstance(active_trial, Mapping):
            run["active_trial"] = None
            return
        invalidated = deepcopy(dict(active_trial))
        invalidated.update({"reason": reason, "invalidated_at": self._timestamp()})
        run.setdefault("invalidated_trials", []).append(invalidated)
        run["active_trial"] = None

    def _timestamp(self) -> str:
        return self._now().astimezone(UTC).isoformat()

    def _run_path(self, run_id: str) -> Path:
        _validate_run_id(run_id)
        return self._runs_root / f"{run_id}.json"

    def _artifact_dir(self, run_id: str) -> Path:
        _validate_run_id(run_id)
        return self._runs_root / run_id

    def _load_run(self, run_id: str) -> dict[str, Any]:
        path = self._run_path(run_id)
        if not path.is_file():
            raise VoiceLabNotFound(f"voice lab run not found: {run_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VoiceLabError(f"voice lab run cannot be read: {run_id}") from exc
        if not isinstance(payload, dict) or payload.get("run_id") != run_id:
            raise VoiceLabError(f"voice lab run is invalid: {run_id}")
        payload.setdefault("active_trial", None)
        payload.setdefault("invalidated_trials", [])
        return payload

    def _save_run(self, run: Mapping[str, Any]) -> None:
        self._atomic_write_json(self._run_path(str(run["run_id"])), run)

    def _load_create_index(self) -> dict[str, Any]:
        if not self._idempotency_path.is_file():
            return {}
        try:
            payload = json.loads(self._idempotency_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VoiceLabError("voice lab idempotency index cannot be read") from exc
        if not isinstance(payload, dict):
            raise VoiceLabError("voice lab idempotency index is invalid")
        return payload

    def _save_create_index(self, index: Mapping[str, Any]) -> None:
        self._atomic_write_json(self._idempotency_path, index)

    def _atomic_write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _public_run(run: Mapping[str, Any]) -> dict[str, Any]:
        payload = deepcopy(dict(run))
        payload.pop("operations", None)
        return payload


def _normalize_create_body(body: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(body, Mapping):
        raise VoiceLabValidationError("request body must be an object")
    operator_id = _bounded_text(body.get("operator_id"), "operator_id", maximum=128)
    room = _bounded_text(body.get("room"), "room", maximum=128)
    raw_binding = body.get("device_binding")
    if not isinstance(raw_binding, Mapping):
        raise VoiceLabValidationError("device_binding must be an object")
    binding = {
        "input_device_id": _device_id(raw_binding.get("input_device_id"), "input_device_id"),
        "output_device_id": _device_id(
            raw_binding.get("output_device_id"), "output_device_id"
        ),
        "audio_device": _bounded_text(
            raw_binding.get("audio_device"), "audio_device", maximum=256
        ),
        "audio_driver": _bounded_text(
            raw_binding.get("audio_driver"), "audio_driver", maximum=128
        ),
        "input_sample_rate_hz": _sample_rate(
            raw_binding.get("input_sample_rate_hz"), "input_sample_rate_hz"
        ),
        "output_sample_rate_hz": _sample_rate(
            raw_binding.get("output_sample_rate_hz"), "output_sample_rate_hz"
        ),
        "aec_backend": _bounded_text(
            raw_binding.get("aec_backend", "none"), "aec_backend", maximum=64
        ).lower(),
    }
    return {"operator_id": operator_id, "room": room, "device_binding": binding}


def _normalize_trial_body(body: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(body, Mapping):
        raise VoiceLabValidationError("trial body must be an object")
    scenario = str(body.get("scenario") or "").strip()
    if scenario not in SCENARIOS:
        raise VoiceLabValidationError(f"scenario must be one of {', '.join(SCENARIOS)}")
    ordinal = body.get("ordinal")
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or not 1 <= ordinal <= 20:
        raise VoiceLabValidationError("ordinal must be an integer between 1 and 20")
    quality = str(body.get("quality") or "").strip().lower()
    if quality not in QUALITY_VALUES:
        raise VoiceLabValidationError(f"quality must be one of {', '.join(sorted(QUALITY_VALUES))}")
    notes = str(body.get("notes") or "").strip()
    if len(notes) > 500:
        raise VoiceLabValidationError("notes must contain at most 500 characters")
    normalized: dict[str, Any] = {
        "attempt_id": _bounded_text(body.get("attempt_id"), "attempt_id", maximum=128),
        "scenario": scenario,
        "ordinal": ordinal,
        "quality": quality,
        "notes": notes,
    }
    field = {
        "speaker_only": "false_barge_in",
        "human_overlap": "detected",
        "assistant_response": "heard",
    }[scenario]
    value = body.get(field)
    if not isinstance(value, bool):
        raise VoiceLabValidationError(f"{field} must be a boolean")
    normalized[field] = value
    return normalized


def _reference_event(scenario: str) -> str:
    return {
        "speaker_only": "assistant_playback_observation",
        "human_overlap": "operator_overlap_observation",
        "assistant_response": "assistant_response_observation",
    }[scenario]


def _capability_blocked_reasons(capabilities: Mapping[str, Any]) -> list[str]:
    reasons = []
    if capabilities.get("physical_overlap_stop_collector") is not True:
        reasons.append("missing_isolated_speaker_monitor")
    if capabilities.get("physical_first_sound_collector") is not True:
        reasons.append("physical_first_sound_collector_not_connected")
    return reasons


def _validate_run_id(run_id: str) -> str:
    value = str(run_id or "")
    if not _RUN_ID_PATTERN.fullmatch(value):
        raise VoiceLabValidationError("invalid voice lab run id")
    return value


def _validate_idempotency_key(key: str) -> str:
    value = str(key or "").strip()
    if not _IDEMPOTENCY_PATTERN.fullmatch(value):
        raise VoiceLabValidationError(
            "Idempotency-Key must contain 1-128 letters, numbers, dots, colons, dashes, or underscores"
        )
    return value


def _bounded_text(value: Any, field: str, *, maximum: int) -> str:
    text = str(value or "").strip()
    if not text:
        raise VoiceLabValidationError(f"{field} is required")
    if len(text) > maximum:
        raise VoiceLabValidationError(f"{field} must contain at most {maximum} characters")
    return text


def _device_id(value: Any, field: str) -> int | str:
    if isinstance(value, bool) or value is None:
        raise VoiceLabValidationError(f"{field} is required")
    if isinstance(value, int):
        if value < 0:
            raise VoiceLabValidationError(f"{field} must not be negative")
        return value
    return _bounded_text(value, field, maximum=256)


def _sample_rate(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise VoiceLabValidationError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise VoiceLabValidationError(f"{field} must be an integer") from exc
    if not 8_000 <= parsed <= 192_000:
        raise VoiceLabValidationError(f"{field} must be between 8000 and 192000")
    return parsed


def _finite_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise VoiceLabValidationError(f"{field} must be a finite number") from exc
    if not parsed == parsed or parsed in {float("inf"), float("-inf")}:
        raise VoiceLabValidationError(f"{field} must be a finite number")
    return parsed


def _normalize_turn_evidence(
    payload: Mapping[str, Any],
    *,
    expected_correlation_id: str,
    captured_at: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise VoiceLabValidationError("trial evidence provider must return an object")
    _validate_bounded_evidence(payload)
    _reject_unknown_fields(
        payload,
        {
            "correlation_id",
            "source",
            "captured_at",
            "timeline",
            "fallback",
            "interrupt",
            "configured_full_duplex",
            "runtime_full_duplex",
            "echo_control_evidence",
            "aec_stats",
            "residual_audio",
        },
        "turn_evidence",
    )
    correlation_id = _bounded_text(
        payload.get("correlation_id"), "correlation_id", maximum=128
    )
    if correlation_id != expected_correlation_id:
        raise VoiceLabValidationError("trial evidence correlation_id does not match attempt_id")
    source = _bounded_text(payload.get("source"), "source", maximum=32)
    if source != "server_runtime":
        raise VoiceLabValidationError("trial evidence source must be server_runtime")

    timeline = _normalize_evidence_timeline(payload.get("timeline"))
    fallback = _normalize_evidence_fallback(payload.get("fallback"))
    interrupt = _normalize_evidence_interrupt(payload.get("interrupt"))
    echo_control = payload.get("echo_control_evidence")
    if not isinstance(echo_control, Mapping):
        raise VoiceLabValidationError("echo_control_evidence must be an object")
    aec_stats = _normalize_aec_stats(payload.get("aec_stats"))

    normalized: dict[str, Any] = {
        "correlation_id": correlation_id,
        "source": "server_runtime",
        "captured_at": captured_at,
        "timeline": timeline,
        "fallback": fallback,
        "interrupt": interrupt,
        "configured_full_duplex": _required_bool(
            payload.get("configured_full_duplex"), "configured_full_duplex"
        ),
        "runtime_full_duplex": _required_bool(
            payload.get("runtime_full_duplex"), "runtime_full_duplex"
        ),
        "echo_control_evidence": _sanitize_evidence_value(echo_control),
        "aec_stats": aec_stats,
    }
    if "residual_audio" in payload and payload.get("residual_audio") is not None:
        normalized["residual_audio"] = _normalize_residual_audio(
            payload.get("residual_audio")
        )
    return normalized


def _normalize_evidence_timeline(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        raise VoiceLabValidationError("timeline must be an ordered list")
    if not value:
        raise VoiceLabValidationError("timeline must contain at least one event")
    if len(value) > _MAX_TIMELINE_EVENTS:
        raise VoiceLabValidationError(
            f"timeline must contain at most {_MAX_TIMELINE_EVENTS} events"
        )
    normalized: list[dict[str, Any]] = []
    previous_sequence = 0
    previous_offset = -1.0
    for index, item in enumerate(value):
        field = f"timeline[{index}]"
        if not isinstance(item, Mapping):
            raise VoiceLabValidationError(f"{field} must be an object")
        _reject_unknown_fields(item, {"event", "stage", "offset_ms", "sequence"}, field)
        event = _bounded_text(item.get("event"), f"{field}.event", maximum=128)
        stage = _bounded_text(item.get("stage"), f"{field}.stage", maximum=128)
        offset_ms = _finite_float(item.get("offset_ms"), f"{field}.offset_ms")
        if not 0.0 <= offset_ms <= 86_400_000.0:
            raise VoiceLabValidationError(f"{field}.offset_ms is outside the supported range")
        sequence = item.get("sequence")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or not 1 <= sequence <= 1_000_000
        ):
            raise VoiceLabValidationError(f"{field}.sequence must be a positive integer")
        if sequence <= previous_sequence or offset_ms < previous_offset:
            raise VoiceLabValidationError("timeline events must be ordered by sequence and offset_ms")
        normalized.append(
            {
                "event": event,
                "stage": stage,
                "offset_ms": offset_ms,
                "sequence": sequence,
            }
        )
        previous_sequence = sequence
        previous_offset = offset_ms
    return normalized


def _normalize_evidence_fallback(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VoiceLabValidationError("fallback must be an object")
    _reject_unknown_fields(value, {"used", "from", "to", "reason"}, "fallback")
    return {
        "used": _required_bool(value.get("used"), "fallback.used"),
        "from": _bounded_optional_text(value.get("from"), "fallback.from", maximum=128),
        "to": _bounded_optional_text(value.get("to"), "fallback.to", maximum=128),
        "reason": _bounded_optional_text(value.get("reason"), "fallback.reason", maximum=256),
    }


def _normalize_evidence_interrupt(value: Any) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise VoiceLabValidationError("interrupt must be an object")
    fields = {"detected", "confirmed", "dismissed", "playback_resumed"}
    _reject_unknown_fields(value, fields, "interrupt")
    return {field: _required_bool(value.get(field), f"interrupt.{field}") for field in fields}


def _normalize_aec_stats(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VoiceLabValidationError("aec_stats must be an object")
    fields = {
        "backend",
        "active",
        "degraded",
        "erl_db",
        "erle_db",
        "residual_echo_likelihood",
        "evidence_kind",
    }
    _reject_unknown_fields(value, fields, "aec_stats")
    evidence_kind = _bounded_text(
        value.get("evidence_kind"), "aec_stats.evidence_kind", maximum=64
    )
    if evidence_kind != "algorithm_telemetry":
        raise VoiceLabValidationError(
            "aec_stats.evidence_kind must be algorithm_telemetry"
        )
    likelihood = _optional_finite_float(
        value.get("residual_echo_likelihood"), "aec_stats.residual_echo_likelihood"
    )
    if likelihood is not None and not 0.0 <= likelihood <= 1.0:
        raise VoiceLabValidationError(
            "aec_stats.residual_echo_likelihood must be between 0 and 1"
        )
    return {
        "backend": _bounded_text(value.get("backend"), "aec_stats.backend", maximum=128),
        "active": _required_bool(value.get("active"), "aec_stats.active"),
        "degraded": _required_bool(value.get("degraded"), "aec_stats.degraded"),
        "erl_db": _bounded_db(value.get("erl_db"), "aec_stats.erl_db"),
        "erle_db": _bounded_db(value.get("erle_db"), "aec_stats.erle_db"),
        "residual_echo_likelihood": likelihood,
        "evidence_kind": "algorithm_telemetry",
    }


def _normalize_residual_audio(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise VoiceLabValidationError("residual_audio must be an object")
    fields = {
        "evidence_kind",
        "measurement_source",
        "clock_domain",
        "dropped_frames",
        "tail_ms",
    }
    _reject_unknown_fields(value, fields, "residual_audio")
    if value.get("evidence_kind") != "physical":
        raise VoiceLabValidationError("residual_audio.evidence_kind must be physical")
    dropped_frames = value.get("dropped_frames")
    if (
        isinstance(dropped_frames, bool)
        or not isinstance(dropped_frames, int)
        or not 0 <= dropped_frames <= 1_000_000_000
    ):
        raise VoiceLabValidationError(
            "residual_audio.dropped_frames must be a non-negative integer"
        )
    tail_ms = _finite_float(value.get("tail_ms"), "residual_audio.tail_ms")
    if not 0.0 <= tail_ms <= 60_000.0:
        raise VoiceLabValidationError("residual_audio.tail_ms is outside the supported range")
    return {
        "evidence_kind": "physical",
        "measurement_source": _bounded_text(
            value.get("measurement_source"),
            "residual_audio.measurement_source",
            maximum=128,
        ),
        "clock_domain": _bounded_text(
            value.get("clock_domain"), "residual_audio.clock_domain", maximum=128
        ),
        "dropped_frames": dropped_frames,
        "tail_ms": tail_ms,
    }


def _required_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise VoiceLabValidationError(f"{field} must be a boolean")
    return value


def _bounded_optional_text(value: Any, field: str, *, maximum: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise VoiceLabValidationError(f"{field} must be text")
    text = value.strip()
    if len(text) > maximum:
        raise VoiceLabValidationError(f"{field} must contain at most {maximum} characters")
    return text


def _optional_finite_float(value: Any, field: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, field)


def _bounded_db(value: Any, field: str) -> float | None:
    number = _optional_finite_float(value, field)
    if number is not None and not -200.0 <= number <= 200.0:
        raise VoiceLabValidationError(f"{field} is outside the supported range")
    return number


def _reject_unknown_fields(
    value: Mapping[str, Any], allowed: set[str], field: str
) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise VoiceLabValidationError(f"{field} contains unsupported fields: {', '.join(unknown)}")


def _validate_bounded_evidence(
    value: Any,
    *,
    depth: int = 0,
    node_count: list[int] | None = None,
) -> None:
    if node_count is None:
        node_count = [0]
    node_count[0] += 1
    if node_count[0] > _MAX_EVIDENCE_NODES:
        raise VoiceLabValidationError("trial evidence contains too many values")
    if depth > _MAX_EVIDENCE_DEPTH:
        raise VoiceLabValidationError("trial evidence nesting is too deep")
    if isinstance(value, Mapping):
        if len(value) > _MAX_EVIDENCE_CONTAINER_ITEMS:
            raise VoiceLabValidationError("trial evidence object contains too many fields")
        for key, item in value.items():
            if not isinstance(key, str):
                raise VoiceLabValidationError("trial evidence field names must be text")
            normalized_key = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
            key_parts = frozenset(normalized_key.split("_"))
            if normalized_key in _FORBIDDEN_EVIDENCE_KEYS or (
                key_parts & _FORBIDDEN_EVIDENCE_KEY_PARTS
            ):
                raise VoiceLabValidationError(
                    f"trial evidence contains forbidden field: {normalized_key}"
                )
            _validate_bounded_evidence(
                item,
                depth=depth + 1,
                node_count=node_count,
            )
        return
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_TIMELINE_EVENTS:
            raise VoiceLabValidationError("trial evidence list contains too many items")
        for item in value:
            _validate_bounded_evidence(
                item,
                depth=depth + 1,
                node_count=node_count,
            )
        return
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if -(2**63) <= value <= 2**63 - 1:
            return
        raise VoiceLabValidationError("trial evidence integer is outside the supported range")
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise VoiceLabValidationError("trial evidence numbers must be finite")
    if isinstance(value, str):
        if len(value) <= _MAX_EVIDENCE_STRING_LENGTH:
            return
        raise VoiceLabValidationError("trial evidence text exceeds the supported length")
    raise VoiceLabValidationError("trial evidence contains an unsupported value type")


def _sanitize_evidence_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_evidence_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_evidence_value(item) for item in value]
    return value


def _immutable_context(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = _freeze_context_value(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover - the input contract is a mapping
        raise TypeError("provider context must be a mapping")
    return frozen


def _freeze_context_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_context_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_context_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_context_value(item) for item in value)
    return deepcopy(value)


def _payload_hash(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "QUALITY_VALUES",
    "SCENARIOS",
    "VOICE_LAB_SCHEMA_VERSION",
    "VoiceLabConflict",
    "VoiceLabEvidenceUnavailable",
    "VoiceLabError",
    "VoiceLabNotFound",
    "VoiceLabService",
    "VoiceLabStateError",
    "VoiceLabTrialEvidenceProvider",
    "VoiceLabValidationError",
]
