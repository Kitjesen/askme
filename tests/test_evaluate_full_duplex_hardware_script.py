from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from askme.voice.diagnostics.full_duplex_hardware import HARDWARE_REPORT_SCHEMA_VERSION
from scripts.eval.evaluate_full_duplex_hardware import main


def _healthy_hardware_status() -> dict[str, object]:
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


def test_interactive_manual_entry_never_claims_product_grade_instrumented(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
voice:
  full_duplex:
    enabled: true
    echo_control: hardware
    echo_control_verified: true
""".strip(),
        encoding="utf-8",
    )
    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps(_healthy_hardware_status()),
        encoding="utf-8",
    )
    output_path = tmp_path / "report.json"
    answers = iter([*["n"] * 20, *["180"] * 20, *["850"] * 20])

    return_code = main(
        [
            "--config",
            str(config_path),
            "--status-source",
            str(status_path),
            "--output",
            str(output_path),
            "--latency-mode",
            "entry",
            "--operator",
            "operator-1",
            "--room",
            "lab-a",
            "--audio-device",
            "speakerphone-1",
            "--audio-driver",
            "WASAPI",
            "--input-device-id",
            "1",
            "--output-device-id",
            "3",
            "--input-sample-rate-hz",
            "16000",
            "--output-sample-rate-hz",
            "16000",
        ],
        input_fn=lambda _prompt: next(answers),
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert return_code == 1
    assert report["schema_version"] == HARDWARE_REPORT_SCHEMA_VERSION
    assert report["status"] == "failed"
    assert report["checks"]["physical_speaker_stop_sample_count"] is False
    assert report["checks"]["physical_first_sound_sample_count"] is False
    assert report["summary"]["speaker_only"]["count"] == 20
    assert report["summary"]["human_overlap"]["count"] == 20
    assert report["summary"]["assistant_response"]["count"] == 20
    assert (
        report["summary"]["assistant_response"]["speech_end_to_manual_first_sound_ms"]["p95"]
        == 850.0
    )
    assert all(
        trial["evidence_kind"] == "manual" for group in report["trials"].values() for trial in group
    )


def test_cli_fails_before_trials_when_echo_control_is_unverified(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
voice:
  full_duplex:
    enabled: true
    echo_control: hardware
    echo_control_verified: false
""".strip(),
        encoding="utf-8",
    )
    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps(_healthy_hardware_status()),
        encoding="utf-8",
    )
    output_path = tmp_path / "failed-report.json"

    return_code = main(
        [
            "--config",
            str(config_path),
            "--status-source",
            str(status_path),
            "--output",
            str(output_path),
        ],
        input_fn=lambda _prompt: (_ for _ in ()).throw(
            AssertionError("preflight must not prompt for trials")
        ),
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert return_code == 1
    assert report["status"] == "failed"
    assert report["preflight"]["errors"] == ["echo_control_unproven"]
    assert report["aborted_reason"] == "preflight_failed"


def test_cli_fails_before_trials_when_health_snapshot_is_incomplete(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
voice:
  full_duplex:
    enabled: true
    echo_control: hardware
    echo_control_verified: true
""".strip(),
        encoding="utf-8",
    )
    status = _healthy_hardware_status()
    status.pop("snapshot_at")
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    output_path = tmp_path / "failed-report.json"

    return_code = main(
        [
            "--config",
            str(config_path),
            "--status-source",
            str(status_path),
            "--output",
            str(output_path),
        ],
        input_fn=lambda _prompt: (_ for _ in ()).throw(
            AssertionError("preflight must not prompt for trials")
        ),
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert return_code == 1
    assert report["status"] == "failed"
    assert report["preflight"]["runtime_reason"] == "snapshot_at_missing"
    assert report["aborted_reason"] == "preflight_failed"
