from __future__ import annotations

from pathlib import Path

from scripts.eval.check_runtime_handoff_readiness import check_readiness


def test_runtime_handoff_readiness_passes_with_scenarios(tmp_path: Path) -> None:
    audit_path = tmp_path / "runtime-audit.jsonl"
    audit_path.write_text('{"event":"ok"}\n', encoding="utf-8")

    payload = check_readiness(
        runtime_profile="sim",
        scenario_report_path=tmp_path / "scenario-evaluation.json",
        readiness_path=tmp_path / "readiness.json",
        audit_path=audit_path,
        require_audit=True,
    )

    names = {item["name"] for item in payload["checks"]}
    assert payload["status"] == "ok"
    assert payload["hardware_dispatch"] is False
    assert payload["failed_checks"] == []
    assert "runtime_voice_turn_endpoint_present" in names
    assert "scenario_evaluation_passed" in names


def test_runtime_handoff_readiness_refuses_lab_profile(tmp_path: Path) -> None:
    payload = check_readiness(
        runtime_profile="lab",
        scenario_report_path=tmp_path / "scenario-evaluation.json",
        readiness_path=tmp_path / "readiness.json",
        audit_path=tmp_path / "missing-audit.jsonl",
        require_audit=False,
    )

    assert payload["status"] == "degraded"
    assert "profile_refuses_unsafe_promotion" in payload["failed_checks"]
