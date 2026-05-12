from __future__ import annotations

from pathlib import Path

from scripts.eval.evaluate_robot_scenarios import evaluate_scenarios, write_report


def test_robot_scenario_evaluation_suite_passes_and_writes_report(tmp_path: Path) -> None:
    payload = evaluate_scenarios()
    report = write_report(payload, tmp_path / "scenario-evaluation.json")

    names = {item["name"] for item in payload["scenarios"]}
    assert payload["status"] == "passed"
    assert payload["hardware_dispatch"] is False
    assert payload["failed"] == 0
    assert report.exists()
    assert {
        "happy_path_completed",
        "estop_blocks_and_replans",
        "unauthorized_viewer_blocked",
        "localization_blocked_with_perception_request",
        "voice_runtime_controls_share_state",
        "voice_confirm_cannot_submit_runtime",
        "direct_motor_skill_rejected",
    }.issubset(names)
