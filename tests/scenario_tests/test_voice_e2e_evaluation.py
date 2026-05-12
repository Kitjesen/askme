from __future__ import annotations

from scripts.eval.evaluate_voice_e2e_scenarios import evaluate_scenarios, write_report


def test_voice_e2e_scenario_evaluation_suite_writes_readiness_report(tmp_path):
    payload = evaluate_scenarios()
    report = write_report(payload, tmp_path / "voice-e2e.json")

    names = {item["name"] for item in payload["scenarios"]}
    assert payload["suite"] == "askme-voice-e2e"
    assert payload["status"] == "passed"
    assert payload["external_services"] is False
    assert payload["failed"] == 0
    assert report.exists()
    assert {
        "visitor_wayfinding_grounded",
        "visitor_unknown_location_refused",
        "patrol_sop_grounded",
        "equipment_location_grounded",
        "stale_route_refused",
        "noise_bystander_casual_recorded_only",
        "multi_person_ambiguous_clarifies",
        "emergency_stop_bypasses_normal_gate",
    }.issubset(names)


def test_voice_e2e_scenario_evidence_metrics_are_customer_readable():
    payload = evaluate_scenarios()
    metrics = payload["metrics"]
    scenarios = {item["name"]: item for item in payload["scenarios"]}

    assert metrics["false_respond_rate"] == 0
    assert metrics["missed_help_rate"] == 0
    assert metrics["unsupported_claim_count"] == 0
    assert metrics["stale_evidence_usage"] == 0
    assert metrics["tts_first_audio_ms"] is not None
    assert metrics["first_useful_response_latency_ms"] is not None

    noise = scenarios["noise_bystander_casual_recorded_only"]
    assert noise["interaction_gate"]["action"] == "record_only"
    assert noise["tts"]["status"] == "skipped"

    ambiguous = scenarios["multi_person_ambiguous_clarifies"]
    assert ambiguous["interaction_gate"]["action"] == "clarify"
    assert ambiguous["tts"]["status"] == "queued"

    emergency = scenarios["emergency_stop_bypasses_normal_gate"]
    assert emergency["interaction_gate"]["action"] == "respond"
    assert emergency["rag"]["answer_policy"]["action"] == "safety_stop"
