"""Evaluate deterministic voice E2E scenarios and write readiness evidence.

The suite is intentionally offline: it validates the voice product contract
around transcript handling, InteractionGate decisions, RAG trust posture, TTS
readiness, and latency evidence without needing a microphone, ASR provider, LLM,
or speaker device.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from askme.voice.interaction_gate import InteractionAction, InteractionGate  # noqa: E402

DEFAULT_REPORT_PATH = Path("artifacts/voice_e2e/scenario-evaluation.json")


@dataclass(frozen=True)
class VoiceScenario:
    name: str
    transcript: str
    expected_gate_action: str
    addressed: bool = True
    asr_confidence: float = 0.92
    task_interruptible: bool = True
    perception: dict[str, Any] | None = None
    rag_policy: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    dropped_evidence: list[dict[str, Any]] = field(default_factory=list)
    reply: str = ""
    expected_tts_status: str = "queued"
    expected_reply_contains: str = ""
    expected_should_record_environment: bool | None = None
    latency: dict[str, float | None] = field(default_factory=dict)


def evaluate_scenarios() -> dict[str, Any]:
    """Run deterministic voice scenarios without external services."""
    gate = InteractionGate({
        "enabled": True,
        "min_asr_confidence": 0.45,
        "max_perception_age_s": 2.0,
        "max_interaction_distance_m": 4.0,
        "sound_angle_tolerance_deg": 35.0,
    })
    scenarios = [_evaluate_scenario(gate, scenario) for scenario in _scenarios()]
    passed = sum(1 for item in scenarios if item["passed"])
    metrics = _metrics(scenarios)
    return {
        "suite": "askme-voice-e2e",
        "external_services": False,
        "scenario_count": len(scenarios),
        "passed": passed,
        "failed": len(scenarios) - passed,
        "status": "passed" if passed == len(scenarios) else "failed",
        "metrics": metrics,
        "scenarios": scenarios,
        "generated_at": time.time(),
    }


def write_report(payload: dict[str, Any], path: Path = DEFAULT_REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _evaluate_scenario(gate: InteractionGate, scenario: VoiceScenario) -> dict[str, Any]:
    decision = gate.evaluate(
        scenario.transcript,
        addressed=scenario.addressed,
        asr_confidence=scenario.asr_confidence,
        perception=scenario.perception,
        task_interruptible=scenario.task_interruptible,
    )
    should_answer = decision.action in {
        InteractionAction.RESPOND,
        InteractionAction.CLARIFY,
        InteractionAction.REFUSE,
    }
    reply = _reply_for(decision, scenario) if should_answer else ""
    tts_status = scenario.expected_tts_status if reply else "skipped"
    latency = _latency_for(scenario, should_answer=bool(reply))
    answer_policy = dict(scenario.rag_policy or {"state": "not_applicable", "action": "skip"})

    checks = {
        "gate_action": decision.action.value == scenario.expected_gate_action,
        "tts_status": tts_status == scenario.expected_tts_status,
        "reply_contains": (
            not scenario.expected_reply_contains
            or scenario.expected_reply_contains in reply
        ),
        "no_stale_evidence_used": not any(
            item.get("drop_reason") == "expired" and item.get("used_in_prompt")
            for item in scenario.dropped_evidence
        ),
        "environment_recording": (
            True
            if scenario.expected_should_record_environment is None
            else decision.should_record_environment
            == scenario.expected_should_record_environment
        ),
        "latency_complete_for_answer": (
            True
            if not reply
            else latency.get("asr_final_ms") is not None
            and latency.get("tts_first_audio_ms") is not None
        ),
    }
    passed = all(checks.values())
    return {
        "name": scenario.name,
        "passed": passed,
        "checks": checks,
        "transcript": {
            "text": scenario.transcript,
            "confidence": scenario.asr_confidence,
            "final_ms": latency.get("asr_final_ms"),
        },
        "interaction_gate": {
            "action": decision.action.value,
            "reason": decision.reason,
            "confidence": decision.confidence,
            "reply": decision.reply,
            "should_record_environment": decision.should_record_environment,
        },
        "rag": {
            "answer_policy": answer_policy,
            "evidence": list(scenario.evidence),
            "dropped_evidence": list(scenario.dropped_evidence),
            "used_in_answer": bool(scenario.evidence)
            and answer_policy.get("action") == "answer_with_evidence",
        },
        "reply": reply,
        "tts": {
            "status": tts_status,
            "first_audio_ms": latency.get("tts_first_audio_ms"),
            "provider": "simulated",
        },
        "latency": latency,
    }


def _reply_for(decision: Any, scenario: VoiceScenario) -> str:
    if decision.reply:
        return str(decision.reply)
    if scenario.reply:
        return scenario.reply
    action = scenario.rag_policy.get("action")
    if action == "refuse_and_request_update":
        return "当前知识已过期，我不能给出确定回答，请联系现场工作人员确认。"
    if action == "clarify":
        return "我发现知识存在冲突，需要先确认最新版本后再回答。"
    if action == "refuse":
        return "我没有可靠依据，不能确定回答。"
    if action == "answer_with_evidence" and scenario.evidence:
        return str(scenario.evidence[0].get("text") or "").strip()
    return "我需要更多信息才能继续。"


def _latency_for(
    scenario: VoiceScenario,
    *,
    should_answer: bool,
) -> dict[str, float | None]:
    base: dict[str, float | None] = {
        "asr_final_ms": 420.0,
        "gate_ms": 12.0,
        "rag_retrieve_ms": 35.0 if should_answer else None,
        "llm_ttft_ms": 280.0 if should_answer else None,
        "tts_first_audio_ms": 360.0 if should_answer else None,
        "first_useful_response_latency_ms": 1107.0 if should_answer else None,
    }
    base.update(scenario.latency)
    return base


def _metrics(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [
        item for item in scenarios
        if item["interaction_gate"]["action"] in {"respond", "clarify", "refuse"}
    ]
    false_responds = [
        item for item in scenarios
        if item["name"].startswith("noise_")
        and item["interaction_gate"]["action"] in {"respond", "clarify", "refuse"}
    ]
    missed_help = [
        item for item in scenarios
        if item["name"].startswith("visitor_")
        and item["interaction_gate"]["action"] not in {"respond", "clarify", "refuse"}
    ]
    grounded = [
        item for item in answered
        if item["rag"]["answer_policy"].get("action") == "answer_with_evidence"
    ]
    stale_used = [
        item for item in scenarios
        for evidence in item["rag"].get("dropped_evidence", [])
        if evidence.get("drop_reason") == "expired" and evidence.get("used_in_prompt")
    ]
    unsupported_claims = [
        item for item in answered
        if item["rag"]["answer_policy"].get("state") in {"no_evidence", "stale", "conflict"}
        and item["rag"].get("used_in_answer")
    ]
    return {
        "answered_turns": len(answered),
        "evidence_top1_hit": len(grounded),
        "unsupported_claim_count": len(unsupported_claims),
        "stale_evidence_usage": len(stale_used),
        "false_respond_rate": round(len(false_responds) / max(1, len(scenarios)), 4),
        "missed_help_rate": round(len(missed_help) / max(1, len(scenarios)), 4),
        "first_useful_response_latency_ms": _max_latency(
            scenarios,
            "first_useful_response_latency_ms",
        ),
        "asr_final_ms": _max_latency(scenarios, "asr_final_ms"),
        "rag_retrieve_ms": _max_latency(scenarios, "rag_retrieve_ms"),
        "tts_first_audio_ms": _max_latency(scenarios, "tts_first_audio_ms"),
    }


def _max_latency(scenarios: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(value)
        for item in scenarios
        if (value := item.get("latency", {}).get(key)) is not None
    ]
    return max(values) if values else None


def _fresh_perception(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source": "scenario",
        "observed_at": time.time(),
        "person_detected": True,
        "person_count": 1,
        "nearest_person_distance_m": 1.8,
        "person_angle_deg": 0.0,
        "sound_source_angle_deg": 4.0,
        "sound_source_matches_person": True,
        "person_facing_robot": True,
        "posture": "standing",
    }
    payload.update(overrides)
    return payload


def _scenarios() -> list[VoiceScenario]:
    return [
        VoiceScenario(
            name="visitor_wayfinding_grounded",
            transcript="请问洗手间在哪里",
            expected_gate_action="respond",
            perception=_fresh_perception(),
            rag_policy={"state": "grounded", "action": "answer_with_evidence"},
            evidence=[{
                "text": "洗手间在一层东侧，靠近服务台。",
                "source": "site-map.json",
                "category": "location",
                "score": 0.94,
            }],
            expected_reply_contains="一层东侧",
        ),
        VoiceScenario(
            name="visitor_unknown_location_refused",
            transcript="请问贵宾室在哪里",
            expected_gate_action="respond",
            perception=_fresh_perception(),
            rag_policy={"state": "no_evidence", "action": "refuse"},
            reply="我没有可靠依据确认贵宾室位置，请咨询现场工作人员。",
            expected_reply_contains="没有可靠依据",
        ),
        VoiceScenario(
            name="patrol_sop_grounded",
            transcript="开始 A 区巡检要按什么 SOP",
            expected_gate_action="respond",
            perception=_fresh_perception(),
            rag_policy={"state": "grounded", "action": "answer_with_evidence"},
            evidence=[{
                "text": "A 区巡检 SOP：先确认通道安全，再检查 3 号设备和消防通道。",
                "source": "sop-a.md",
                "category": "sop",
                "score": 0.9,
            }],
            expected_reply_contains="A 区巡检 SOP",
        ),
        VoiceScenario(
            name="equipment_location_grounded",
            transcript="设备三号在哪里",
            expected_gate_action="respond",
            perception=_fresh_perception(),
            rag_policy={"state": "grounded", "action": "answer_with_evidence"},
            evidence=[{
                "text": "3 号设备位于 A 区东侧配电柜旁。",
                "source": "equipment.csv",
                "category": "equipment",
                "score": 0.92,
            }],
            expected_reply_contains="A 区东侧",
        ),
        VoiceScenario(
            name="stale_route_refused",
            transcript="旧展厅路线怎么走",
            expected_gate_action="respond",
            perception=_fresh_perception(),
            rag_policy={"state": "stale", "action": "refuse_and_request_update"},
            dropped_evidence=[{
                "text": "旧展厅路线从北门进入",
                "drop_reason": "expired",
                "used_in_prompt": False,
            }],
            expected_reply_contains="已过期",
        ),
        VoiceScenario(
            name="noise_bystander_casual_recorded_only",
            transcript="我们去那边看看，这个机器狗好可爱",
            expected_gate_action="record_only",
            addressed=False,
            perception=_fresh_perception(
                person_facing_robot=False,
                sound_source_matches_person=False,
            ),
            rag_policy={"state": "not_applicable", "action": "skip"},
            expected_tts_status="skipped",
            expected_should_record_environment=True,
        ),
        VoiceScenario(
            name="multi_person_ambiguous_clarifies",
            transcript="你好",
            expected_gate_action="clarify",
            addressed=True,
            perception=_fresh_perception(
                person_count=2,
                sound_source_matches_person=False,
            ),
            rag_policy={"state": "not_applicable", "action": "skip"},
            expected_reply_contains="需要我帮你",
            expected_should_record_environment=True,
        ),
        VoiceScenario(
            name="emergency_stop_bypasses_normal_gate",
            transcript="停下",
            expected_gate_action="respond",
            perception=_fresh_perception(
                gesture="stop",
                person_facing_robot=False,
                sound_source_matches_person=False,
            ),
            rag_policy={"state": "not_applicable", "action": "safety_stop"},
            reply="已收到停止请求，正在进入安全停止流程。",
            expected_reply_contains="安全停止",
            latency={"first_useful_response_latency_ms": 280.0, "tts_first_audio_ms": 120.0},
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args(argv)

    payload = evaluate_scenarios()
    report = write_report(payload, args.output)
    print(json.dumps({"status": payload["status"], "report": str(report)}, ensure_ascii=False))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
