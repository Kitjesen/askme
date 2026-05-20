import json
from pathlib import Path
from unittest.mock import patch

from askme.skills.audit import SkillAuditLog
from fastapi.testclient import TestClient

import askme.health_server as health_server
from askme.audit import AuditExportService, AuditQueryService, AuditReviewService
from askme.audit.query import AuditPaths, _jsonl_record_count, _read_jsonl
from askme.api.schemas.audit import AuditEventsResponse
from askme.api.schemas.audit import AuditExportResponse
from askme.api.schemas.audit import AuditExportRetryResponse
from askme.api.schemas.audit import AuditExportRetryStatusResponse
from askme.api.schemas.audit import AuditExportsResponse
from askme.api.schemas.audit import AuditReviewSubmitResponse
from askme.api.schemas.audit import AuditReviewsResponse
from askme.health_server import create_health_app


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n",
        encoding="utf-8",
    )


def _runtime_snapshot() -> dict:
    return {"status": "ok", "components": {}}


def test_audit_routes_expose_product_response_schemas() -> None:
    app = create_health_app(lambda: _runtime_snapshot())
    paths = app.openapi()["paths"]

    expected = {
        ("/api/audit/events", "get"): "AuditEventsResponse",
        ("/api/audit/reviews", "get"): "AuditReviewsResponse",
        ("/api/audit/reviews", "post"): "AuditReviewSubmitResponse",
        ("/api/audit/export", "post"): "AuditExportResponse",
        ("/api/audit/exports", "get"): "AuditExportsResponse",
        ("/api/audit/export/retry", "get"): "AuditExportRetryStatusResponse",
        ("/api/audit/export/retry", "post"): "AuditExportRetryResponse",
    }
    for (path, method), schema_name in expected.items():
        schema = paths[path][method]["responses"]["200"]["content"]["application/json"]["schema"]

        assert schema["$ref"].endswith(f"/{schema_name}")


def test_audit_query_unifies_skill_field_and_runtime_records(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    field_path = tmp_path / "field-action-audit.jsonl"
    runtime_path = tmp_path / "runtime-audit.jsonl"
    SkillAuditLog(skill_path).append(
        skill_name="guide_visitor",
        status="approved",
        event_type="governance",
        operator_id="supervisor-1",
        action="approve",
        result_preview="approved wayfinding skill",
    )
    _write_jsonl(
        field_path,
        [
            {
                "kind": "field_event_action",
                "created_at": 10,
                "event_id": "evt-1",
                "audit": {
                    "at": 10,
                    "action": "close",
                    "outcome": "accepted",
                    "operator_id": "security-1",
                    "reason": "handled",
                },
            }
        ],
    )
    _write_jsonl(
        runtime_path,
        [
            {
                "kind": "operator_action",
                "created_at": 20,
                "run_id": "run-1",
                "action": {
                    "action": "pause",
                    "outcome": "accepted",
                    "operator_id": "guard-1",
                    "reason": "visitor in path",
                },
            }
        ],
    )

    payload = AuditQueryService(
        paths=AuditPaths(
            skill_audit=skill_path,
            field_action_audit=field_path,
            runtime_audit=runtime_path,
        )
    ).query(limit=10)

    assert payload["count"] == 3
    assert payload["summary"]["by_source"] == {"field": 1, "runtime": 1, "skill": 1}
    assert {record["source"] for record in payload["records"]} == {"field", "runtime", "skill"}
    assert any(record["action"] == "approve" for record in payload["records"])


def test_audit_query_filters_by_operator(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    SkillAuditLog(skill_path).append(
        skill_name="one",
        status="blocked",
        operator_id="operator-a",
        reason="disabled",
    )
    SkillAuditLog(skill_path).append(
        skill_name="two",
        status="ok",
        operator_id="operator-b",
    )

    payload = AuditQueryService(paths=AuditPaths(skill_audit=skill_path)).query(
        operator_id="operator-a",
    )

    assert payload["count"] == 1
    assert payload["records"][0]["operator_id"] == "operator-a"


def test_audit_query_filters_by_customer_project_scope(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    field_path = tmp_path / "field-action-audit.jsonl"
    _write_jsonl(
        field_path,
        [
            {
                "kind": "field_event_action",
                "created_at": 10,
                "event_id": "evt-demo-project",
                "tenant_id": "default",
                "delivery_namespace": "default",
                "customer_id": "demo-customer",
                "project_id": "demo-field-ops",
                "site_id": "demo-park",
                "managed_object_id": "north-gate-camera",
                "audit": {
                    "at": 10,
                    "action": "acknowledge",
                    "outcome": "accepted",
                    "operator_id": "security-1",
                },
            },
            {
                "kind": "field_event_action",
                "created_at": 20,
                "event_id": "evt-other-project",
                "tenant_id": "default",
                "delivery_namespace": "default",
                "customer_id": "other-customer",
                "project_id": "other-field-ops",
                "site_id": "other-park",
                "managed_object_id": "south-gate-camera",
                "audit": {
                    "at": 20,
                    "action": "acknowledge",
                    "outcome": "accepted",
                    "operator_id": "security-2",
                },
            },
        ],
    )

    payload = AuditQueryService(
        paths=AuditPaths(skill_audit=skill_path, field_action_audit=field_path)
    ).query(project_id="demo-field-ops", managed_object_id="north-gate-camera")

    assert payload["filtered_total"] == 1
    assert payload["filters"]["project_id"] == "demo-field-ops"
    assert payload["filters"]["managed_object_id"] == "north-gate-camera"
    assert payload["records"][0]["resource_id"] == "evt-demo-project"
    assert payload["records"][0]["customer_id"] == "demo-customer"
    assert payload["records"][0]["project_id"] == "demo-field-ops"
    assert payload["records"][0]["site_id"] == "demo-park"
    assert payload["records"][0]["managed_object_id"] == "north-gate-camera"
    assert payload["summary"]["by_customer"] == {"demo-customer": 1}
    assert payload["summary"]["by_project"] == {"demo-field-ops": 1}
    assert payload["summary"]["by_managed_object"] == {"north-gate-camera": 1}
    assert payload["query_engine"]["index_used"] is True


def test_audit_query_adds_customer_review_integrity_and_evidence_fields(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    field_path = tmp_path / "field-action-audit.jsonl"
    _write_jsonl(
        field_path,
        [
            {
                "kind": "field_event_action",
                "created_at": 10,
                "event_id": "evt-risk",
                "sequence": 1,
                "prev_hash": "GENESIS",
                "record_hash": "hash-1",
                "hash_alg": "sha256",
                "record_signature": "signature-1",
                "signature_alg": "hmac-sha256",
                "evidence_media": [{"type": "photo", "path": "artifacts/evidence/risk.jpg"}],
                "audit": {
                    "at": 10,
                    "action": "request_close",
                    "outcome": "denied",
                    "operator_id": "visitor-1",
                    "reason": "operator_not_authorized",
                    "note": "not allowed",
                },
            }
        ],
    )

    payload = AuditQueryService(
        paths=AuditPaths(skill_audit=skill_path, field_action_audit=field_path)
    ).query(limit=10)

    record = payload["records"][0]
    assert record["record_id"] == "field:1"
    assert record["customer_label"] == "现场事件处置审计"
    assert record["display_title"] == "现场事件处置审计：evt-risk 执行申请关闭，结果为已拒绝"
    assert record["customer_impact"] == "field_response_and_evidence"
    assert record["action_label"] == "申请关闭"
    assert record["outcome_label"] == "已拒绝"
    assert record["severity_label"] == "高风险"
    assert record["customer_copy"] == {
        "title": "现场事件处置记录",
        "what_happened": "访客 visitor-1 对现场事件 evt-risk 执行申请关闭，结果为已拒绝。",
        "why_it_matters": "这条记录影响现场事件处理、证据闭环和客户验收。",
        "next_step": "主管需要核对操作权限、现场证据和处理说明。",
        "review_owner": "现场主管",
    }
    assert record["governance"] == {
        "handoff_status": "blocked_until_reviewed",
        "customer_visible": True,
        "exportable": False,
        "requires_supervisor": True,
    }
    assert record["actor_type"] == "visitor"
    assert record["resource_type"] == "field_event"
    assert record["severity"] == "high"
    assert record["requires_review"] is True
    assert record["review_reason"] == "failed_or_high_risk_action"
    assert record["recommended_action"] == (
        "主管需要核对操作权限、现场证据和处理说明。"
    )
    assert record["acceptance"] == {
        "status": "blocked",
        "label": "待复核，不可交付",
        "can_export": False,
        "reason": "该记录仍需主管复核，不能进入客户验收审计包。",
        "next_step": "补齐现场证据并由主管确认",
    }
    assert record["integrity"] == {
        "hash_chain": True,
        "signed": True,
        "hash_alg": "sha256",
        "signature_alg": "hmac-sha256",
        "sequence": 1,
    }
    assert record["evidence_refs"] == [{"label": "photo", "path": "artifacts/evidence/risk.jpg"}]
    assert record["evidence_status"] == {
        "status": "linked",
        "ref_count": 1,
        "labels": {"photo": 1},
    }
    assert payload["summary"]["by_severity"] == {"high": 1}
    assert payload["summary"]["requires_review_count"] == 1
    assert payload["product_summary"]["status"] == "needs_review"
    assert payload["product_summary"]["customer_status_label"] == "待主管复核"
    assert payload["product_summary"]["customer_status"] == (
        "存在待复核记录，暂不能交付给客户验收"
    )
    assert payload["product_summary"]["handoff"]["customer_ready"] is False
    assert payload["product_summary"]["high_or_critical_count"] == 1
    assert payload["product_summary"]["integrity"]["signed_record_count"] == 1
    assert payload["customer_report"]["title"] == "AskMe 产品审计报告"
    assert payload["customer_report"]["status_label"] == "待主管复核"
    assert payload["customer_report"]["handoff_brief"] == {
        "claim": "本审计包尚不能交付客户验收。",
        "customer_message": "仍有 1 条记录需要主管确认处理依据。",
        "delivery_owner": "现场主管",
        "next_step": "先完成待复核记录，再重新生成审计包。",
    }
    assert payload["delivery_dossier"]["title"] == "客户交付审计档案"
    assert payload["delivery_dossier"]["decision"] == "blocked"
    assert payload["delivery_dossier"]["decision_label"] == "验收前存在阻断项"
    assert payload["delivery_dossier"]["customer_claim"] == "当前审计范围仍有阻断项，暂不能作为客户验收证据。"
    assert payload["delivery_dossier"]["allowed_uses"] == [
        "内部复核材料",
        "问题诊断材料",
        "证据缺口清单",
    ]
    assert "无人值守生产上线声明" in payload["delivery_dossier"]["blocked_uses"]
    assert payload["delivery_dossier"]["handoff_owner"] == "现场主管"
    assert payload["delivery_dossier"]["record_scope"] == {
        "record_count": 1,
        "source_counts": {"field": 1},
        "source_labels": {"field": "现场事件处置审计"},
        "high_or_critical_count": 1,
        "evidence_linked_count": 1,
        "review_required_count": 1,
    }
    assert payload["customer_report"]["acceptance_summary"] == {
        "status": "blocked",
        "status_label": "需先处理阻断项",
        "customer_message": "仍有 1 个必要验收项未通过，暂不能提交客户验收。",
        "required_blocker_count": 1,
        "warning_count": 1,
        "passed_count": 3,
        "total_count": 5,
        "blocked_item_ids": ["supervisor_review"],
        "warning_item_ids": ["risk_visibility"],
        "next_step": "优先处理 supervisor_review、source_health 或 review_log_integrity 阻断项。",
    }
    checklist = {
        item["id"]: item for item in payload["customer_report"]["acceptance_checklist"]
    }
    assert checklist["supervisor_review"]["status"] == "blocked"
    assert checklist["source_health"]["status"] == "passed"
    assert checklist["review_log_integrity"]["status"] == "passed"
    assert checklist["evidence_links"]["status"] == "passed"
    assert checklist["risk_visibility"]["status"] == "warning"
    assert payload["customer_report"]["sections"]["review"]["pending_count"] == 1
    assert payload["customer_report"]["sections"]["source_health"]["source_labels"][
        "field_action_audit"
    ] == "现场处置动作日志"
    assert payload["audit_readiness"]["status"] == "blocked"
    assert payload["audit_readiness"]["status_label"] == "阻断交付"
    assert payload["audit_readiness"]["review_required_count"] == 1
    assert payload["audit_readiness"]["blockers"] == [
        {"reason": "supervisor_review_required", "count": 1}
    ]
    assert payload["audit_readiness"]["blocker_labels"] == ["存在待主管复核记录"]
    assert "主管需要处理待复核队列并给出复核结论。" in payload["audit_readiness"]["next_actions"]
    assert payload["review_queue"][0]["record_id"] == "field:1"
    assert payload["source_health"]["field_action_audit"]["record_count"] == 1


def test_audit_query_extracts_field_archive_event_evidence(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    archive_path = tmp_path / "events.jsonl"
    _write_jsonl(
        archive_path,
        [
            {
                "event_id": "evt-smoke",
                "event_type": "smoke_detected",
                "image_path": "artifacts/evidence/smoke.jpg",
                "report_path": "artifacts/reports/smoke-report.json",
                "payload": {
                    "video_path": "artifacts/evidence/smoke.mp4",
                    "evidence_media": [
                        {"type": "thermal", "path": "artifacts/evidence/thermal.png"},
                    ],
                },
                "action_audit": [
                    {
                        "at": 10,
                        "action": "notify",
                        "outcome": "accepted",
                        "operator_id": "robot.sensor",
                        "reason": "smoke_detected",
                    },
                ],
            },
        ],
    )

    payload = AuditQueryService(
        paths=AuditPaths(skill_audit=skill_path, field_event_archive=archive_path)
    ).query(limit=10)

    refs = payload["records"][0]["evidence_refs"]
    assert {"label": "photo", "path": "artifacts/evidence/smoke.jpg"} in refs
    assert {"label": "report", "path": "artifacts/reports/smoke-report.json"} in refs
    assert {"label": "video", "path": "artifacts/evidence/smoke.mp4"} in refs
    assert {"label": "thermal", "path": "artifacts/evidence/thermal.png"} in refs


def test_audit_review_decision_clears_review_queue_without_mutating_source(
    tmp_path: Path,
) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    field_path = tmp_path / "field-action-audit.jsonl"
    review_path = tmp_path / "audit-reviews.jsonl"
    _write_jsonl(
        field_path,
        [
            {
                "kind": "field_event_action",
                "sequence": 1,
                "event_id": "evt-denied",
                "audit": {
                    "at": 10,
                    "action": "close",
                    "outcome": "denied",
                    "operator_id": "guard-1",
                    "reason": "supervisor_not_authorized",
                },
            }
        ],
    )
    service = AuditQueryService(
        paths=AuditPaths(
            skill_audit=skill_path,
            field_action_audit=field_path,
            audit_reviews=review_path,
        )
    )
    before = service.query()
    record_id = before["review_queue"][0]["record_id"]

    decision = AuditReviewService(path=review_path).submit(
        record_id=record_id,
        reviewer_id="supervisor-1",
        decision="waived",
        note="duplicate denial already handled",
        created_at=20,
    )
    after = service.query()

    assert decision["ok"] is True
    assert before["product_summary"]["requires_review_count"] == 1
    assert after["review_integrity"]["valid"] is True
    assert after["review_integrity"]["checked_count"] == 1
    assert after["product_summary"]["requires_review_count"] == 0
    assert after["review_queue"] == []
    assert after["records"][0]["requires_review"] is False
    assert after["records"][0]["review_status"] == "cleared"
    assert after["records"][0]["review_decision"]["reviewer_id"] == "supervisor-1"
    assert after["records"][0]["review_decision"]["decision_label"] == "复核豁免"
    assert after["records"][0]["review_decision"]["customer_effect"] == (
        "该复核决定解除交付阻断，记录可进入客户验收审计包。"
    )
    assert after["records"][0]["review_decision"]["next_step"] == "重新生成审计包并随验收材料归档。"
    assert after["records"][0]["review_reason"] == ""
    assert after["records"][0]["governance"]["handoff_status"] == "ready"
    assert after["records"][0]["governance"]["exportable"] is True
    assert after["records"][0]["customer_copy"]["review_owner"] == "无需复核"
    assert after["records"][0]["customer_copy"]["next_step"] == "无需复核，保留在审计时间线中备查。"
    assert after["records"][0]["recommended_action"] == (
        "无需复核，保留在审计时间线中备查。"
    )
    assert after["records"][0]["acceptance"] == {
        "status": "ready",
        "label": "可归档",
        "can_export": True,
        "reason": "该记录无需复核，可保留在审计时间线中。",
        "next_step": "保留记录用于追溯",
    }
    assert "supervisor_not_authorized" in field_path.read_text(encoding="utf-8")


def test_audit_review_log_integrity_detects_tampering(tmp_path: Path) -> None:
    review_path = tmp_path / "audit-reviews.jsonl"
    service = AuditReviewService(path=review_path)
    service.submit(
        record_id="field:1",
        reviewer_id="supervisor-1",
        decision="accepted",
        note="ok",
        created_at=10,
    )
    lines = review_path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["decision"] = "waived"
    review_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")

    payload = service.integrity()

    assert payload["valid"] is False
    assert payload["failures"][0]["reason"] == "record_hash_mismatch"


def test_read_jsonl_streams_tail_limit_over_valid_records_and_skips_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "old-1"}),
                "{bad json",
                json.dumps({"id": "old-2"}),
                json.dumps(["not", "a", "record"]),
                json.dumps({"id": "tail-1"}),
                json.dumps({"id": "tail-2"}),
                json.dumps({"id": "tail-3"}),
                "not json either",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with patch.object(Path, "read_text", side_effect=AssertionError("read_text should not be used")):
        records = _read_jsonl(path, limit=3)

    assert [record["id"] for record in records] == ["tail-1", "tail-2", "tail-3"]


def test_jsonl_record_count_streams_non_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "one"}),
                "",
                "{bad json",
                json.dumps({"id": "two"}),
                "   ",
            ]
        ),
        encoding="utf-8",
    )

    with patch.object(Path, "read_text", side_effect=AssertionError("read_text should not be used")):
        count = _jsonl_record_count(path)

    assert count == 3


def test_audit_query_source_health_reports_invalid_jsonl_records(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    field_path = tmp_path / "field-action-audit.jsonl"
    field_path.write_text(
        "\n".join(
            [
                json.dumps({
                    "kind": "field_event_action",
                    "created_at": 10,
                    "event_id": "evt-ok",
                    "audit": {"at": 10, "action": "close", "outcome": "accepted"},
                }),
                "{bad json",
                json.dumps(["not", "an", "object"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = AuditQueryService(
        paths=AuditPaths(skill_audit=skill_path, field_action_audit=field_path)
    ).query()

    health = payload["source_health"]["field_action_audit"]
    assert payload["count"] == 1
    assert health["record_count"] == 3
    assert health["valid_record_count"] == 1
    assert health["invalid_record_count"] == 2
    assert health["readable"] is True
    assert payload["audit_readiness"]["status"] == "blocked"
    assert payload["audit_readiness"]["blockers"][0]["reason"] == "audit_source_has_invalid_records"


def test_audit_query_reads_runtime_handoff_audit_path_from_config(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    runtime_path = tmp_path / "runtime-handoff-audit.jsonl"
    _write_jsonl(
        runtime_path,
        [
            {
                "kind": "operator_action",
                "created_at": 20,
                "run_id": "run-runtime-1",
                "action": {
                    "action": "pause",
                    "outcome": "accepted",
                    "operator_id": "guard-1",
                },
            }
        ],
    )

    payload = AuditQueryService(
        {
            "skills": {"audit_path": str(skill_path)},
            "runtime_handoff": {"audit": {"path": str(runtime_path)}},
        }
    ).query(limit=10)

    assert payload["count"] == 1
    assert payload["records"][0]["source"] == "runtime"
    assert payload["source_health"]["runtime_audit"]["path"] == str(runtime_path)


def test_audit_query_filters_by_time_window_and_reports_truncation(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    field_path = tmp_path / "field-action-audit.jsonl"
    _write_jsonl(
        field_path,
        [
            {
                "kind": "field_event_action",
                "created_at": 10,
                "event_id": "evt-old",
                "audit": {"at": 10, "action": "acknowledge", "outcome": "accepted"},
            },
            {
                "kind": "field_event_action",
                "created_at": 20,
                "event_id": "evt-mid",
                "audit": {"at": 20, "action": "acknowledge", "outcome": "accepted"},
            },
            {
                "kind": "field_event_action",
                "created_at": 30,
                "event_id": "evt-new",
                "audit": {"at": 30, "action": "acknowledge", "outcome": "accepted"},
            },
        ],
    )

    payload = AuditQueryService(
        paths=AuditPaths(skill_audit=skill_path, field_action_audit=field_path)
    ).query(limit=1, since="15", until="35")

    assert payload["filtered_total"] == 2
    assert payload["count"] == 1
    assert payload["truncated"] is True
    assert payload["omitted_record_count"] == 1
    assert payload["records"][0]["resource_id"] == "evt-new"
    assert payload["time_window"]["valid"] is True


def test_audit_query_uses_index_for_exact_filters(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    field_path = tmp_path / "field-action-audit.jsonl"
    records = []
    for index in range(120):
        targeted = index % 20 == 0
        records.append({
            "kind": "field_event_action",
            "created_at": index,
            "event_id": f"evt-{index}",
            "audit": {
                "at": index,
                "action": "close" if targeted else "acknowledge",
                "outcome": "accepted",
                "operator_id": "security-target" if targeted else f"security-{index % 7}",
                "reason": "indexed-filter-test",
            },
        })
    _write_jsonl(field_path, records)

    payload = AuditQueryService(
        paths=AuditPaths(skill_audit=skill_path, field_action_audit=field_path)
    ).query(
        limit=20,
        source="field",
        operator_id="security-target",
        action="close",
        outcome="accepted",
    )

    assert payload["count"] == 6
    assert payload["filtered_total"] == 6
    assert payload["query_engine"]["indexed"] is True
    assert payload["query_engine"]["index_used"] is True
    assert payload["query_engine"]["scanned_records"] == 6
    assert payload["query_engine"]["scan_avoidance_records"] >= 100


def test_audit_query_reuses_index_until_jsonl_changes(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    field_path = tmp_path / "field-action-audit.jsonl"
    _write_jsonl(
        field_path,
        [
            {
                "kind": "field_event_action",
                "created_at": 1,
                "event_id": "evt-1",
                "audit": {
                    "at": 1,
                    "action": "close",
                    "outcome": "accepted",
                    "operator_id": "security-target",
                },
            }
        ],
    )
    service = AuditQueryService(
        paths=AuditPaths(skill_audit=skill_path, field_action_audit=field_path)
    )

    first = service.query(operator_id="security-target")
    second = service.query(operator_id="security-target")
    with field_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "kind": "field_event_action",
            "created_at": 2,
            "event_id": "evt-2",
            "audit": {
                "at": 2,
                "action": "close",
                "outcome": "accepted",
                "operator_id": "security-target",
            },
        }, ensure_ascii=False) + "\n")
    third = service.query(operator_id="security-target")

    assert first["filtered_total"] == 1
    assert first["query_engine"]["cache_hits"] == 0
    assert second["filtered_total"] == 1
    assert second["query_engine"]["cache_hits"] >= 1
    assert third["filtered_total"] == 2
    assert third["query_engine"]["cache_misses"] >= 1


def test_audit_events_endpoint_requires_and_uses_rbac(tmp_path: Path, monkeypatch) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    SkillAuditLog(skill_path).append(
        skill_name="guide_visitor",
        status="ok",
        operator_id="dashboard.operator",
    )
    config = {
        "skills": {"audit_path": str(skill_path)},
        "field_operations": {
            "operator_directory": {
                "mode": "demo_config",
                "identity_provider": "local_config",
                "permissions": {"operator": ["audit:read"]},
            },
            "operators": {"dashboard.operator": {"roles": ["operator"]}},
        },
    }
    monkeypatch.setattr(health_server, "get_config", lambda: config)
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    denied = client.get("/api/audit/events", params={"actor_id": "ghost.operator"})
    allowed = client.get("/api/audit/events", params={"actor_id": "dashboard.operator"})

    assert denied.status_code == 403
    assert allowed.status_code == 200
    allowed_payload = AuditEventsResponse.model_validate(allowed.json())
    assert allowed_payload.records[0]["source"] == "skill"


def test_audit_events_endpoint_applies_operator_project_scope(tmp_path: Path, monkeypatch) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    field_path = tmp_path / "field-action-audit.jsonl"
    _write_jsonl(
        field_path,
        [
            {
                "kind": "field_event_action",
                "created_at": 10,
                "event_id": "evt-demo-project",
                "customer_id": "demo-customer",
                "project_id": "demo-field-ops",
                "site_id": "demo-park",
                "audit": {"at": 10, "action": "acknowledge", "outcome": "accepted"},
            },
            {
                "kind": "field_event_action",
                "created_at": 20,
                "event_id": "evt-other-project",
                "customer_id": "other-customer",
                "project_id": "other-field-ops",
                "site_id": "other-park",
                "audit": {"at": 20, "action": "acknowledge", "outcome": "accepted"},
            },
        ],
    )
    config = {
        "skills": {"audit_path": str(skill_path)},
        "field_operations": {
            "action_audit": {"path": str(field_path)},
            "operator_directory": {
                "mode": "demo_config",
                "identity_provider": "local_config",
                "permissions": {"operator": ["audit:read"]},
            },
            "operators": {
                "project.operator": {
                    "roles": ["operator"],
                    "project_scope": {
                        "tenant_ids": ["default"],
                        "delivery_namespaces": ["default"],
                        "customer_ids": ["demo-customer"],
                        "project_ids": ["demo-field-ops"],
                        "site_ids": ["demo-park"],
                    },
                }
            },
        },
    }
    monkeypatch.setattr(health_server, "get_config", lambda: config)
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    scoped = client.get("/api/audit/events", params={"actor_id": "project.operator"})
    forbidden = client.get(
        "/api/audit/events",
        params={"actor_id": "project.operator", "project_id": "other-field-ops"},
    )

    assert scoped.status_code == 200
    payload = scoped.json()
    assert payload["filtered_total"] == 1
    assert payload["filters"]["customer_id"] == "demo-customer"
    assert payload["filters"]["project_id"] == "demo-field-ops"
    assert payload["filters"]["site_id"] == "demo-park"
    assert payload["records"][0]["resource_id"] == "evt-demo-project"
    assert forbidden.status_code == 403
    assert forbidden.json()["reason"] == "project_scope_not_allowed"


def test_audit_routes_support_cors_preflight(monkeypatch) -> None:
    monkeypatch.setattr(health_server, "get_config", lambda: {})
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    events = client.options("/api/audit/events")
    reviews = client.options("/api/audit/reviews")
    export = client.options("/api/audit/export")
    exports = client.options("/api/audit/exports")
    retry = client.options("/api/audit/export/retry")

    assert events.status_code == 204
    assert events.headers["access-control-allow-methods"] == "GET, OPTIONS"
    assert reviews.status_code == 204
    assert "POST" in reviews.headers["access-control-allow-methods"]
    assert export.status_code == 204
    assert export.headers["access-control-allow-methods"] == "POST, OPTIONS"
    assert exports.status_code == 204
    assert exports.headers["access-control-allow-methods"] == "GET, OPTIONS"
    assert retry.status_code == 204
    assert "X-Askme-Operator-Id" in retry.headers["access-control-allow-headers"]


def test_audit_events_endpoint_filters_by_time_window(tmp_path: Path, monkeypatch) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    field_path = tmp_path / "field-action-audit.jsonl"
    _write_jsonl(
        field_path,
        [
            {
                "kind": "field_event_action",
                "created_at": 10,
                "event_id": "evt-before",
                "audit": {"at": 10, "action": "acknowledge", "outcome": "accepted"},
            },
            {
                "kind": "field_event_action",
                "created_at": 20,
                "event_id": "evt-window",
                "audit": {"at": 20, "action": "acknowledge", "outcome": "accepted"},
            },
        ],
    )
    config = {
        "skills": {"audit_path": str(skill_path)},
        "field_operations": {
            "action_audit": {"path": str(field_path)},
            "operator_directory": {
                "mode": "demo_config",
                "identity_provider": "local_config",
                "permissions": {"operator": ["audit:read"]},
            },
            "operators": {"dashboard.operator": {"roles": ["operator"]}},
        },
    }
    monkeypatch.setattr(health_server, "get_config", lambda: config)
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    response = client.get(
        "/api/audit/events",
        params={"actor_id": "dashboard.operator", "since": "15", "until": "25"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filtered_total"] == 1
    assert payload["records"][0]["resource_id"] == "evt-window"
    assert payload["time_window"]["valid"] is True


def test_audit_review_endpoint_requires_permission_and_clears_record(
    tmp_path: Path,
    monkeypatch,
) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    field_path = tmp_path / "field-action-audit.jsonl"
    review_path = tmp_path / "audit-reviews.jsonl"
    _write_jsonl(
        field_path,
        [
            {
                "kind": "field_event_action",
                "sequence": 1,
                "event_id": "evt-denied",
                "audit": {
                    "at": 10,
                    "action": "close",
                    "outcome": "denied",
                    "operator_id": "guard-1",
                    "reason": "supervisor_not_authorized",
                },
            }
        ],
    )
    config = {
        "skills": {"audit_path": str(skill_path)},
        "audit": {"review": {"path": str(review_path)}},
        "field_operations": {
            "action_audit": {"path": str(field_path)},
            "operator_directory": {
                "mode": "demo_config",
                "identity_provider": "local_config",
                "permissions": {
                    "operator": ["audit:read", "field:project:read"],
                    "supervisor": ["audit:read", "audit:review", "field:project:read"],
                },
            },
            "operators": {
                "dashboard.operator": {"roles": ["operator"]},
                "supervisor-1": {"roles": ["supervisor"]},
            },
        },
    }
    monkeypatch.setattr(health_server, "get_config", lambda: config)
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))
    unresolved = client.get("/api/audit/events", params={"actor_id": "dashboard.operator"}).json()
    record_id = unresolved["review_queue"][0]["record_id"]

    denied = client.post(
        "/api/audit/reviews",
        json={
            "operator_id": "dashboard.operator",
            "record_id": record_id,
            "decision": "waived",
            "note": "handled elsewhere",
        },
    )
    allowed = client.post(
        "/api/audit/reviews",
        json={
            "operator_id": "supervisor-1",
            "record_id": record_id,
            "decision": "waived",
            "note": "handled elsewhere",
        },
    )
    missing = client.post(
        "/api/audit/reviews",
        json={
            "operator_id": "supervisor-1",
            "record_id": "field:missing",
            "decision": "waived",
            "note": "should not be accepted",
        },
    )
    resolved = client.get("/api/audit/events", params={"actor_id": "dashboard.operator"}).json()

    assert denied.status_code == 403
    assert missing.status_code == 404
    assert missing.json()["reason"] == "audit_record_not_found"
    assert allowed.status_code == 200
    allowed_payload = AuditReviewSubmitResponse.model_validate(allowed.json())
    assert allowed_payload.ok is True
    assert allowed.json()["record"]["clears_review"] is True
    assert resolved["product_summary"]["requires_review_count"] == 0
    assert resolved["records"][0]["review_status"] == "cleared"


def test_audit_reviews_endpoint_lists_history_with_integrity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    review_path = tmp_path / "audit-reviews.jsonl"
    AuditReviewService(path=review_path).submit(
        record_id="field:1",
        reviewer_id="supervisor-1",
        decision="accepted",
        note="handled",
        created_at=20,
    )
    config = {
        "audit": {"review": {"path": str(review_path)}},
        "field_operations": {
            "operator_directory": {
                "mode": "demo_config",
                "identity_provider": "local_config",
                "permissions": {
                    "operator": ["audit:read", "field:project:read"],
                    "supervisor": ["audit:read", "audit:review", "field:project:read"],
                },
            },
            "operators": {
                "dashboard.operator": {"roles": ["operator"]},
                "ghost.operator": {"roles": ["unknown"]},
            },
        },
    }
    monkeypatch.setattr(health_server, "get_config", lambda: config)
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    denied = client.get("/api/audit/reviews", params={"actor_id": "ghost.operator"})
    allowed = client.get("/api/audit/reviews", params={"actor_id": "dashboard.operator"})

    assert denied.status_code == 403
    assert allowed.status_code == 200
    payload = allowed.json()
    schema_payload = AuditReviewsResponse.model_validate(payload)
    assert schema_payload.count == 1
    assert payload["records"][0]["record_id"] == "field:1"
    assert payload["records"][0]["decision"] == "accepted"
    assert payload["integrity"]["valid"] is True
    assert payload["integrity"]["checked_count"] == 1


def test_unified_audit_endpoint_surfaces_field_archive_evidence_and_review_flow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    archive_path = tmp_path / "events.jsonl"
    review_path = tmp_path / "audit-reviews.jsonl"
    evidence_path = Path("artifacts/evidence/audit-unit-evidence.txt")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text("audit-evidence", encoding="utf-8")
    _write_jsonl(
        archive_path,
        [
            {
                "event_id": "evt-evidence",
                "scenario_id": "fire_or_smoke",
                "image_path": str(evidence_path).replace("\\", "/"),
                "action_audit": [
                    {
                        "at": 10,
                        "action": "close",
                        "outcome": "denied",
                        "operator_id": "guard-1",
                        "reason": "supervisor_not_authorized",
                    },
                ],
            },
        ],
    )
    config = {
        "skills": {"audit_path": str(skill_path)},
        "audit": {"review": {"path": str(review_path)}},
        "field_operations": {
            "archive_path": str(archive_path),
            "operator_directory": {
                "mode": "demo_config",
                "identity_provider": "local_config",
                "permissions": {
                    "operator": ["audit:read", "field:project:read"],
                    "supervisor": ["audit:read", "audit:review", "field:project:read"],
                },
            },
            "operators": {
                "dashboard.operator": {"roles": ["operator"]},
                "supervisor-1": {"roles": ["supervisor"]},
            },
        },
    }
    monkeypatch.setattr(health_server, "get_config", lambda: config)
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    try:
        events = client.get("/api/audit/events", params={"actor_id": "dashboard.operator"})
        record = events.json()["review_queue"][0]
        evidence = client.get("/api/field/evidence", params={"path": record["evidence_refs"][0]["path"]})
        review = client.post(
            "/api/audit/reviews",
            json={
                "operator_id": "supervisor-1",
                "record_id": record["record_id"],
                "decision": "accepted",
                "note": "verified field evidence and closure authority",
            },
        )
        cleared = client.get("/api/audit/events", params={"actor_id": "dashboard.operator"})

        assert events.status_code == 200
        assert record["resource_id"] == "evt-evidence"
        assert record["evidence_refs"][0] == {
            "label": "photo",
            "path": str(evidence_path).replace("\\", "/"),
        }
        assert evidence.status_code == 200
        assert evidence.text == "audit-evidence"
        assert review.status_code == 200
        assert review.json()["record"]["clears_review"] is True
        assert cleared.json()["review_queue"] == []
        assert cleared.json()["records"][0]["review_status"] == "cleared"
    finally:
        evidence_path.unlink(missing_ok=True)


def test_audit_export_writes_signed_manifest_and_jsonl(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    review_path = tmp_path / "audit-reviews.jsonl"
    SkillAuditLog(skill_path).append(
        skill_name="guide_visitor",
        status="approved",
        operator_id="supervisor-1",
        action="approve",
    )
    config = {
        "skills": {"audit_path": str(skill_path)},
        "audit": {
            "review": {"path": str(review_path)},
            "export": {
                "output_dir": str(tmp_path / "exports"),
                "hmac_secret": "export-secret",
                "signature_key_id": "unit-test-key",
            }
        },
    }

    payload = AuditExportService(config).create_export(actor_id="supervisor-1")

    assert payload["ok"] is True
    export = payload["export"]
    assert export["record_count"] == 1
    assert export["sha256"]
    assert export["signature"]
    assert export["signature_key_id"] == "unit-test-key"
    assert export["product_summary"]["record_count"] == 1
    assert export["customer_report"]["status_label"] == "可交付"
    assert export["audit_readiness"]["status"] == "ready"
    assert export["delivery_dossier"]["decision"] == "ready"
    assert export["delivery_dossier"]["decision_label"] == "可进入客户验收"
    assert export["delivery_dossier"]["customer_claim"] == (
        "已审计的操作、证据和复核记录可用于客户验收、事件复盘和责任追溯。"
    )
    assert export["customer_package"]["package_name"] == "AskMe 客户验收审计包"
    assert export["customer_package"]["acceptance_label"] == "可提交客户验收"
    assert export["customer_package"]["delivery_mode"] == "local_archive"
    assert export["customer_package"]["delivery_contract"]["allowed_uses"] == [
        "客户验收材料",
        "试点复盘材料",
        "事件闭环材料",
        "责任追溯材料",
    ]
    assert (
        "替代现场验收结果"
        in export["customer_package"]["delivery_contract"]["blocked_uses"]
    )
    assert export["customer_package"]["acceptance_summary"]["status"] == "ready"
    assert export["customer_package"]["acceptance_summary"]["status_label"] == "可进入客户验收"
    assert export["customer_package"]["files"][0]["label"] == "审计记录 JSONL"
    assert export["review_queue_count"] == 0
    assert export["review_decision_count"] == 0
    assert export["evidence_summary"]["ready"] is True
    assert export["evidence_summary"]["ref_count"] == 0
    assert export["review_integrity"]["valid"] is True
    assert export["source_health"]["skill_audit"]["exists"] is True
    assert Path(export["records_path"]).exists()
    assert Path(export["manifest_path"]).exists()


def test_audit_export_service_preserves_project_scope_filters(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    field_path = tmp_path / "field-action-audit.jsonl"
    _write_jsonl(
        field_path,
        [
            {
                "kind": "field_event_action",
                "created_at": 10,
                "event_id": "evt-demo-project",
                "customer_id": "demo-customer",
                "project_id": "demo-field-ops",
                "site_id": "demo-park",
                "audit": {"at": 10, "action": "acknowledge", "outcome": "accepted"},
            },
            {
                "kind": "field_event_action",
                "created_at": 20,
                "event_id": "evt-other-project",
                "customer_id": "other-customer",
                "project_id": "other-field-ops",
                "site_id": "other-park",
                "audit": {"at": 20, "action": "acknowledge", "outcome": "accepted"},
            },
        ],
    )

    payload = AuditExportService(
        {
            "skills": {"audit_path": str(skill_path)},
            "field_operations": {"action_audit": {"path": str(field_path)}},
            "audit": {"export": {"output_dir": str(tmp_path / "exports")}},
        }
    ).create_export(
        actor_id="supervisor-1",
        customer_id="demo-customer",
        project_id="demo-field-ops",
        site_id="demo-park",
    )

    export = payload["export"]
    assert export["record_count"] == 1
    assert export["filters"]["customer_id"] == "demo-customer"
    assert export["filters"]["project_id"] == "demo-field-ops"
    assert export["filters"]["site_id"] == "demo-park"
    records = Path(export["records_path"]).read_text(encoding="utf-8")
    assert "evt-demo-project" in records
    assert "evt-other-project" not in records


def test_audit_export_service_lists_recent_manifests(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    SkillAuditLog(skill_path).append(
        skill_name="guide_visitor",
        status="approved",
        operator_id="supervisor-1",
        action="approve",
    )
    service = AuditExportService(
        {
            "skills": {"audit_path": str(skill_path)},
            "audit": {"export": {"output_dir": str(tmp_path / "exports")}},
        }
    )

    created = service.create_export(actor_id="supervisor-1")
    (tmp_path / "exports" / "bad.manifest.json").write_text("{bad json", encoding="utf-8")
    history = service.list_exports()

    assert history["count"] == 1
    assert history["total"] == 1
    assert history["invalid"] == 1
    assert history["exports"][0]["export_id"] == created["export"]["export_id"]
    assert history["exports"][0]["manifest_path"] == created["export"]["manifest_path"]
    assert history["exports"][0]["evidence_summary"]["ready"] is True
    assert history["exports"][0]["customer_report"]["customer_ready"] is True
    assert history["exports"][0]["audit_readiness"]["status"] == "ready"
    assert history["exports"][0]["customer_package"]["acceptance_status"] == "ready"


def test_audit_export_manifest_summarizes_evidence_availability(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    field_path = tmp_path / "field-action-audit.jsonl"
    evidence_path = Path("artifacts/evidence/export-unit-evidence.txt")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text("export-evidence", encoding="utf-8")
    missing_path = "artifacts/evidence/export-missing-evidence.txt"
    _write_jsonl(
        field_path,
        [
            {
                "kind": "field_event_action",
                "created_at": 10,
                "event_id": "evt-export-evidence",
                "evidence_media": [
                    {"type": "photo", "path": str(evidence_path).replace("\\", "/")},
                    {"type": "report", "path": missing_path},
                    {"type": "remote", "path": "https://example.com/evidence.jpg"},
                ],
                "audit": {
                    "at": 10,
                    "action": "notify",
                    "outcome": "accepted",
                    "operator_id": "robot.sensor",
                },
            },
        ],
    )

    try:
        payload = AuditExportService(
            {
                "skills": {"audit_path": str(skill_path)},
                "field_operations": {"action_audit": {"path": str(field_path)}},
                "audit": {"export": {"output_dir": str(tmp_path / "exports")}},
            }
        ).create_export(actor_id="supervisor-1")

        summary = payload["export"]["evidence_summary"]
        assert summary["ref_count"] == 3
        assert summary["local_available_count"] == 1
        assert summary["local_missing_count"] == 1
        assert summary["remote_count"] == 1
        assert summary["labels"] == {"photo": 1, "remote": 1, "report": 1}
        assert summary["ready"] is False
    finally:
        evidence_path.unlink(missing_ok=True)


def test_audit_export_includes_review_decision_evidence(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    field_path = tmp_path / "field-action-audit.jsonl"
    review_path = tmp_path / "audit-reviews.jsonl"
    _write_jsonl(
        field_path,
        [
            {
                "kind": "field_event_action",
                "sequence": 1,
                "event_id": "evt-denied",
                "audit": {
                    "at": 10,
                    "action": "close",
                    "outcome": "denied",
                    "operator_id": "guard-1",
                    "reason": "supervisor_not_authorized",
                },
            }
        ],
    )
    record_id = AuditQueryService(
        paths=AuditPaths(
            skill_audit=skill_path,
            field_action_audit=field_path,
            audit_reviews=review_path,
        )
    ).query()["review_queue"][0]["record_id"]
    AuditReviewService(path=review_path).submit(
        record_id=record_id,
        reviewer_id="supervisor-1",
        decision="accepted",
        note="handled",
        created_at=20,
    )
    service = AuditExportService(
        {
            "skills": {"audit_path": str(skill_path)},
            "field_operations": {"action_audit": {"path": str(field_path)}},
            "audit": {
                "review": {"path": str(review_path)},
                "export": {"output_dir": str(tmp_path / "exports")},
            },
        }
    )

    payload = service.create_export(actor_id="supervisor-1")
    export = payload["export"]
    records_text = Path(export["records_path"]).read_text(encoding="utf-8")

    assert export["review_queue_count"] == 0
    assert export["review_decision_count"] == 1
    assert export["review_integrity"]["checked_count"] == 1
    assert '"review_decision"' in records_text
    assert "handled" in records_text


def test_audit_export_delivery_posts_manifest(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    SkillAuditLog(skill_path).append(skill_name="guide_visitor", status="ok")
    seen = {}

    def fake_post(url: str, body: dict, headers: dict, timeout_s: float) -> dict:
        seen["url"] = url
        seen["body"] = body
        seen["headers"] = headers
        seen["timeout_s"] = timeout_s
        return {"sent": True, "status_code": 200}

    payload = AuditExportService(
        {
            "skills": {"audit_path": str(skill_path)},
            "audit": {
                "export": {
                    "output_dir": str(tmp_path / "exports"),
                    "hmac_secret": "export-secret",
                    "webhook_url": "http://siem.local/audit",
                }
            },
        },
        post_json=fake_post,
    ).create_export(actor_id="supervisor-1", deliver=True)

    assert payload["delivery"]["sent"] is True
    assert seen["url"] == "http://siem.local/audit"
    assert seen["body"]["type"] == "askme.unified_audit_export"
    assert seen["body"]["manifest"]["record_count"] == 1
    assert seen["headers"]["X-Askme-Audit-Signature"]


def test_audit_export_delivery_failure_queues_and_replays_retry(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    retry_queue = tmp_path / "exports" / "retry.jsonl"
    SkillAuditLog(skill_path).append(skill_name="guide_visitor", status="ok")
    config = {
        "skills": {"audit_path": str(skill_path)},
        "audit": {
            "export": {
                "output_dir": str(tmp_path / "exports"),
                "retry_queue_path": str(retry_queue),
                "webhook_url": "http://siem.local/audit",
            }
        },
    }

    def failing_post(url: str, body: dict, headers: dict, timeout_s: float) -> dict:
        raise RuntimeError("siem temporarily unavailable")

    failed = AuditExportService(config, post_json=failing_post).create_export(
        actor_id="supervisor-1",
        deliver=True,
    )

    assert failed["delivery"]["sent"] is False
    assert failed["delivery"]["reason"] == "webhook_delivery_failed"
    assert retry_queue.exists()

    seen = {}

    def successful_post(url: str, body: dict, headers: dict, timeout_s: float) -> dict:
        seen["url"] = url
        seen["body"] = body
        return {"sent": True, "status_code": 200}

    service = AuditExportService(config, post_json=successful_post)
    status = service.retry_status()
    replay = service.retry_queued_deliveries()

    assert status["pending"] == 1
    assert status["items"][0]["export_id"] == failed["export"]["export_id"]
    assert replay["attempted"] == 1
    assert replay["sent"] == 1
    assert replay["remaining"] == 0
    assert not retry_queue.exists()
    assert seen["url"] == "http://siem.local/audit"
    assert seen["body"]["manifest"]["export_id"] == failed["export"]["export_id"]


def test_audit_export_delivery_non_2xx_queues_retry(tmp_path: Path) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    retry_queue = tmp_path / "exports" / "retry.jsonl"
    SkillAuditLog(skill_path).append(skill_name="guide_visitor", status="ok")
    config = {
        "skills": {"audit_path": str(skill_path)},
        "audit": {
            "export": {
                "output_dir": str(tmp_path / "exports"),
                "retry_queue_path": str(retry_queue),
                "webhook_url": "http://siem.local/audit",
            }
        },
    }

    def rejected_post(url: str, body: dict, headers: dict, timeout_s: float) -> dict:
        return {"sent": False, "status_code": 503, "response_preview": "unavailable"}

    payload = AuditExportService(config, post_json=rejected_post).create_export(
        actor_id="supervisor-1",
        deliver=True,
    )
    status = AuditExportService(config).retry_status()

    assert payload["delivery"]["sent"] is False
    assert payload["delivery"]["reason"] == "webhook_delivery_unsent"
    assert retry_queue.exists()
    assert status["pending"] == 1
    assert status["items"][0]["export_id"] == payload["export"]["export_id"]


def test_audit_export_endpoint_requires_export_permission(tmp_path: Path, monkeypatch) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    SkillAuditLog(skill_path).append(skill_name="guide_visitor", status="ok")
    config = {
        "skills": {"audit_path": str(skill_path)},
        "audit": {"export": {"output_dir": str(tmp_path / "exports")}},
        "field_operations": {
            "operator_directory": {
                "mode": "demo_config",
                "identity_provider": "local_config",
                "permissions": {
                    "operator": ["audit:read"],
                    "supervisor": ["audit:read", "audit:export"],
                },
            },
            "operators": {
                "dashboard.operator": {"roles": ["operator"]},
                "supervisor-1": {"roles": ["supervisor"]},
            },
        },
    }
    monkeypatch.setattr(health_server, "get_config", lambda: config)
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    denied = client.post("/api/audit/export", json={"operator_id": "dashboard.operator"})
    allowed = client.post("/api/audit/export", json={"operator_id": "supervisor-1"})

    assert denied.status_code == 403
    assert allowed.status_code == 200
    allowed_payload = AuditExportResponse.model_validate(allowed.json())
    assert allowed_payload.ok is True
    assert Path(allowed.json()["export"]["manifest_path"]).exists()


def test_audit_exports_endpoint_lists_recent_manifests_with_permission(
    tmp_path: Path,
    monkeypatch,
) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    SkillAuditLog(skill_path).append(skill_name="guide_visitor", status="ok")
    config = {
        "skills": {"audit_path": str(skill_path)},
        "audit": {"export": {"output_dir": str(tmp_path / "exports")}},
        "field_operations": {
            "operator_directory": {
                "mode": "demo_config",
                "identity_provider": "local_config",
                "permissions": {
                    "operator": ["audit:read"],
                    "supervisor": ["audit:read", "audit:export"],
                },
            },
            "operators": {
                "dashboard.operator": {"roles": ["operator"]},
                "supervisor-1": {"roles": ["supervisor"]},
            },
        },
    }
    monkeypatch.setattr(health_server, "get_config", lambda: config)
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    created = client.post("/api/audit/export", json={"operator_id": "supervisor-1"})
    denied = client.get("/api/audit/exports", params={"actor_id": "dashboard.operator"})
    allowed = client.get("/api/audit/exports", params={"actor_id": "supervisor-1"})

    assert created.status_code == 200
    assert denied.status_code == 403
    assert allowed.status_code == 200
    history = allowed.json()
    schema_history = AuditExportsResponse.model_validate(history)
    assert schema_history.count == 1
    assert history["count"] == 1
    assert history["exports"][0]["export_id"] == created.json()["export"]["export_id"]
    assert Path(history["exports"][0]["manifest_path"]).exists()
    assert history["exports"][0]["evidence_summary"]["ready"] is True


def test_audit_export_endpoint_preserves_time_window_in_manifest(tmp_path: Path, monkeypatch) -> None:
    skill_path = tmp_path / "skill-audit.jsonl"
    field_path = tmp_path / "field-action-audit.jsonl"
    _write_jsonl(
        field_path,
        [
            {
                "kind": "field_event_action",
                "created_at": 10,
                "event_id": "evt-before",
                "audit": {"at": 10, "action": "close", "outcome": "accepted"},
            },
            {
                "kind": "field_event_action",
                "created_at": 20,
                "event_id": "evt-window",
                "audit": {"at": 20, "action": "close", "outcome": "accepted"},
            },
        ],
    )
    config = {
        "skills": {"audit_path": str(skill_path)},
        "audit": {"export": {"output_dir": str(tmp_path / "exports")}},
        "field_operations": {
            "action_audit": {"path": str(field_path)},
            "operator_directory": {
                "mode": "demo_config",
                "identity_provider": "local_config",
                "permissions": {"supervisor": ["audit:read", "audit:export"]},
            },
            "operators": {"supervisor-1": {"roles": ["supervisor"]}},
        },
    }
    monkeypatch.setattr(health_server, "get_config", lambda: config)
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    response = client.post(
        "/api/audit/export",
        json={"operator_id": "supervisor-1", "since": "15", "until": "25"},
    )

    assert response.status_code == 200
    export = response.json()["export"]
    assert export["record_count"] == 1
    assert export["filters"]["since"] == "15"
    assert export["filters"]["until"] == "25"
    assert export["time_window"]["valid"] is True
    assert export["product_summary"]["record_count"] == 1


def test_audit_export_retry_endpoint_requires_export_permission(tmp_path: Path, monkeypatch) -> None:
    config = {
        "audit": {"export": {"output_dir": str(tmp_path / "exports")}},
        "field_operations": {
            "operator_directory": {
                "mode": "demo_config",
                "identity_provider": "local_config",
                "permissions": {
                    "operator": ["audit:read"],
                    "supervisor": ["audit:read", "audit:export"],
                },
            },
            "operators": {
                "dashboard.operator": {"roles": ["operator"]},
                "supervisor-1": {"roles": ["supervisor"]},
            },
        },
    }
    monkeypatch.setattr(health_server, "get_config", lambda: config)
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    denied_get = client.get("/api/audit/export/retry", params={"actor_id": "dashboard.operator"})
    allowed_get = client.get("/api/audit/export/retry", params={"actor_id": "supervisor-1"})
    denied_post = client.post("/api/audit/export/retry", json={"operator_id": "dashboard.operator"})
    allowed_post = client.post("/api/audit/export/retry", json={"operator_id": "supervisor-1"})

    assert denied_get.status_code == 403
    assert allowed_get.status_code == 200
    allowed_get_payload = AuditExportRetryStatusResponse.model_validate(allowed_get.json())
    assert allowed_get_payload.status == "empty"
    assert allowed_get.json()["status"] == "empty"
    assert denied_post.status_code == 403
    assert allowed_post.status_code == 200
    allowed_post_payload = AuditExportRetryResponse.model_validate(allowed_post.json())
    assert allowed_post_payload.status == "empty"
    assert allowed_post.json()["status"] == "empty"
