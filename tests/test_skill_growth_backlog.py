from __future__ import annotations

from pathlib import Path

from askme.skills.audit import SkillAuditLog
from askme.skills.growth_backlog import SkillGrowthBacklog


def test_growth_backlog_groups_repeated_failed_requests(tmp_path: Path) -> None:
    audit = SkillAuditLog(tmp_path / "skill-audit.jsonl")
    audit.append(skill_name="unknown", status="failed", user_text="帮我检查喷泉灯", reason="no_skill")
    audit.append(skill_name="unknown", status="blocked", user_text="帮我检查喷泉灯", reason="not_found")
    audit.append(skill_name="robot_estop", status="blocked", user_text="停下", reason="estop_active")

    backlog = SkillGrowthBacklog(audit, tmp_path / "growth.json")
    payload = backlog.payload(min_occurrences=2)

    assert payload["policy"]["auto_create_or_enable_skills"] is False
    assert payload["summary"]["candidate_count"] == 1
    candidate = payload["candidates"][0]
    assert candidate["summary"] == "帮我检查喷泉灯"
    assert candidate["evidence_count"] == 2
    assert candidate["status"] == "candidate"
    assert candidate["priority"] == "P1"
    assert candidate["suggested_skill_name"].startswith("skill_")


def test_growth_backlog_mark_persists_product_decision(tmp_path: Path) -> None:
    audit = SkillAuditLog(tmp_path / "skill-audit.jsonl")
    audit.append(skill_name="unknown", status="failed", user_text="巡检西门灯箱", reason="no_skill")
    audit.append(skill_name="unknown", status="blocked", user_text="巡检西门灯箱", reason="not_found")

    backlog = SkillGrowthBacklog(audit, tmp_path / "growth.json")
    candidate_id = backlog.payload(min_occurrences=1)["candidates"][0]["candidate_id"]

    result = backlog.mark(
        candidate_id,
        action="promote",
        operator_id="pm-1",
        note="common site operation",
    )
    updated = result["backlog"]["candidates"][0]

    assert result["ok"] is True
    assert updated["status"] == "promoted"
    assert updated["updated_by"] == "pm-1"
    assert updated["note"] == "common site operation"


def test_growth_backlog_flags_motion_requests_as_dangerous(tmp_path: Path) -> None:
    audit = SkillAuditLog(tmp_path / "skill-audit.jsonl")
    audit.append(skill_name="unknown", status="failed", user_text="带路去二号楼", reason="no_route")

    candidate = SkillGrowthBacklog(audit, tmp_path / "growth.json").payload(
        min_occurrences=1
    )["candidates"][0]

    assert candidate["risk_level"] == "dangerous"
