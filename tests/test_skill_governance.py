from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from askme.llm.intent_router import IntentRouter, IntentType
from askme.skills.audit import SkillAuditLog
from askme.skills.skill_manager import SkillManager
from askme.tools.skill_tools import CreateSkillTool


def _patch_skill_data(tmp_path: Path, monkeypatch) -> None:
    import askme.skills.skill_manager as skill_manager_module
    from askme.skills.audit import SkillAuditLog

    monkeypatch.setattr(skill_manager_module, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        skill_manager_module,
        "_SETTINGS_FILE",
        tmp_path / "skills_settings.json",
    )
    monkeypatch.setattr(
        skill_manager_module,
        "SkillAuditLog",
        lambda: SkillAuditLog(tmp_path / "skill-audit.jsonl"),
    )
    (tmp_path / "skills").mkdir(parents=True, exist_ok=True)


def test_generated_skill_defaults_to_pending_and_disabled(tmp_path: Path, monkeypatch) -> None:
    _patch_skill_data(tmp_path, monkeypatch)
    manager = SkillManager(project_dir=tmp_path)
    manager.load()
    router = IntentRouter(voice_triggers=manager.get_voice_triggers())

    tool = CreateSkillTool()
    tool.set_context(manager, router)
    result = tool.execute(
        name="site_tip",
        description="Answer a fixed site tip",
        voice_trigger="site-tip-trigger",
        prompt="Answer with the approved site tip.",
    )

    assert "待审批" in result
    skill = manager.get("site_tip")
    assert skill is not None
    assert skill.enabled is False
    assert router.route("site-tip-trigger").type != IntentType.VOICE_TRIGGER

    queue = manager.get_generated_skill_governance()
    record = next(item for item in queue["records"] if item["skill_name"] == "site_tip")
    assert record["status"] == "pending_approval"
    assert record["enabled"] is False
    packages = manager.get_skill_packages()
    assert packages["policy"]["approved_generated_skills_require_package"] is True
    assert packages["summary"]["package_count"] >= 1


def test_approving_generated_skill_enables_voice_trigger(tmp_path: Path, monkeypatch) -> None:
    _patch_skill_data(tmp_path, monkeypatch)
    manager = SkillManager(project_dir=tmp_path)
    manager.load()
    router = IntentRouter(voice_triggers=manager.get_voice_triggers())

    tool = CreateSkillTool()
    tool.set_context(manager, router)
    tool.execute(
        name="visitor_help",
        description="Offer visitor help",
        voice_trigger="visitor-help-trigger",
        prompt="Offer visitor help.",
    )

    approved = manager.review_generated_skill(
        "visitor_help",
        action="approve",
        operator_id="pm.operator",
        note="approved test skill",
        router=router,
    )

    assert approved["ok"] is True
    assert approved["status"] == "approved"
    assert approved["enabled"] is True
    assert router.route("visitor-help-trigger").type == IntentType.VOICE_TRIGGER
    assert router.route("visitor-help-trigger").skill_name == "visitor_help"
    queue = manager.get_generated_skill_governance()
    record = next(item for item in queue["records"] if item["skill_name"] == "visitor_help")
    assert "default-demo" in record["package_ids"]
    packages = manager.get_skill_packages()
    default_package = next(
        item for item in packages["packages"] if item["package_id"] == "default-demo"
    )
    assert "visitor_help" in default_package["skill_names"]
    audit_records = SkillAuditLog(tmp_path / "skill-audit.jsonl").recent()
    assert audit_records[-1]["event_type"] == "governance"
    assert audit_records[-1]["operator_id"] == "pm.operator"
    assert audit_records[-1]["action"] == "approve"


def test_unassigning_generated_skill_from_package_disables_it(tmp_path: Path, monkeypatch) -> None:
    _patch_skill_data(tmp_path, monkeypatch)
    manager = SkillManager(project_dir=tmp_path)
    manager.load()
    tool = CreateSkillTool()
    tool.set_context(manager, IntentRouter())
    tool.execute(
        name="visitor_route_tip",
        description="Offer a route tip",
        voice_trigger="visitor-route-tip-trigger",
        prompt="Offer a route tip.",
    )
    approved = manager.review_generated_skill(
        "visitor_route_tip",
        action="approve",
        operator_id="supervisor-1",
    )
    assert approved["ok"] is True
    assert manager.get("visitor_route_tip").enabled is True

    unassigned = manager.update_skill_package(
        skill_name="visitor_route_tip",
        package_id="default-demo",
        action="unassign",
        operator_id="supervisor-1",
    )

    assert unassigned["ok"] is True
    assert unassigned["enabled"] is False
    manager.load()
    assert manager.get("visitor_route_tip").enabled is False
    assert "visitor-route-tip-trigger" not in manager.get_voice_triggers()


def test_upserting_skill_package_records_customer_scope(tmp_path: Path, monkeypatch) -> None:
    _patch_skill_data(tmp_path, monkeypatch)
    manager = SkillManager(project_dir=tmp_path)
    manager.load()

    result = manager.upsert_skill_package(
        package_id="fanmu-phase-1",
        display_name="Fanmu phase 1 package",
        site_id="fanmu",
        customer_name="Fanmu",
        description="Visitor and patrol abilities for phase 1.",
        enabled=True,
        release_channel="pilot",
        rollout_percent=25,
        operator_id="supervisor-1",
    )

    assert result["ok"] is True
    package = result["package"]
    assert package["package_id"] == "fanmu-phase-1"
    assert package["site_id"] == "fanmu"
    assert package["customer_name"] == "Fanmu"
    assert package["release_channel"] == "pilot"
    assert package["rollout_percent"] == 25
    assert package["release_version"] == 1
    packages = manager.get_skill_packages()
    assert any(item["package_id"] == "fanmu-phase-1" for item in packages["packages"])


def test_skill_package_release_and_rollback_restore_snapshot(tmp_path: Path, monkeypatch) -> None:
    _patch_skill_data(tmp_path, monkeypatch)
    manager = SkillManager(project_dir=tmp_path)
    manager.load()
    router = IntentRouter(voice_triggers=manager.get_voice_triggers())
    tool = CreateSkillTool()
    tool.set_context(manager, router)
    tool.execute(
        name="visitor_release_tip",
        description="Offer a visitor release tip",
        voice_trigger="visitor-release-tip-trigger",
        prompt="Offer a visitor release tip.",
    )
    approved = manager.review_generated_skill(
        "visitor_release_tip",
        action="approve",
        operator_id="supervisor-1",
        router=router,
    )
    assert approved["ok"] is True
    released = manager.release_skill_package(
        package_id="default-demo",
        release_channel="pilot",
        rollout_percent=30,
        operator_id="supervisor-1",
        note="gray release for site test",
    )
    release_version = released["package"]["release_version"]

    unassigned = manager.update_skill_package(
        skill_name="visitor_release_tip",
        package_id="default-demo",
        action="unassign",
        operator_id="supervisor-1",
    )
    assert unassigned["enabled"] is False
    manager.load()
    assert manager.get("visitor_release_tip").enabled is False

    rolled_back = manager.rollback_skill_package(
        package_id="default-demo",
        target_version=release_version,
        operator_id="supervisor-1",
        note="rollback after failed test",
    )

    assert rolled_back["ok"] is True
    package = rolled_back["package"]
    assert package["rollback_of_version"] == release_version
    assert "visitor_release_tip" in package["skill_names"]
    assert package["release_channel"] == "pilot"
    assert package["rollout_percent"] == 30
    assert rolled_back["history"]["count"] >= 3
    latest_history = rolled_back["history"]["records"][-1]
    changed_fields = {item["field"] for item in latest_history["changed_fields"]}
    assert "skill_names" in changed_fields
    assert "rollback_of_version" in changed_fields
    manager.load()
    assert manager.get("visitor_release_tip").enabled is True


def test_rollout_zero_disables_package_skill(tmp_path: Path, monkeypatch) -> None:
    _patch_skill_data(tmp_path, monkeypatch)
    manager = SkillManager(project_dir=tmp_path)
    manager.load()
    tool = CreateSkillTool()
    tool.set_context(manager, IntentRouter())
    tool.execute(
        name="visitor_zero_rollout",
        description="Rollout zero visitor skill",
        voice_trigger="visitor-zero-rollout-trigger",
        prompt="Offer a rollout zero response.",
    )
    approved = manager.review_generated_skill(
        "visitor_zero_rollout",
        action="approve",
        operator_id="supervisor-1",
    )
    assert approved["enabled"] is True

    released = manager.release_skill_package(
        package_id="default-demo",
        release_channel="pilot",
        rollout_percent=0,
        operator_id="supervisor-1",
    )

    assert released["package"]["rollout_percent"] == 0
    manager.load()
    assert manager.get("visitor_zero_rollout").enabled is False


def test_skill_manager_creates_generated_draft_for_review(tmp_path: Path, monkeypatch) -> None:
    _patch_skill_data(tmp_path, monkeypatch)
    manager = SkillManager(project_dir=tmp_path)

    result = manager.create_generated_skill_draft(
        name="Inspect Fountain Light!",
        description="Check repeated fountain light requests",
        voice_trigger="检查喷泉灯",
        prompt="Check whether the fountain light request can be handled. User: {{user_input}}",
        tags=["growth", "site"],
        operator_id="pm.operator",
    )

    assert result["ok"] is True
    assert result["skill_name"] == "inspect_fountain_light_"
    assert result["enabled"] is False
    assert result["status"] == "pending_approval"
    assert Path(result["path"]).exists()

    manager.load()
    skill = manager.get("inspect_fountain_light_")
    assert skill is not None
    assert skill.source == "generated"
    assert skill.enabled is False
    record = next(
        item for item in manager.get_generated_skill_governance()["records"]
        if item["skill_name"] == "inspect_fountain_light_"
    )
    assert record["status"] == "pending_approval"
    audit_records = SkillAuditLog(tmp_path / "skill-audit.jsonl").recent()
    assert audit_records[-1]["status"] == "draft_created"


def test_rejecting_generated_skill_keeps_trigger_blocked(tmp_path: Path, monkeypatch) -> None:
    _patch_skill_data(tmp_path, monkeypatch)
    manager = SkillManager(project_dir=tmp_path)
    manager.load()
    router = IntentRouter(voice_triggers=manager.get_voice_triggers())

    tool = CreateSkillTool()
    tool.set_context(manager, router)
    tool.execute(
        name="unsafe_generated",
        description="Unsafe generated skill",
        voice_trigger="unsafe-generated-trigger",
        prompt="Do something unsafe.",
    )

    rejected = manager.review_generated_skill(
        "unsafe_generated",
        action="reject",
        operator_id="safety.operator",
        router=router,
    )

    assert rejected["ok"] is True
    assert rejected["status"] == "rejected"
    assert rejected["enabled"] is False
    assert router.route("unsafe-generated-trigger").type != IntentType.VOICE_TRIGGER


def test_approval_blocks_generated_skill_with_high_risk_tool(tmp_path: Path, monkeypatch) -> None:
    _patch_skill_data(tmp_path, monkeypatch)
    manager = SkillManager(project_dir=tmp_path)
    manager.load()
    router = IntentRouter(voice_triggers=manager.get_voice_triggers())

    tool = CreateSkillTool()
    tool.set_context(manager, router)
    tool.execute(
        name="dangerous_draft",
        description="Dangerous draft",
        voice_trigger="dangerous-draft-trigger",
        prompt="Run a risky action.",
        tools_section="bash",
    )

    approved = manager.review_generated_skill(
        "dangerous_draft",
        action="approve",
        operator_id="pm.operator",
        router=router,
    )

    assert approved["ok"] is False
    assert approved["error"] == "generated skill validation failed"
    assert approved["validation"]["ok"] is False
    assert router.route("dangerous-draft-trigger").type != IntentType.VOICE_TRIGGER


def test_generated_skill_governance_works_with_mocked_manager(tmp_path: Path) -> None:
    tool = CreateSkillTool()
    manager = MagicMock()
    manager.create_generated_skill_draft.return_value = {
        "ok": True,
        "skill_name": "draft_skill",
        "path": str(tmp_path / "skills" / "draft_skill" / "SKILL.md"),
        "loaded_count": 42,
    }
    router = MagicMock()
    tool.set_context(manager, router)

    result = tool.execute(name="draft_skill", description="draft", prompt="draft prompt")

    assert "待审批" in result
    manager.create_generated_skill_draft.assert_called_once()
    assert manager.create_generated_skill_draft.call_args.kwargs["router"] is router
