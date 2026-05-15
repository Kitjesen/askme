"""Customer-facing capability center for robot field skills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from askme.contracts import (
    CapabilityDependency,
    CapabilityPackageManifest,
    DependencyKind,
    PackageRuntimeInventory,
    PackageStatus,
    ScenarioPackageManifest,
    evaluate_capability_package_readiness,
    evaluate_scenario_package_readiness,
)
from askme.contracts.io import RiskLevel
from askme.skills.skill_model import SkillDefinition


@dataclass(frozen=True)
class CapabilitySpec:
    skill_name: str
    display_name: str
    group_id: str
    group_name: str
    description: str
    priority: str = "P1"
    customer_visible: bool = True
    requires_approval: bool = False


_GROUPS: dict[str, tuple[str, str]] = {
    "patrol": ("巡检任务", "固定巡检、临时巡检、任务恢复和巡检报告。"),
    "incident": ("异常处置", "摔倒、卡住、故障、挡路等事件闭环。"),
    "security": ("安保巡防", "夜间陌生人、违停、人群聚集、烟火异常和取证通知。"),
    "operations": ("保洁运营", "垃圾桶、公共设施和物业运营问题。"),
    "visitor": ("访客服务", "问路、目的地确认、语音指路和机器狗带路。"),
    "space": ("空间认知", "园区点位、别名、路线、禁行区和地图语义。"),
    "voice": ("语音交互", "交互准入、澄清、重复、静音和音色控制。"),
    "governance": ("管理审计", "能力启停、审批、调用审计和日志导出。"),
    "agent": ("在线增长", "通过受控 agent 生成、评审和发布新的能力包。"),
}


_SPECS: dict[str, CapabilitySpec] = {
    "patrol_scan": CapabilitySpec("patrol_scan", "巡检指定区域", "patrol", _GROUPS["patrol"][0], "到指定区域采集巡检信息。", "P0"),
    "patrol_report": CapabilitySpec("patrol_report", "生成巡检报告", "patrol", _GROUPS["patrol"][0], "汇总本轮巡检结果、异常和证据。", "P0"),
    "navigate": CapabilitySpec("navigate", "前往指定位置", "patrol", _GROUPS["patrol"][0], "把目标地点转换为导航任务。", "P0", requires_approval=True),
    "nav_cancel": CapabilitySpec("nav_cancel", "取消导航", "patrol", _GROUPS["patrol"][0], "取消当前导航或带路任务。", "P0"),
    "nav_query": CapabilitySpec("nav_query", "查询导航状态", "patrol", _GROUPS["patrol"][0], "查询当前位置、目标和导航进度。", "P0"),
    "robot_home": CapabilitySpec("robot_home", "返回待命点", "patrol", _GROUPS["patrol"][0], "返回待命点或充电区域。", "P0", requires_approval=True),
    "robot_estop": CapabilitySpec("robot_estop", "紧急停止", "incident", _GROUPS["incident"][0], "立即进入急停安全状态。", "P0", requires_approval=False),
    "dog_control": CapabilitySpec("dog_control", "机器狗姿态控制", "incident", _GROUPS["incident"][0], "站立、坐下、趴下等高风险动作。", "P0", requires_approval=True),
    "environment_report": CapabilitySpec("environment_report", "现场环境播报", "security", _GROUPS["security"][0], "播报当前环境、异常和传感器摘要。", "P1"),
    "find_person": CapabilitySpec("find_person", "寻找人员", "security", _GROUPS["security"][0], "按描述寻找人员并回报位置。", "P1"),
    "find_object": CapabilitySpec("find_object", "寻找物品", "security", _GROUPS["security"][0], "按描述寻找现场物品或目标。", "P1"),
    "safety_check": CapabilitySpec("safety_check", "安全检查", "security", _GROUPS["security"][0], "执行任务前的基础安全检查。", "P0"),
    "check_location": CapabilitySpec("check_location", "查询当前位置", "space", _GROUPS["space"][0], "查询机器狗当前所在区域或点位。", "P0"),
    "mapping": CapabilitySpec("mapping", "地图建图", "space", _GROUPS["space"][0], "启动或辅助园区建图流程。", "P1", requires_approval=True),
    "greet_person": CapabilitySpec("greet_person", "访客问候", "visitor", _GROUPS["visitor"][0], "在服务点向访客发起问候。", "P0"),
    "follow_person": CapabilitySpec("follow_person", "跟随人员", "visitor", _GROUPS["visitor"][0], "低速跟随指定人员。", "P1", requires_approval=True),
    "repeat_last": CapabilitySpec("repeat_last", "重复上一句", "voice", _GROUPS["voice"][0], "访客没听清时重复上一条播报。", "P0"),
    "mute_mic": CapabilitySpec("mute_mic", "关闭麦克风", "voice", _GROUPS["voice"][0], "临时关闭语音输入。", "P0"),
    "unmute_mic": CapabilitySpec("unmute_mic", "打开麦克风", "voice", _GROUPS["voice"][0], "恢复语音输入。", "P0"),
    "stop_speaking": CapabilitySpec("stop_speaking", "停止播报", "voice", _GROUPS["voice"][0], "立即停止当前语音播报。", "P0"),
    "volume_up": CapabilitySpec("volume_up", "调大音量", "voice", _GROUPS["voice"][0], "提高播报音量。", "P1"),
    "volume_down": CapabilitySpec("volume_down", "调小音量", "voice", _GROUPS["voice"][0], "降低播报音量。", "P1"),
    "speed_up": CapabilitySpec("speed_up", "加快语速", "voice", _GROUPS["voice"][0], "提高播报语速。", "P1"),
    "speed_down": CapabilitySpec("speed_down", "降低语速", "voice", _GROUPS["voice"][0], "降低播报语速。", "P1"),
    "list_skills": CapabilitySpec("list_skills", "查看能力清单", "governance", _GROUPS["governance"][0], "列出当前可用能力。", "P0"),
    "system_status": CapabilitySpec("system_status", "查看系统状态", "governance", _GROUPS["governance"][0], "查看机器人和服务健康状态。", "P0"),
    "agent_task": CapabilitySpec("agent_task", "后台专家任务", "agent", _GROUPS["agent"][0], "把复杂任务交给受控 agent 执行。", "P1", requires_approval=True),
}


_PLANNED: list[CapabilitySpec] = [
    CapabilitySpec("start_patrol_route", "开始巡检路线", "patrol", _GROUPS["patrol"][0], "启动客户配置的固定巡检路线。", "P0", requires_approval=True),
    CapabilitySpec("inspect_point", "检查指定点位", "patrol", _GROUPS["patrol"][0], "检查设备、门口、垃圾桶或停车区。", "P0"),
    CapabilitySpec("report_fall_unrecoverable", "摔倒无法恢复", "incident", _GROUPS["incident"][0], "播报、拍照、通知安保并归档。", "P0"),
    CapabilitySpec("report_stuck", "卡住无法运动", "incident", _GROUPS["incident"][0], "识别卡住并通知人工处理。", "P0"),
    CapabilitySpec("report_motor_fault", "关节电机故障", "incident", _GROUPS["incident"][0], "停止任务、记录故障码并通知维护。", "P0"),
    CapabilitySpec("detect_night_intruder", "夜间陌生人检测", "security", _GROUPS["security"][0], "夜间识别窗户、角落等区域人员。", "P0"),
    CapabilitySpec("detect_illegal_parking", "车辆违停检测", "security", _GROUPS["security"][0], "识别主通道和非停车区违停车辆。", "P0"),
    CapabilitySpec("detect_fire_smoke", "火灾烟雾监测", "security", _GROUPS["security"][0], "接入烟感、温度和图像烟火识别。", "P0"),
    CapabilitySpec("inspect_trash_bin", "垃圾桶满溢检测", "operations", _GROUPS["operations"][0], "定点拍照识别垃圾桶是否满溢。", "P0"),
    CapabilitySpec("detect_crowd_gathering", "人群聚集检测", "security", _GROUPS["security"][0], "识别人数、停留时长和复巡后是否仍聚集。", "P1"),
    CapabilitySpec("notify_security_group", "通知安保群", "security", _GROUPS["security"][0], "发送钉钉/企微/短信给安保。", "P0"),
    CapabilitySpec("notify_cleaning_group", "通知保洁群", "operations", _GROUPS["operations"][0], "通知保洁处理卫生问题。", "P0"),
    CapabilitySpec("offer_wayfinding_help", "主动问路服务", "visitor", _GROUPS["visitor"][0], "固定问询点识别访客停留后主动询问。", "P0"),
    CapabilitySpec("answer_wayfinding", "语音指路", "visitor", _GROUPS["visitor"][0], "查询园区语义地图并播报路线。", "P0"),
    CapabilitySpec("escort_visitor", "机器狗带路", "visitor", _GROUPS["visitor"][0], "低速引导访客前往确认地点。", "P1", requires_approval=True),
    CapabilitySpec("lookup_place", "查询园区点位", "space", _GROUPS["space"][0], "查询商户、楼宇、卫生间、停车区。", "P0"),
    CapabilitySpec("recommend_route", "推荐路线", "space", _GROUPS["space"][0], "按当前位置和目的地生成路线。", "P0"),
    CapabilitySpec("enable_capability", "启用能力", "governance", _GROUPS["governance"][0], "管理员启用能力包。", "P1", requires_approval=True),
    CapabilitySpec("view_skill_audit_log", "查看调用记录", "governance", _GROUPS["governance"][0], "查询能力调用、结果和操作者。", "P1"),
]

_SCENARIO_SKILLS: dict[str, tuple[str, ...]] = {
    "robot_abnormal_incident": (
        "report_fall_unrecoverable",
        "report_stuck",
        "report_motor_fault",
        "robot_estop",
    ),
    "night_stranger_photo": ("detect_night_intruder",),
    "illegal_parking": ("detect_illegal_parking",),
    "fire_or_smoke": ("detect_fire_smoke",),
    "trash_bin_full": ("inspect_trash_bin",),
    "urgent_patrol_dispatch": ("safety_check", "navigate", "patrol_scan"),
    "crowd_gathering": ("detect_crowd_gathering",),
    "wayfinding_help_point": (
        "offer_wayfinding_help",
        "lookup_place",
        "recommend_route",
        "answer_wayfinding",
    ),
    "visitor_escort": ("escort_visitor", "lookup_place", "recommend_route", "navigate"),
}

_SCENARIO_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "robot_abnormal_incident": ("机器人诊断状态", "定位", "现场照片或故障日志", "通知通道"),
    "night_stranger_photo": ("夜间时段配置", "敏感区域配置", "人员检测", "抓拍相机", "通知通道"),
    "illegal_parking": ("停车区/禁停区配置", "车辆检测", "停留计时", "抓拍相机", "通知通道"),
    "fire_or_smoke": ("烟感或温度传感器", "烟火视觉识别", "定位", "抓拍相机", "通知通道"),
    "trash_bin_full": ("垃圾桶点位库", "满溢识别", "抓拍相机", "保洁通知通道"),
    "urgent_patrol_dispatch": ("管理员身份", "目标点位", "runtime handoff", "暂停/恢复当前任务"),
    "crowd_gathering": ("人数检测", "区域停留计时", "复巡策略", "现场照片"),
    "wayfinding_help_point": ("问询点配置", "InteractionGate", "园区语义地图", "已审批路线知识"),
    "visitor_escort": ("目的地确认", "可通行路线", "低速导航策略", "访客跟随检测"),
}


def build_capability_center(
    skills: list[SkillDefinition],
    *,
    voice_triggers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a customer-facing grouped capability catalog."""
    by_name = {skill.name: skill for skill in skills}
    triggers_by_skill: dict[str, list[str]] = {}
    for phrase, skill_name in (voice_triggers or {}).items():
        triggers_by_skill.setdefault(skill_name, []).append(phrase)

    all_specs = dict(_SPECS)
    for spec in _PLANNED:
        all_specs.setdefault(spec.skill_name, spec)
    for skill in skills:
        all_specs.setdefault(skill.name, _fallback_spec(skill))

    groups: dict[str, dict[str, Any]] = {
        group_id: {
            "group_id": group_id,
            "display_name": label,
            "description": description,
            "skills": [],
            "available_count": 0,
            "enabled_count": 0,
            "missing_count": 0,
        }
        for group_id, (label, description) in _GROUPS.items()
    }

    for spec in sorted(all_specs.values(), key=lambda item: (item.group_id, item.priority, item.display_name)):
        skill = by_name.get(spec.skill_name)
        entry = _entry_for_spec(spec, skill, triggers_by_skill.get(spec.skill_name, []))
        group = groups.setdefault(
            spec.group_id,
            {
                "group_id": spec.group_id,
                "display_name": spec.group_name,
                "description": "",
                "skills": [],
                "available_count": 0,
                "enabled_count": 0,
                "missing_count": 0,
            },
        )
        group["skills"].append(entry)
        if entry["status"] == "missing":
            group["missing_count"] += 1
        else:
            group["available_count"] += 1
            if entry["enabled"]:
                group["enabled_count"] += 1

    ordered_groups = [group for group in groups.values() if group["skills"]]
    package_catalog = _build_capability_package_catalog(
        all_specs=all_specs,
        skills_by_name=by_name,
    )
    scenario_blueprints = _build_scenario_blueprints(
        all_specs=all_specs,
        skills_by_name=by_name,
        triggers_by_skill=triggers_by_skill,
    )
    return {
        "title": "园区巡检机器人能力中心",
        "summary": {
            "group_count": len(ordered_groups),
            "available_count": sum(group["available_count"] for group in ordered_groups),
            "enabled_count": sum(group["enabled_count"] for group in ordered_groups),
            "missing_recommended_count": sum(group["missing_count"] for group in ordered_groups),
            "scenario_count": scenario_blueprints["summary"]["scenario_count"],
            "scenario_ready_count": scenario_blueprints["summary"]["ready_count"],
            "scenario_partial_count": scenario_blueprints["summary"]["partial_count"],
            "scenario_blocked_count": scenario_blueprints["summary"]["blocked_count"],
        },
        "groups": ordered_groups,
        "capability_packages": package_catalog,
        "scenario_blueprints": scenario_blueprints,
        "online_growth": {
            "status": "available_with_governance",
            "mechanism": "audit-derived growth backlog + generated SKILL.md + validation + approval + customer skill package + audit",
            "recommended_lifecycle": ["observe", "candidate", "draft", "review", "approve", "assign_package", "enable", "audit"],
            "claude_code_inspired_controls": [
                "file-based skill packages",
                "specialized agent responsibilities",
                "pre-execution approval gate",
                "post-execution audit hook",
                "tool allowlist per skill",
                "versioned package rollout and rollback",
            ],
        },
    }


def _build_scenario_blueprints(
    *,
    all_specs: dict[str, CapabilitySpec],
    skills_by_name: dict[str, SkillDefinition],
    triggers_by_skill: dict[str, list[str]],
) -> dict[str, Any]:
    try:
        from askme.pipeline.field_scenarios import FIELD_SCENARIOS
    except Exception:
        return {
            "summary": {
                "scenario_count": 0,
                "ready_count": 0,
                "partial_count": 0,
                "blocked_count": 0,
            },
            "items": [],
            "policy": {
                "source": "field_scenarios_unavailable",
                "scenario_requires_all_required_skills": True,
            },
        }

    items: list[dict[str, Any]] = []
    for scenario in sorted(FIELD_SCENARIOS.values(), key=lambda item: (item.priority, item.scenario_id)):
        required_skill_names = _SCENARIO_SKILLS.get(scenario.scenario_id, ())
        required_skills = [
            _scenario_skill_entry(
                skill_name,
                all_specs=all_specs,
                skills_by_name=skills_by_name,
                triggers_by_skill=triggers_by_skill,
            )
            for skill_name in required_skill_names
        ]
        installed_count = sum(1 for item in required_skills if item["installed"])
        enabled_count = sum(1 for item in required_skills if item["enabled"])
        missing_skill_names = [
            item["skill_name"] for item in required_skills if not item["installed"]
        ]
        disabled_skill_names = [
            item["skill_name"]
            for item in required_skills
            if item["installed"] and not item["enabled"]
        ]
        if required_skills and enabled_count == len(required_skills):
            coverage_status = "ready"
            next_action = "可进入场景联调：验证真实传感器、通知和现场验收口径。"
        elif installed_count:
            coverage_status = "partial"
            next_action = _next_action(missing_skill_names, disabled_skill_names)
        else:
            coverage_status = "blocked"
            next_action = _next_action(missing_skill_names, disabled_skill_names)
        scenario_manifest = _scenario_package_manifest(
            scenario,
            required_skills=required_skills,
        )
        scenario_readiness = evaluate_scenario_package_readiness(
            scenario_manifest,
            inventory=_scenario_runtime_inventory(required_skills),
        )

        items.append({
            "scenario_id": scenario.scenario_id,
            "display_name": scenario.name,
            "category": scenario.category,
            "priority": scenario.priority,
            "coverage_status": coverage_status,
            "installed_count": installed_count,
            "enabled_count": enabled_count,
            "required_skill_count": len(required_skills),
            "missing_skill_names": missing_skill_names,
            "disabled_skill_names": disabled_skill_names,
            "required_skills": required_skills,
            "trigger_rule": scenario.trigger_rule,
            "required_evidence": list(scenario.required_evidence),
            "robot_behavior": list(scenario.robot_behavior),
            "dependencies": list(_SCENARIO_DEPENDENCIES.get(scenario.scenario_id, ())),
            "notification_group": scenario.notification_group,
            "archive_required": scenario.archive_required,
            "interrupts_current_task": scenario.interrupts_current_task,
            "requires_operator_approval": scenario.requires_operator_approval
            or any(item["requires_approval"] for item in required_skills),
            "acceptance_criteria": list(scenario.acceptance_criteria),
            "runtime_entry": "field_event_trigger",
            "next_action": next_action,
            "package_manifest": scenario_manifest.to_dict(),
            "package_readiness": scenario_readiness,
        })

    return {
        "summary": {
            "scenario_count": len(items),
            "ready_count": sum(1 for item in items if item["coverage_status"] == "ready"),
            "partial_count": sum(1 for item in items if item["coverage_status"] == "partial"),
            "blocked_count": sum(1 for item in items if item["coverage_status"] == "blocked"),
        },
        "items": items,
        "policy": {
            "source": "askme.pipeline.field_scenarios",
            "scenario_requires_all_required_skills": True,
            "runtime_entry": "field_event_trigger",
            "customer_claim_rule": "ready means software skill path exists; real sensor, notification, and robot hardware still need site validation",
        },
    }


def _build_capability_package_catalog(
    *,
    all_specs: dict[str, CapabilitySpec],
    skills_by_name: dict[str, SkillDefinition],
) -> dict[str, Any]:
    manifests: list[dict[str, Any]] = []
    readiness_items: list[dict[str, Any]] = []
    inventory = _capability_runtime_inventory(skills_by_name.values())
    for spec in sorted(all_specs.values(), key=lambda item: (item.group_id, item.priority, item.skill_name)):
        manifest = _capability_package_manifest(spec)
        readiness = evaluate_capability_package_readiness(manifest, inventory=inventory)
        manifests.append(manifest.to_dict())
        readiness_items.append(readiness)
    return {
        "summary": {
            "package_count": len(manifests),
            "ready_count": sum(1 for item in readiness_items if item["status"] == "ready"),
            "manual_check_count": sum(1 for item in readiness_items if item["status"] == "manual_check"),
            "blocked_count": sum(1 for item in readiness_items if item["status"] == "blocked"),
        },
        "items": manifests,
        "readiness": readiness_items,
        "policy": {
            "package_id_rule": "capability.<skill_name>",
            "customer_enablement_requires_readiness": True,
            "ready_still_requires_site_validation": True,
        },
    }


def _capability_package_manifest(spec: CapabilitySpec) -> CapabilityPackageManifest:
    risk_level = RiskLevel.HIGH if spec.requires_approval else RiskLevel.LOW
    return CapabilityPackageManifest(
        package_id=_capability_package_id(spec.skill_name),
        display_name=spec.display_name,
        status=PackageStatus.PILOT,
        capability=spec.skill_name,
        summary=spec.description,
        inputs=("operator_intent", "site_context", "runtime_state"),
        outputs=("customer_visible_result", "audit_record"),
        dependencies=(
            CapabilityDependency(
                name=spec.skill_name,
                kind=DependencyKind.SKILL,
                required=True,
                reason=f"能力包需要底层技能 {spec.skill_name} 已安装并启用。",
                customer_visible=True,
            ),
            *_approval_dependencies(f"approval.{spec.skill_name}", spec.requires_approval),
        ),
        risk_level=risk_level,
        risk_controls=(
            ("主管审批后才能启用或执行高风险动作。",)
            if spec.requires_approval
            else ("记录调用审计，保留人工接管入口。",)
        ),
        customer_visible_name=spec.display_name,
        customer_visible_description=spec.description,
        customer_visible_outputs=("结果展示", "事件记录", "审计记录"),
        tags=(spec.group_id, spec.priority),
        metadata={
            "skill_name": spec.skill_name,
            "group_id": spec.group_id,
            "group_name": spec.group_name,
            "priority": spec.priority,
            "requires_approval": spec.requires_approval,
        },
    )


def _scenario_package_manifest(
    scenario: Any,
    *,
    required_skills: list[dict[str, Any]],
) -> ScenarioPackageManifest:
    requires_approval = bool(scenario.requires_operator_approval) or any(
        item["requires_approval"] for item in required_skills
    )
    return ScenarioPackageManifest(
        package_id=f"scenario.{scenario.scenario_id}",
        display_name=scenario.name,
        status=PackageStatus.PILOT,
        scenario=scenario.scenario_id,
        site_id="site-profile",
        customer_name="customer-site",
        capability_packages=tuple(
            _capability_package_id(item["skill_name"]) for item in required_skills
        ),
        inputs=("site_event", "perception_evidence", "operator_policy", "runtime_context"),
        outputs=("customer_visible_response", "field_event", "audit_record"),
        dependencies=(
            *(
                CapabilityDependency(
                    name=item["skill_name"],
                    kind=DependencyKind.SKILL,
                    required=True,
                    reason=f"场景 {scenario.name} 需要技能 {item['display_name']}。",
                    customer_visible=True,
                )
                for item in required_skills
            ),
            *_approval_dependencies(
                f"approval.{scenario.scenario_id}",
                requires_approval or scenario.interrupts_current_task,
            ),
        ),
        risk_level=RiskLevel.HIGH if requires_approval or scenario.interrupts_current_task else RiskLevel.MEDIUM,
        risk_controls=(
            ("需要操作员审批或人工接管入口。",)
            if requires_approval or scenario.interrupts_current_task
            else ("仅在已配置服务点或事件规则内触发。", "记录触发证据和回答依据。")
        ),
        customer_visible_name=scenario.name,
        customer_visible_description=scenario.trigger_rule,
        customer_visible_steps=tuple(scenario.robot_behavior),
        customer_visible_outputs=tuple(scenario.acceptance_criteria or scenario.required_evidence),
        rollout_notes="先在演示或试点站点启用；真实硬件、传感器、通知和现场规则通过后再进入生产。",
        metadata={
            "category": scenario.category,
            "priority": scenario.priority,
            "notification_group": scenario.notification_group,
            "archive_required": scenario.archive_required,
            "interrupts_current_task": scenario.interrupts_current_task,
            "requires_operator_approval": requires_approval,
        },
    )


def _approval_dependencies(name: str, required: bool) -> tuple[CapabilityDependency, ...]:
    if not required:
        return ()
    return (
        CapabilityDependency(
            name=name,
            kind=DependencyKind.HUMAN_APPROVAL,
            required=False,
            reason="Human approval is required before this package is enabled for a customer site.",
            customer_visible=True,
        ),
    )


def _capability_runtime_inventory(skills: Any) -> PackageRuntimeInventory:
    enabled_skill_names = {
        skill.name
        for skill in skills
        if getattr(skill, "enabled", False)
    }
    return PackageRuntimeInventory(
        skills=frozenset(enabled_skill_names),
        capability_packages=frozenset(
            _capability_package_id(name) for name in enabled_skill_names
        ),
    )


def _scenario_runtime_inventory(required_skills: list[dict[str, Any]]) -> PackageRuntimeInventory:
    enabled_skill_names = {
        item["skill_name"]
        for item in required_skills
        if item.get("installed") and item.get("enabled")
    }
    return PackageRuntimeInventory(
        skills=frozenset(enabled_skill_names),
        capability_packages=frozenset(
            _capability_package_id(name) for name in enabled_skill_names
        ),
    )


def _capability_package_id(skill_name: str) -> str:
    return f"capability.{skill_name}"


def _scenario_skill_entry(
    skill_name: str,
    *,
    all_specs: dict[str, CapabilitySpec],
    skills_by_name: dict[str, SkillDefinition],
    triggers_by_skill: dict[str, list[str]],
) -> dict[str, Any]:
    spec = all_specs.get(skill_name)
    skill = skills_by_name.get(skill_name)
    if spec is None:
        spec = _fallback_spec(skill) if skill else CapabilitySpec(
            skill_name=skill_name,
            display_name=_humanize(skill_name),
            group_id="governance",
            group_name=_GROUPS["governance"][0],
            description="场景蓝图引用的待定义能力。",
            priority="P2",
        )
    entry = _entry_for_spec(spec, skill, triggers_by_skill.get(skill_name, []))
    return {
        "skill_name": entry["skill_name"],
        "display_name": entry["display_name"],
        "status": entry["status"],
        "installed": entry["installed"],
        "enabled": entry["enabled"],
        "source": entry["source"],
        "safety_level": entry["safety_level"],
        "requires_approval": entry["requires_approval"],
        "voice_triggers": entry["voice_triggers"],
    }


def _next_action(missing_skill_names: list[str], disabled_skill_names: list[str]) -> str:
    if missing_skill_names:
        return "需要补齐技能：" + "、".join(missing_skill_names)
    if disabled_skill_names:
        return "需要启用或审批技能：" + "、".join(disabled_skill_names)
    return "需要补充场景蓝图或现场验收数据。"


def _fallback_spec(skill: SkillDefinition) -> CapabilitySpec:
    group_id = _group_for_skill(skill)
    group_name = _GROUPS.get(group_id, ("其他能力", ""))[0]
    return CapabilitySpec(
        skill_name=skill.name,
        display_name=_humanize(skill.name),
        group_id=group_id,
        group_name=group_name,
        description=skill.description or "项目自定义能力。",
        priority="P2",
        requires_approval=skill.safety_level in {"dangerous", "critical"} or skill.confirm_before_execute,
    )


def _group_for_skill(skill: SkillDefinition) -> str:
    tags = {tag.lower() for tag in skill.tags}
    name = skill.name.lower()
    if "agent" in tags or "agent" in name:
        return "agent"
    if "voice" in tags or any(token in name for token in ("mute", "volume", "speed", "repeat", "speak")):
        return "voice"
    if "robot" in tags or "nav" in name or "patrol" in name:
        return "patrol"
    if "safety" in tags or "estop" in name or "fault" in name:
        return "incident"
    if "vision" in tags or "find" in name or "environment" in name:
        return "security"
    if "memory" in tags or "location" in name or "map" in name:
        return "space"
    return "governance"


def _entry_for_spec(
    spec: CapabilitySpec,
    skill: SkillDefinition | None,
    voice_triggers: list[str],
) -> dict[str, Any]:
    enabled = bool(skill and skill.enabled)
    safety_level = skill.safety_level if skill else "planned"
    requires_approval = (
        spec.requires_approval
        or safety_level in {"dangerous", "critical"}
        or bool(skill and skill.confirm_before_execute)
    )
    return {
        "skill_name": spec.skill_name,
        "display_name": spec.display_name,
        "description": spec.description if not skill else (skill.description or spec.description),
        "priority": spec.priority,
        "status": "enabled" if enabled else ("disabled" if skill else "missing"),
        "enabled": enabled,
        "installed": skill is not None,
        "safety_level": safety_level,
        "requires_approval": requires_approval,
        "approval_policy": "supervisor_required" if requires_approval else "operator_allowed",
        "source": skill.source if skill else "planned",
        "execution": skill.execution if skill else "not_implemented",
        "customer_visible": spec.customer_visible,
        "voice_triggers": sorted(voice_triggers)[:12],
        "tags": list(skill.tags) if skill else [],
    }


def _humanize(name: str) -> str:
    return name.replace("_", " ").strip().title()
