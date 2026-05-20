"""Product-level incident alert templates for robot field operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IncidentAlertTemplate:
    """Fixed announcement and notification contract for an incident topic."""

    topic: str
    severity: str
    voice: str
    dingtalk: str
    operator_action: str
    archive_required: bool = True
    notification_group: str = "security"

    def format(self, payload: dict[str, Any] | None = None) -> dict[str, str | bool]:
        data = payload or {}
        return {
            "topic": self.topic,
            "severity": self.severity,
            "voice": _safe_format(self.voice, data),
            "dingtalk": _safe_format(self.dingtalk, data),
            "operator_action": _safe_format(self.operator_action, data),
            "archive_required": self.archive_required,
            "notification_group": self.notification_group,
        }


@dataclass(frozen=True)
class IncidentPlaybook:
    """Operator-facing response plan attached to a field incident."""

    topic: str
    customer_status: str
    robot_motion_policy: str
    tts_profile: str
    responder_group: str
    operator_checklist: tuple[str, ...]
    evidence_policy: tuple[str, ...]
    escalation_after_s: int
    allow_llm_narrative: bool = False

    def format(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = payload or {}
        return {
            "topic": self.topic,
            "customer_status": _safe_format(self.customer_status, data),
            "robot_motion_policy": _safe_format(self.robot_motion_policy, data),
            "tts_profile": self.tts_profile,
            "responder_group": self.responder_group,
            "operator_checklist": [
                _safe_format(item, data) for item in self.operator_checklist
            ],
            "evidence_policy": [
                _safe_format(item, data) for item in self.evidence_policy
            ],
            "escalation_after_s": self.escalation_after_s,
            "allow_llm_narrative": self.allow_llm_narrative,
        }


INCIDENT_ALERTS: dict[str, IncidentAlertTemplate] = {
    "robot.fall_unrecoverable": IncidentAlertTemplate(
        topic="robot.fall_unrecoverable",
        severity="error",
        voice="机器人发生摔倒且无法自行恢复，已停止运动。请安保人员前往{location}处理。",
        dingtalk="【机器人摔倒】位置：{location}\n任务：{mission_id}\n证据：{image_path}",
        operator_action="立即停止任务，确认现场安全，通知安保到场扶正并检查机身。",
    ),
    "navigation.immobilized": IncidentAlertTemplate(
        topic="navigation.immobilized",
        severity="error",
        voice="机器人在{location}无法继续移动，已保持原地安全状态。请运维人员处理。",
        dingtalk="【机器人卡住】位置：{location}\n持续：{duration_s} 秒\n任务：{mission_id}",
        operator_action="远程暂停任务，检查障碍物和定位状态，必要时安排人员到场。",
    ),
    "security.malicious_blocking": IncidentAlertTemplate(
        topic="security.malicious_blocking",
        severity="error",
        voice="检测到有人恶意挡路，机器人已暂停并保持安全距离。请安保人员到场处理。",
        dingtalk="【人为挡路】位置：{location}\n距离：{distance_m} 米\n证据：{image_path}",
        operator_action="通知安保到场，保留视频和照片证据，避免机器人强行通过。",
    ),
    "actuator.joint_motor_fault": IncidentAlertTemplate(
        topic="actuator.joint_motor_fault",
        severity="error",
        voice="机器人关节电机故障，已停止当前任务。请运维人员前往{location}检查。",
        dingtalk="【关节电机故障】位置：{location}\n关节：{joint_id}\n故障码：{fault_code}\n证据：{image_path}",
        operator_action="停止机器人运动，检查关节电机、电源和驱动状态，记录维修结果。",
    ),
    "security.night_stranger_photo": IncidentAlertTemplate(
        topic="security.night_stranger_photo",
        severity="error",
        voice="夜间检测到陌生人在限制区域停留，已拍照记录并通知安保。",
        dingtalk="【夜间陌生人】位置：{location}\n区域：{zone_name}\n停留：{duration_s} 秒\n证据：{image_path}",
        operator_action="通知安保核查身份，保留照片和位置记录，必要时持续观察。",
    ),
    "traffic.illegal_parking": IncidentAlertTemplate(
        topic="traffic.illegal_parking",
        severity="warning",
        voice="检测到车辆停在非停车区域，已拍照记录并通知安保处理。",
        dingtalk="【车辆违停】位置：{location}\n区域：{zone_name}\n车牌：{plate_number}\n证据：{image_path}",
        operator_action="核对车牌和位置，通知安保处理，保留照片和时间记录。",
    ),
    "safety.fire_or_smoke": IncidentAlertTemplate(
        topic="safety.fire_or_smoke",
        severity="error",
        voice="检测到烟雾或高温异常，机器人已退到安全距离并通知安保。",
        dingtalk="【烟雾/高温】位置：{location}\n温度：{temperature_c}\n烟雾：{smoke_level}\n证据：{image_path}",
        operator_action="立即核查火情，通知安保和现场负责人，必要时启动应急预案。",
    ),
    "sanitation.trash_bin_full": IncidentAlertTemplate(
        topic="sanitation.trash_bin_full",
        severity="warning",
        voice="检测到垃圾桶已满，已通知保洁人员处理。",
        dingtalk="【垃圾桶已满】请保洁处理\n位置：{location}\n垃圾桶：{bin_id}\n满溢比例：{fill_ratio}\n证据：{image_path}",
        operator_action="通知保洁到场清理，处理后关闭事件并保留照片记录。",
        notification_group="cleaning",
    ),
    "security.crowd_gathering": IncidentAlertTemplate(
        topic="security.crowd_gathering",
        severity="warning",
        voice="检测到人员聚集，请注意通道秩序。系统已通知安保关注现场。",
        dingtalk="【人员聚集】位置：{location}\n人数：{person_count}\n持续：{duration_min} 分钟\n证据：{image_path}",
        operator_action="安保远程查看现场，必要时到场疏导，并持续复查人数变化。",
    ),
    "patrol.urgent_dispatch": IncidentAlertTemplate(
        topic="patrol.urgent_dispatch",
        severity="warning",
        voice="收到紧急巡检任务，机器人将暂停当前巡检并前往指定位置。",
        dingtalk="【紧急巡检】目标位置：{target_location}\n操作员：{operator_id}\n被打断任务：{interrupted_mission_id}",
        operator_action="确认目标位置、机器人电量和实时画面，提交任务交接。",
        notification_group="operations",
    ),
}


INCIDENT_PLAYBOOKS: dict[str, IncidentPlaybook] = {
    "robot.fall_unrecoverable": IncidentPlaybook(
        topic="robot.fall_unrecoverable",
        customer_status="机器人摔倒且无法恢复，已停止运动并等待人工处理。",
        robot_motion_policy="stop_and_hold",
        tts_profile="robot_fault",
        responder_group="security",
        operator_checklist=("停止任务", "确认现场安全", "通知安保到场处理"),
        evidence_policy=("location", "fault_type", "image_path", "mission_id"),
        escalation_after_s=60,
    ),
    "navigation.immobilized": IncidentPlaybook(
        topic="navigation.immobilized",
        customer_status="机器人被困或无法移动，已保持安全状态。",
        robot_motion_policy="stop_and_hold",
        tts_profile="robot_fault",
        responder_group="security",
        operator_checklist=("暂停任务", "检查障碍物", "安排现场处理"),
        evidence_policy=("location", "duration_s", "image_path", "mission_id"),
        escalation_after_s=120,
    ),
    "security.malicious_blocking": IncidentPlaybook(
        topic="security.malicious_blocking",
        customer_status="疑似人为挡路，机器人已暂停并保留证据。",
        robot_motion_policy="safe_pause",
        tts_profile="security_alert",
        responder_group="security",
        operator_checklist=("查看实时画面", "通知安保", "保留照片和视频"),
        evidence_policy=("location", "distance_m", "image_path", "duration_s"),
        escalation_after_s=90,
    ),
    "actuator.joint_motor_fault": IncidentPlaybook(
        topic="actuator.joint_motor_fault",
        customer_status="关节电机故障，机器人已停止当前任务。",
        robot_motion_policy="stop_and_hold",
        tts_profile="robot_fault",
        responder_group="security",
        operator_checklist=("停止运动", "记录关节和故障码", "通知运维检修"),
        evidence_policy=("location", "joint_id", "fault_code", "image_path"),
        escalation_after_s=60,
    ),
    "security.night_stranger_photo": IncidentPlaybook(
        topic="security.night_stranger_photo",
        customer_status="夜间陌生人停留已拍照记录，等待安保核查。",
        robot_motion_policy="keep_distance_observe",
        tts_profile="security_alert",
        responder_group="security",
        operator_checklist=("核查身份", "查看照片证据", "必要时持续观察"),
        evidence_policy=("location", "zone_name", "duration_s", "image_path", "confidence"),
        escalation_after_s=180,
    ),
    "traffic.illegal_parking": IncidentPlaybook(
        topic="traffic.illegal_parking",
        customer_status="车辆违停已记录，等待安保处理。",
        robot_motion_policy="observe_then_continue",
        tts_profile="patrol_notice",
        responder_group="security",
        operator_checklist=("确认位置", "核对车牌", "通知安保处理"),
        evidence_policy=("location", "zone_name", "plate_number", "image_path", "duration_s"),
        escalation_after_s=300,
        allow_llm_narrative=True,
    ),
    "safety.fire_or_smoke": IncidentPlaybook(
        topic="safety.fire_or_smoke",
        customer_status="烟雾或高温异常，机器人已退到安全距离。",
        robot_motion_policy="retreat_to_safe_distance",
        tts_profile="emergency_alert",
        responder_group="security",
        operator_checklist=("核查火情", "通知现场负责人", "保留传感器和照片证据"),
        evidence_policy=("location", "temperature_c", "smoke_level", "image_path", "sensor_updated_at"),
        escalation_after_s=30,
    ),
    "sanitation.trash_bin_full": IncidentPlaybook(
        topic="sanitation.trash_bin_full",
        customer_status="垃圾桶已满，等待保洁处理。",
        robot_motion_policy="record_then_continue",
        tts_profile="cleaning_notice",
        responder_group="cleaning",
        operator_checklist=("通知保洁", "确认清理结果", "关闭事件"),
        evidence_policy=("location", "bin_id", "fill_ratio", "image_path"),
        escalation_after_s=1800,
        allow_llm_narrative=True,
    ),
    "security.crowd_gathering": IncidentPlaybook(
        topic="security.crowd_gathering",
        customer_status="人员聚集已记录，安保需要关注现场秩序。",
        robot_motion_policy="observe_then_recheck",
        tts_profile="crowd_notice",
        responder_group="security",
        operator_checklist=("查看实时画面", "必要时到场疏导", "复查人数变化"),
        evidence_policy=("location", "person_count", "duration_min", "image_path"),
        escalation_after_s=900,
        allow_llm_narrative=True,
    ),
    "patrol.urgent_dispatch": IncidentPlaybook(
        topic="patrol.urgent_dispatch",
        customer_status="紧急巡检任务已接收，等待任务交接。",
        robot_motion_policy="pause_current_and_dispatch",
        tts_profile="operations_notice",
        responder_group="operations",
        operator_checklist=("确认目标位置", "查看实时画面", "提交任务交接"),
        evidence_policy=("operator_id", "target_location", "interrupted_mission_id", "reason"),
        escalation_after_s=120,
    ),
}


def format_incident_alert(topic: str, payload: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Return formatted incident alert payload for ``topic``."""
    template = INCIDENT_ALERTS.get(topic)
    if template is None:
        return None
    return template.format(payload)


def format_incident_playbook(topic: str, payload: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Return a structured response playbook for ``topic``."""
    playbook = INCIDENT_PLAYBOOKS.get(topic)
    if playbook is None:
        return None
    return playbook.format(payload)


def _safe_format(template: str, payload: dict[str, Any]) -> str:
    class _Missing(dict[str, Any]):
        def __missing__(self, key: str) -> str:
            return "-"

    return template.format_map(_Missing(payload))
