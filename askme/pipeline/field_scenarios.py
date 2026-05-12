"""Product scenario registry for field robot operations.

The registry is product-facing. It describes what the customer expects the
robot service to recognize, say, notify, archive, and hand off to runtime.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldScenario:
    """A customer-visible field operation scenario."""

    scenario_id: str
    name: str
    category: str
    priority: str
    trigger_rule: str
    required_evidence: tuple[str, ...]
    robot_behavior: tuple[str, ...]
    notification_group: str
    archive_required: bool
    interrupts_current_task: bool = False
    requires_operator_approval: bool = False
    acceptance_criteria: tuple[str, ...] = ()


FIELD_SCENARIOS: dict[str, FieldScenario] = {
    "robot_abnormal_incident": FieldScenario(
        scenario_id="robot_abnormal_incident",
        name="机器人异常事件",
        category="safety_incident",
        priority="P0",
        trigger_rule="机器人摔倒、卡住、被恶意阻挡或关键部件故障时触发。",
        required_evidence=("位置", "故障类型", "现场照片或诊断日志", "任务编号"),
        robot_behavior=("立即停止运动", "播放安全提示", "等待保安或运维处理"),
        notification_group="security",
        archive_required=True,
        interrupts_current_task=True,
        acceptance_criteria=(
            "异常发生后 3 秒内停止运动",
            "钉钉通知包含位置、故障类型和证据",
            "事件归档包含处置状态和关闭记录",
        ),
    ),
    "night_stranger_photo": FieldScenario(
        scenario_id="night_stranger_photo",
        name="夜间陌生人拍照",
        category="security_incident",
        priority="P0",
        trigger_rule="夜间在窗户、角落、围栏等敏感区域检测到陌生人停留或拍照。",
        required_evidence=("抓拍照片", "人员位置", "当前地点", "停留时长", "检测置信度"),
        robot_behavior=("保持安全距离观察", "通知保安", "记录事件并关联证据"),
        notification_group="security",
        archive_required=True,
        acceptance_criteria=(
            "通知中包含当前地点和抓拍证据",
            "不主动贴近人员",
            "可在事件详情中追溯判断依据",
        ),
    ),
    "illegal_parking": FieldScenario(
        scenario_id="illegal_parking",
        name="车辆违停检测",
        category="security_incident",
        priority="P0",
        trigger_rule="在普通道路、主通道或禁停区域检测到车辆停留超过配置阈值。",
        required_evidence=("车辆照片", "车牌", "区域规则", "当前地点", "停留时长"),
        robot_behavior=("拍照记录", "通知保安", "继续巡检或按任务策略复查"),
        notification_group="security",
        archive_required=True,
        acceptance_criteria=(
            "只在非停车区域触发违停事件",
            "通知包含地点、车牌和照片",
            "事件可被保安确认、关闭和导出",
        ),
    ),
    "fire_or_smoke": FieldScenario(
        scenario_id="fire_or_smoke",
        name="火灾及烟雾监测",
        category="safety_incident",
        priority="P0",
        trigger_rule="烟雾传感器、温度传感器或视觉烟火检测超过安全阈值。",
        required_evidence=("温度", "烟雾浓度", "现场照片", "传感器时间", "当前地点"),
        robot_behavior=("播放紧急提示", "退到安全距离", "通知保安并归档证据"),
        notification_group="security",
        archive_required=True,
        interrupts_current_task=True,
        acceptance_criteria=(
            "检测到风险后立即播报并通知",
            "机器人不继续进入危险区域",
            "事件保留传感器 freshness 和现场证据",
        ),
    ),
    "trash_bin_full": FieldScenario(
        scenario_id="trash_bin_full",
        name="垃圾桶满溢监测",
        category="facility_service",
        priority="P1",
        trigger_rule="定点垃圾桶视觉或传感器判断满溢比例超过配置阈值。",
        required_evidence=("垃圾桶编号", "满溢比例", "现场照片", "当前地点"),
        robot_behavior=("拍照记录", "通知保洁", "继续巡检"),
        notification_group="cleaning",
        archive_required=True,
        acceptance_criteria=(
            "只通知保洁组，不通知保安组",
            "通知包含垃圾桶编号和照片",
            "保洁处理后可关闭事件",
        ),
    ),
    "urgent_patrol_dispatch": FieldScenario(
        scenario_id="urgent_patrol_dispatch",
        name="突发任务巡检",
        category="mission_control",
        priority="P0",
        trigger_rule="管理员要求机器狗打断当前巡检并前往指定位置。",
        required_evidence=("管理员身份", "目标位置", "任务原因", "当前任务编号"),
        robot_behavior=("暂停当前任务", "进入安全确认", "提交 runtime handoff"),
        notification_group="operations",
        archive_required=True,
        interrupts_current_task=True,
        requires_operator_approval=True,
        acceptance_criteria=(
            "必须由管理员或授权操作员发起",
            "Dashboard 能看到打断、确认、运行和恢复过程",
            "任务结束后生成报告并保留审计记录",
        ),
    ),
    "crowd_gathering": FieldScenario(
        scenario_id="crowd_gathering",
        name="人群聚集检测",
        category="security_incident",
        priority="P1",
        trigger_rule="同一区域人数超过 5 人且持续超过 30 分钟，或复巡仍未消散。",
        required_evidence=("人数", "持续时间", "当前地点", "照片", "复查记录"),
        robot_behavior=("记录证据", "必要时文明提醒", "通知保安关注"),
        notification_group="security",
        archive_required=True,
        acceptance_criteria=(
            "短暂停留不触发告警",
            "通知包含人数、时长和照片",
            "支持复巡后升级或关闭",
        ),
    ),
    "wayfinding_help_point": FieldScenario(
        scenario_id="wayfinding_help_point",
        name="路人指路",
        category="visitor_service",
        priority="P1",
        trigger_rule="在配置的路引帮助点检测到有人停留并触发交互准入。",
        required_evidence=("帮助点编号", "当前地点", "访客问题", "地图版本"),
        robot_behavior=("主动询问需求", "只回答园区路线问题", "不触发机器人任务"),
        notification_group="none",
        archive_required=False,
        acceptance_criteria=(
            "游客问路不会误触发巡检或硬件任务",
            "回答必须基于已审批地图知识",
            "未知地名不编路线",
            "无可靠路径时要求确认或转人工",
        ),
    ),
    "visitor_escort": FieldScenario(
        scenario_id="visitor_escort",
        name="路人带路",
        category="visitor_service",
        priority="P2",
        trigger_rule="授权场景下，访客请求带路到地图数据库中的明确地点。",
        required_evidence=("目的地", "当前位置", "路线", "地图版本", "安全边界"),
        robot_behavior=("低速带路", "保持安全距离", "到达后结束服务"),
        notification_group="none",
        archive_required=True,
        acceptance_criteria=(
            "目的地必须在园区地图中",
            "带路速度和距离满足安全策略",
            "路线不穿越禁行或危险区域",
        ),
    ),
}


def get_field_scenario(scenario_id: str) -> FieldScenario | None:
    """Return a registered field scenario by id."""

    return FIELD_SCENARIOS.get(scenario_id)
