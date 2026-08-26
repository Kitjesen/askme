"""Dashboard page registry.

The static dashboard is still a single-page shell, but product page ownership
must be code-owned and testable. This registry is the backend contract for
customer-visible pages, operator work pages, and governance pages.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DashboardPageSpec:
    """One customer-facing Dashboard page contract."""

    key: str
    path: str
    label: str
    title: str
    kicker: str
    description: str
    audience: str
    section: str
    order: int
    hint: str = ""
    customer_visible: bool = True
    exposes_internal_runtime: bool = False
    primary_endpoint: str = ""
    evidence_promises: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "path": self.path,
            "label": self.label,
            "title": self.title,
            "kicker": self.kicker,
            "description": self.description,
            "audience": self.audience,
            "section": self.section,
            "order": self.order,
            "hint": self.hint,
            "customer_visible": self.customer_visible,
            "exposes_internal_runtime": self.exposes_internal_runtime,
            "primary_endpoint": self.primary_endpoint,
            "evidence_promises": list(self.evidence_promises),
        }


DASHBOARD_SECTIONS: dict[str, dict[str, str]] = {
    "customer": {
        "label": "业务工作区",
        "description": "现场业务、场景、空间与知识入口。",
    },
    "operations": {
        "label": "运行控制",
        "description": "对话、语音和现场处置入口。",
    },
    "governance": {
        "label": "交付治理",
        "description": "验收、证据、权限和审计入口。",
    },
}


DASHBOARD_PAGE_SPECS: tuple[DashboardPageSpec, ...] = (
    DashboardPageSpec(
        key="overview",
        path="/dashboard",
        label="总览",
        title="现场运行总览",
        kicker="运行总览",
        description="查看当前状态、待办事件和交付阻塞。",
        audience="customer",
        section="customer",
        order=10,
        hint="运行状态",
        primary_endpoint="/api/surfaces",
        evidence_promises=("页面分层", "接口边界", "能力地图"),
    ),
    DashboardPageSpec(
        key="scenarios",
        path="/dashboard/scenarios",
        label="场景验收",
        title="场景验收矩阵",
        kicker="客户能看懂",
        description="逐条验收问路、带路、违停、烟火、垃圾桶、陌生人、机器人故障和恶意挡路。",
        audience="customer",
        section="customer",
        order=20,
        hint="业务场景",
        primary_endpoint="/api/scenario-intents",
        evidence_promises=("触发条件", "机器人动作", "通知对象", "归档证据"),
    ),
    DashboardPageSpec(
        key="field",
        path="/dashboard/field",
        label="现场事件",
        title="现场事件处置",
        kicker="安防巡检",
        description="展示异常事件的地点、证据、响应组、处理状态、关闭条件和播报策略。",
        audience="customer",
        section="customer",
        order=30,
        hint="事件处置",
        primary_endpoint="/api/field/events",
        evidence_promises=("现场证据", "通知记录", "处置状态"),
    ),
    DashboardPageSpec(
        key="space",
        path="/dashboard/space",
        label="空间认知",
        title="园区空间认知",
        kicker="问路带路",
        description="管理点位、别名、服务点和带路路线，让机器人能回答和执行受控园区问路。",
        audience="customer",
        section="customer",
        order=40,
        hint="问路带路",
        primary_endpoint="/api/space/health",
        evidence_promises=("点位来源", "路线说明", "带路可行性"),
    ),
    DashboardPageSpec(
        key="knowledge",
        path="/dashboard/knowledge",
        label="知识库",
        title="客户知识库",
        kicker="可审计回答",
        description="先确认机器人知道什么、哪些知识可回答、回答依据在哪里，再处理上传和审批。",
        audience="customer",
        section="customer",
        order=50,
        hint="证据问答",
        primary_endpoint="/api/knowledge/list",
        evidence_promises=("知识状态", "引用依据", "拒答原因"),
    ),
    DashboardPageSpec(
        key="capabilities",
        path="/dashboard/capabilities",
        label="能力中心",
        title="机器人能力中心",
        kicker="客户可见能力",
        description="说明机器人当前能做什么、缺什么、哪些能力需要审批和客户项目启用。",
        audience="customer",
        section="customer",
        order=60,
        hint="能力包",
        primary_endpoint="/api/capability-center",
        evidence_promises=("能力分组", "启用状态", "审批状态"),
    ),
    DashboardPageSpec(
        key="conversation",
        path="/dashboard/conversation",
        label="对话",
        title="语音和文本对话",
        kicker="真实交互",
        description="用于任务下达、问路、知识问答和安全确认，回答需要展示证据和拒答原因。",
        audience="operator",
        section="operations",
        order=70,
        hint="语音文本",
        primary_endpoint="/api/chat",
        evidence_promises=("识别文本", "回答依据", "任务确认"),
    ),
    DashboardPageSpec(
        key="voice",
        path="/dashboard/voice",
        label="语音系统",
        title="小算语音系统",
        kicker="运行控制台",
        description="管理 ASR、LLM、TTS、Prompt、记忆和音频链路，在线切换模型并追踪运行缺口。",
        audience="operator",
        section="operations",
        order=90,
        hint="模型与记忆",
        primary_endpoint="/api/voice/system",
        evidence_promises=("模型路由", "Prompt", "记忆", "链路状态"),
    ),
    DashboardPageSpec(
        key="delivery",
        path="/dashboard/delivery",
        label="交付检查",
        title="交付检查",
        kicker="上线门禁",
        description="把演示、试点、真实硬件、外部通知和客户验收拆成清晰门禁。",
        audience="delivery",
        section="governance",
        order=100,
        hint="可验收",
        primary_endpoint="/api/field/solution-delivery-readiness",
        evidence_promises=("试点状态", "生产声明边界", "缺口清单"),
    ),
    DashboardPageSpec(
        key="audit",
        path="/dashboard/audit",
        label="审计",
        title="审计证据包",
        kicker="交付证据",
        description="查看客户可读的事件证据、复核状态、导出历史和交付声明边界。",
        audience="supervisor",
        section="governance",
        order=110,
        hint="证据包",
        primary_endpoint="/api/audit/events",
        evidence_promises=("审计事件", "复核状态", "导出记录"),
    ),
)


def dashboard_page_slugs(*, include_root: bool = False) -> tuple[str, ...]:
    """Return URL slugs served by the Dashboard shell route."""

    slugs: list[str] = []
    for page in DASHBOARD_PAGE_SPECS:
        if page.path == "/dashboard":
            if include_root:
                slugs.append("")
            continue
        slugs.append(page.path.removeprefix("/dashboard/"))
    return tuple(slugs)


def dashboard_pages_payload(*, route_inventory: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the product page map used by Dashboard and verification tests."""

    route_index = _route_index(route_inventory)
    pages = []
    for page in sorted(DASHBOARD_PAGE_SPECS, key=lambda item: item.order):
        item = page.to_dict()
        item["primary_endpoint_status"] = _primary_endpoint_status(
            item["primary_endpoint"],
            route_index=route_index,
        )
        pages.append(item)
    section_counts = Counter(page["section"] for page in pages)
    audience_counts = Counter(page["audience"] for page in pages)
    internal_page_count = sum(1 for page in pages if page["exposes_internal_runtime"])
    endpoint_statuses = [page["primary_endpoint_status"] for page in pages]
    endpoint_missing_count = sum(1 for status in endpoint_statuses if not status["available"])
    endpoint_internal_count = sum(1 for status in endpoint_statuses if "internal" in status["surfaces"])
    endpoint_unclassified_count = sum(1 for status in endpoint_statuses if "unclassified" in status["surfaces"])
    return {
        "ok": True,
        "pages": pages,
        "sections": DASHBOARD_SECTIONS,
        "summary": {
            "page_count": len(pages),
            "customer_visible_count": sum(1 for page in pages if page["customer_visible"]),
            "internal_page_count": internal_page_count,
            "primary_endpoint_available_count": sum(
                1 for status in endpoint_statuses if status["available"]
            ),
            "primary_endpoint_missing_count": endpoint_missing_count,
            "primary_endpoint_internal_count": endpoint_internal_count,
            "primary_endpoint_unclassified_count": endpoint_unclassified_count,
            "section_counts": dict(sorted(section_counts.items())),
            "audience_counts": dict(sorted(audience_counts.items())),
        },
        "policy": {
            "customer_pages_must_use_business_language": True,
            "internal_runtime_is_not_a_customer_page": internal_page_count == 0,
            "dashboard_shell_uses_registered_pages": True,
            "new_pages_must_have_audience_section_and_primary_endpoint": True,
            "primary_endpoints_must_exist_in_route_inventory": endpoint_missing_count == 0,
            "customer_pages_must_not_point_to_internal_or_unclassified_routes": (
                endpoint_internal_count == 0 and endpoint_unclassified_count == 0
            ),
        },
    }


def _route_index(route_inventory: Mapping[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    routes = route_inventory.get("routes") if isinstance(route_inventory, Mapping) else None
    index: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(routes, list):
        return index
    for route in routes:
        if not isinstance(route, Mapping):
            continue
        path = str(route.get("path") or "")
        if not path:
            continue
        index.setdefault(path, []).append(dict(route))
    return index


def _primary_endpoint_status(
    endpoint: str,
    *,
    route_index: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    routes = list(route_index.get(endpoint, []))
    methods = sorted(
        {
            str(method)
            for route in routes
            for method in (route.get("methods") if isinstance(route.get("methods"), list) else [])
        }
    )
    surfaces = sorted({str(route.get("surface") or "unclassified") for route in routes})
    return {
        "endpoint": endpoint,
        "available": bool(routes),
        "route_count": len(routes),
        "methods": methods,
        "surfaces": surfaces,
        "customer_safe": bool(routes) and "internal" not in surfaces and "unclassified" not in surfaces,
    }


__all__ = [
    "DASHBOARD_PAGE_SPECS",
    "DASHBOARD_SECTIONS",
    "DashboardPageSpec",
    "dashboard_page_slugs",
    "dashboard_pages_payload",
]
