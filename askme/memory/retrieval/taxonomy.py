"""Product taxonomy for customer knowledge records.

The taxonomy keeps Knowledge Console categories stable across upload,
catalog storage, retrieval evidence, and UI rendering.
"""

from __future__ import annotations

from typing import Any

KNOWLEDGE_CATEGORY_SCHEMA_VERSION = "askme.knowledge_taxonomy.v1"
DEFAULT_KNOWLEDGE_CATEGORY = "faq"
UNKNOWN_KNOWLEDGE_CATEGORY = "general"

KNOWLEDGE_CATEGORIES: tuple[dict[str, str], ...] = (
    {
        "id": "route",
        "label": "路线与带路",
        "group": "space",
        "description": "道路、路线说明、带路路径、不可通行路段。",
    },
    {
        "id": "location",
        "label": "地点与点位",
        "group": "space",
        "description": "楼宇、入口、卫生间、服务点、打卡点等标准点位。",
    },
    {
        "id": "zone",
        "label": "区域与地图",
        "group": "space",
        "description": "园区分区、禁行区、停车区、巡检区域和地图边界。",
    },
    {
        "id": "merchant",
        "label": "商户与服务",
        "group": "visitor",
        "description": "商户、业态、服务窗口、开放状态和常用别名。",
    },
    {
        "id": "visitor_service",
        "label": "游客服务话术",
        "group": "visitor",
        "description": "问询、欢迎、解释、固定话术和服务边界。",
    },
    {
        "id": "equipment",
        "label": "设备资产",
        "group": "operations",
        "description": "设备位置、编号、状态说明、保养要求和责任人。",
    },
    {
        "id": "inspection",
        "label": "巡检 SOP",
        "group": "operations",
        "description": "巡检步骤、检查项、拍照要求、记录规范。",
    },
    {
        "id": "incident",
        "label": "异常处置",
        "group": "operations",
        "description": "摔倒、卡住、挡路、故障、违停、垃圾桶满等处置流程。",
    },
    {
        "id": "safety",
        "label": "安防应急",
        "group": "safety",
        "description": "陌生人、烟火、人员聚集、危险区域和应急规则。",
    },
    {
        "id": "contact",
        "label": "通知联系人",
        "group": "safety",
        "description": "保安、保洁、值班、物业、钉钉群和升级联系人。",
    },
    {
        "id": "schedule",
        "label": "时间与班次",
        "group": "operations",
        "description": "开放时间、巡检频次、值班时间、任务窗口。",
    },
    {
        "id": "sensor",
        "label": "传感器与协议",
        "group": "technical",
        "description": "摄像头、烟感、温度、电机、定位和第三方设备协议。",
    },
    {
        "id": "policy",
        "label": "管理制度",
        "group": "governance",
        "description": "客户规章、权限、审批、运营要求和交付边界。",
    },
    {
        "id": "faq",
        "label": "常见问答",
        "group": "visitor",
        "description": "客户和访客高频问题及标准回答。",
    },
    {
        "id": "general",
        "label": "其他资料",
        "group": "general",
        "description": "暂未归类但需要保留来源和责任人的资料。",
    },
)

_CATEGORY_BY_ID = {item["id"]: item for item in KNOWLEDGE_CATEGORIES}
SUPPORTED_KNOWLEDGE_CATEGORIES = frozenset(_CATEGORY_BY_ID)
_CATEGORY_ALIASES = {
    "note": "inspection",
    "sop": "inspection",
    "inspection_sop": "inspection",
    "wayfinding": "route",
    "guide": "route",
    "navigation": "route",
    "place": "location",
    "point": "location",
    "space": "zone",
    "area": "zone",
    "shop": "merchant",
    "store": "merchant",
    "vendor": "merchant",
    "visitor": "visitor_service",
    "service": "visitor_service",
    "device": "equipment",
    "asset": "equipment",
    "emergency": "safety",
    "security": "safety",
    "alarm": "incident",
    "exception": "incident",
    "notification": "contact",
    "responder": "contact",
    "calendar": "schedule",
    "time": "schedule",
    "iot": "sensor",
    "protocol": "sensor",
    "rule": "policy",
    "rules": "policy",
    "制度": "policy",
    "路线": "route",
    "点位": "location",
    "地点": "location",
    "区域": "zone",
    "商户": "merchant",
    "设备": "equipment",
    "巡检": "inspection",
    "异常": "incident",
    "安防": "safety",
    "联系人": "contact",
    "时间": "schedule",
    "传感器": "sensor",
    "问答": "faq",
}


def normalize_knowledge_category(value: Any, *, default: str = DEFAULT_KNOWLEDGE_CATEGORY) -> str:
    """Return a stable category id for storage and retrieval metadata."""

    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not text:
        return default if default in SUPPORTED_KNOWLEDGE_CATEGORIES else DEFAULT_KNOWLEDGE_CATEGORY
    normalized = _CATEGORY_ALIASES.get(text, text)
    return normalized if normalized in SUPPORTED_KNOWLEDGE_CATEGORIES else UNKNOWN_KNOWLEDGE_CATEGORY


def knowledge_category_metadata(value: Any) -> dict[str, str]:
    """Return public metadata for one category id or alias."""

    category = normalize_knowledge_category(value)
    item = _CATEGORY_BY_ID.get(category, _CATEGORY_BY_ID[UNKNOWN_KNOWLEDGE_CATEGORY])
    return {
        "id": item["id"],
        "label": item["label"],
        "group": item["group"],
        "description": item["description"],
        "schema_version": KNOWLEDGE_CATEGORY_SCHEMA_VERSION,
    }


def knowledge_category_taxonomy_payload() -> dict[str, Any]:
    """Return the category taxonomy exposed by API payloads."""

    return {
        "schema_version": KNOWLEDGE_CATEGORY_SCHEMA_VERSION,
        "default_category": DEFAULT_KNOWLEDGE_CATEGORY,
        "unknown_category": UNKNOWN_KNOWLEDGE_CATEGORY,
        "categories": [dict(item) for item in KNOWLEDGE_CATEGORIES],
    }
