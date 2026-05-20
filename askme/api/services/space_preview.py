"""Shared park-space preview helpers for safe conversation evidence.

The preview path is intentionally read-only. It may resolve destinations and
return evidence, but it must never start escorting, navigation, or robot tasks.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

SpaceDispatch = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]] | dict[str, Any]]

_SPACE_SCENARIOS = {"wayfinding", "visitor_escort"}
_SPACE_SKILLS = {"lookup_place", "answer_wayfinding", "escort_visitor"}
_SPACE_QUERY_TERMS = (
    "在哪",
    "哪里",
    "哪儿",
    "怎么走",
    "怎么去",
    "带我去",
    "请带路",
    "有哪些",
    "有什么",
    "附近",
    "咖啡",
    "厕所",
    "卫生间",
    "停车",
    "西门",
    "东门",
    "南门",
    "北门",
)


def should_resolve_space_preview(text: str, decision: object | None) -> bool:
    """Return whether a preview should include park-space candidate evidence."""

    scenario_id = str(getattr(decision, "scenario_id", "") or "")
    skill_name = str(getattr(decision, "skill_name", "") or "")
    if scenario_id in _SPACE_SCENARIOS or skill_name in _SPACE_SKILLS:
        return True
    return any(term in text for term in _SPACE_QUERY_TERMS)


async def space_resolution_preview(
    *,
    text: str,
    body: dict[str, Any],
    decision: object | None,
    space_dispatch: SpaceDispatch | None,
) -> dict[str, Any] | None:
    """Resolve wayfinding phrases to park points without executing a guide task."""

    if space_dispatch is None or not should_resolve_space_preview(text, decision):
        return None
    request_body = {
        "query": text,
        "current_point_id": body.get("current_point_id") or body.get("from_point_id") or "",
        "service_point_id": body.get("service_point_id") or body.get("help_point_id") or "",
        "operator_id": body.get("operator_id") or "",
    }
    try:
        result = space_dispatch("resolve_destination_payload", request_body)
        if inspect.isawaitable(result):
            result = await result
    except Exception as exc:
        return {
            "available": False,
            "reason": "space_resolution_failed",
            "error": str(exc),
        }
    if not isinstance(result, dict):
        return {
            "available": False,
            "reason": "space_resolution_returned_non_object",
        }
    return {
        "available": True,
        "preview_only": True,
        "does_not_start_guide": True,
        "resolution": result,
    }


def space_resolution_evidence_items(
    space_resolution: dict[str, Any] | None,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Convert a space preview payload into chat-bubble evidence records."""

    if not isinstance(space_resolution, dict) or not space_resolution.get("available"):
        return []
    resolution = space_resolution.get("resolution")
    if not isinstance(resolution, dict):
        return []

    candidates: list[dict[str, Any]] = []
    point = resolution.get("point")
    if isinstance(point, dict):
        candidates.append(point)
    for candidate in resolution.get("candidates") or []:
        if isinstance(candidate, dict):
            candidates.append(candidate)

    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        point_id = str(candidate.get("point_id") or candidate.get("id") or "").strip()
        name = str(candidate.get("point_name") or candidate.get("name") or point_id).strip()
        if not point_id or point_id in seen:
            continue
        seen.add(point_id)
        evidence.append(
            {
                "text": _space_point_text(candidate, resolution=resolution),
                "source": "园区空间认知库",
                "source_system": "space_cognition",
                "record_id": point_id,
                "source_record_id": point_id,
                "category": "园区点位",
                "kind": "space_point",
                "freshness_state": "只读预览",
                "used_in_prompt": False,
                "point_name": name,
                "point_type": candidate.get("point_type") or "",
            }
        )
        if len(evidence) >= limit:
            break
    return evidence


def _space_point_text(point: dict[str, Any], *, resolution: dict[str, Any]) -> str:
    name = str(point.get("point_name") or point.get("name") or point.get("point_id") or "地点")
    point_type = str(point.get("point_type") or resolution.get("point_type_label") or "").strip()
    building = str(point.get("building") or "").strip()
    floor = str(point.get("floor") or "").strip()
    guide_mode = str(point.get("guide_mode") or "").strip()
    parts = [f"空间认知库候选地点：{name}"]
    if point_type:
        parts.append(f"类型：{point_type}")
    if building or floor:
        parts.append(f"位置：{building}{floor}")
    if guide_mode:
        parts.append(f"服务方式：{guide_mode}")
    return "；".join(parts)


__all__ = [
    "SpaceDispatch",
    "should_resolve_space_preview",
    "space_resolution_evidence_items",
    "space_resolution_preview",
]
