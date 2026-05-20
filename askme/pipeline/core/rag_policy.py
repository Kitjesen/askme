"""Shared RAG answer policy helpers.

The memory layer decides whether retrieved knowledge is usable. Every user
surface must then apply the same blocking rules so expired, conflicting, or
unapproved knowledge cannot leak through one channel while another refuses it.
"""

from __future__ import annotations

from typing import Any

BLOCKING_RAG_STATES = frozenset({"filtered", "stale", "conflict", "unapproved"})

DEFAULT_RAG_POLICY_REPLIES = {
    "filtered": "检索到的知识没有通过可信校验，我不能直接回答。",
    "stale": "相关知识已过期或版本不一致，我不能按旧信息回答。请先刷新知识库。",
    "conflict": "相关知识存在冲突，我不能直接给出确定答案。请管理员先确认知识库。",
    "unapproved": "相关知识还没有通过审批，我不能把它作为正式答案。",
    "no_evidence": "我这里没有可靠依据，不能直接回答。请补充位置或让管理员上传知识。",
}


def is_rag_policy_blocking(policy: dict[str, Any] | None) -> bool:
    """Return whether a RAG policy requires deterministic refusal."""
    if not isinstance(policy, dict) or not policy:
        return False
    state = str(policy.get("state") or "").strip().lower()
    action = str(policy.get("action") or "").strip().lower()
    if not state or state == "grounded":
        return False
    return state in BLOCKING_RAG_STATES or action.startswith("refuse")


def forced_rag_reply(
    policy: dict[str, Any] | None,
    *,
    templates: dict[str, str] | None = None,
) -> str:
    """Return the deterministic customer-facing reply for a blocking policy."""
    if not is_rag_policy_blocking(policy):
        return ""
    state = str((policy or {}).get("state") or "").strip().lower()
    clean_templates = {
        str(key).strip().lower(): str(value).strip()
        for key, value in (templates or {}).items()
        if str(key).strip() and str(value).strip()
    }
    return (
        clean_templates.get(state)
        or DEFAULT_RAG_POLICY_REPLIES.get(state)
        or DEFAULT_RAG_POLICY_REPLIES["filtered"]
    )
