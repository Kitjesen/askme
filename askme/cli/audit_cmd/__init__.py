"""Unified audit CLI commands extracted from askme.cli."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _run_unified_audit_events(
    *,
    limit: int,
    source: str,
    operator_id: str,
    action: str,
    outcome: str,
    q: str,
    since: str,
    until: str,
    skill_audit: str = "",
    field_action_audit: str = "",
    field_event_archive: str = "",
    runtime_audit: str = "",
    review_path: str = "",
) -> dict[str, Any]:
    from askme.audit import AuditQueryService
    from askme.config import get_config

    config = get_config()
    return AuditQueryService(
        config,
        paths=_unified_audit_paths_from_cli(
            config,
            skill_audit=skill_audit,
            field_action_audit=field_action_audit,
            field_event_archive=field_event_archive,
            runtime_audit=runtime_audit,
            review_path=review_path,
        ),
    ).query(
        limit=limit,
        source=source,
        operator_id=operator_id,
        action=action,
        outcome=outcome,
        q=q,
        since=since,
        until=until,
    )


def _emit_unified_audit_events_payload(
    payload: dict[str, Any],
    *,
    review_queue_only: bool,
) -> None:
    product = payload.get("product_summary") if isinstance(payload.get("product_summary"), dict) else {}
    print(
        "audit-events: "
        f"records={payload.get('count', 0)} "
        f"filtered={payload.get('filtered_total', 0)} "
        f"status={product.get('status') or '-'} "
        f"review_queue={product.get('requires_review_count', 0)}"
    )
    customer_report = (
        payload.get("customer_report") if isinstance(payload.get("customer_report"), dict) else {}
    )
    if customer_report:
        print(f"customer-status: {customer_report.get('status_label') or '-'}")
        summary_sentence = customer_report.get("summary_sentence")
        if summary_sentence:
            print(f"summary: {summary_sentence}")
    records_key = "review_queue" if review_queue_only else "records"
    records = payload.get(records_key) if isinstance(payload.get(records_key), list) else []
    if not records:
        print("records: none")
        return
    print("records:")
    for record in records[:20]:
        print(
            "  - "
            f"{record.get('record_id') or '-'} "
            f"{record.get('severity') or '-'} "
            f"{record.get('source') or '-'} "
            f"{record.get('action') or '-'} "
            f"{record.get('outcome') or '-'} "
            f"owner={record.get('customer_copy', {}).get('review_owner') if isinstance(record.get('customer_copy'), dict) else '-'}"
        )


def _run_unified_audit_review(
    *,
    record_id: str,
    reviewer_id: str,
    decision: str,
    note: str,
    skill_audit: str = "",
    field_action_audit: str = "",
    field_event_archive: str = "",
    runtime_audit: str = "",
    review_path: str = "",
) -> dict[str, Any]:
    from askme.audit import AuditQueryService, AuditReviewService
    from askme.config import get_config

    config = get_config()
    paths = _unified_audit_paths_from_cli(
        config,
        skill_audit=skill_audit,
        field_action_audit=field_action_audit,
        field_event_archive=field_event_archive,
        runtime_audit=runtime_audit,
        review_path=review_path,
    )
    service = AuditQueryService(config, paths=paths)
    if not service.record_exists(record_id):
        return {
            "ok": False,
            "reason": "audit_record_not_found",
            "record_id": str(record_id or ""),
        }
    payload = AuditReviewService(config, path=paths.audit_reviews).submit(
        record_id=record_id,
        reviewer_id=reviewer_id,
        decision=decision,
        note=note,
    )
    if payload.get("ok"):
        refreshed = AuditQueryService(config, paths=paths).query(limit=25)
        payload["post_review"] = {
            "requires_review_count": refreshed.get("product_summary", {}).get("requires_review_count", 0),
            "review_queue_count": len(refreshed.get("review_queue") or []),
            "customer_status_label": refreshed.get("product_summary", {}).get("customer_status_label", ""),
        }
    return payload


def _emit_unified_audit_review_payload(payload: dict[str, Any]) -> None:
    record = payload.get("record") if isinstance(payload.get("record"), dict) else {}
    post_review = payload.get("post_review") if isinstance(payload.get("post_review"), dict) else {}
    print(
        "audit-review: "
        f"ok={bool(payload.get('ok'))} "
        f"record={payload.get('record_id') or record.get('record_id') or '-'} "
        f"decision={record.get('decision') or '-'} "
        f"clears_review={bool(record.get('clears_review'))}"
    )
    if payload.get("reason"):
        print(f"reason: {payload.get('reason')}")
    if post_review:
        print(
            "post-review: "
            f"queue={post_review.get('review_queue_count', 0)} "
            f"requires_review={post_review.get('requires_review_count', 0)} "
            f"status={post_review.get('customer_status_label') or '-'}"
        )
    if payload.get("path"):
        print(f"path: {payload.get('path')}")


def _unified_audit_paths_from_cli(
    config: dict[str, Any],
    *,
    skill_audit: str = "",
    field_action_audit: str = "",
    field_event_archive: str = "",
    runtime_audit: str = "",
    review_path: str = "",
):
    from askme.audit import AuditQueryService
    from askme.audit.query import AuditPaths

    default_paths = AuditQueryService(config)._paths
    return AuditPaths(
        skill_audit=Path(skill_audit) if skill_audit else default_paths.skill_audit,
        field_action_audit=(
            Path(field_action_audit) if field_action_audit else default_paths.field_action_audit
        ),
        field_event_archive=(
            Path(field_event_archive) if field_event_archive else default_paths.field_event_archive
        ),
        runtime_audit=Path(runtime_audit) if runtime_audit else default_paths.runtime_audit,
        audit_reviews=Path(review_path) if review_path else default_paths.audit_reviews,
    )
