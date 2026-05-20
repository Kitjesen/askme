"""Read-only unified audit timeline for product governance views."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from askme.skills.governance.audit import default_skill_audit_path

from .review import AuditReviewService, decision_clears_review


@dataclass(frozen=True)
class AuditPaths:
    skill_audit: Path
    field_action_audit: Path | None = None
    field_event_archive: Path | None = None
    runtime_audit: Path | None = None
    audit_reviews: Path | None = None


@dataclass(frozen=True)
class AuditRecordSet:
    source_name: str
    path: str
    records: tuple[dict[str, Any], ...]
    indexes: dict[str, dict[str, frozenset[int]]]
    cache_hit: bool = False
    indexed: bool = True


_SCOPE_FIELDS = (
    "tenant_id",
    "delivery_namespace",
    "customer_id",
    "project_id",
    "site_id",
    "managed_object_id",
)
_INDEX_FIELDS = ("source", "operator_id", "action", "outcome", *_SCOPE_FIELDS)
_AUDIT_INDEX_CACHE: dict[tuple[str, int, int, int, str], AuditRecordSet] = {}
_AUDIT_INDEX_LOCK = RLock()
_AUDIT_INDEX_MAX_ENTRIES = 32


class AuditQueryService:
    """Aggregate append-only audit sources into one customer-facing timeline."""

    def __init__(self, config: dict[str, Any] | None = None, *, paths: AuditPaths | None = None) -> None:
        self._config = config or {}
        self._paths = paths or self._paths_from_config(self._config)

    def query(
        self,
        *,
        limit: int = 100,
        source: str = "",
        operator_id: str = "",
        action: str = "",
        outcome: str = "",
        tenant_id: str = "",
        delivery_namespace: str = "",
        customer_id: str = "",
        project_id: str = "",
        site_id: str = "",
        managed_object_id: str = "",
        q: str = "",
        since: str = "",
        until: str = "",
    ) -> dict[str, Any]:
        safe_limit = max(1, min(int(limit or 100), 500))
        since_boundary = _time_boundary(since)
        until_boundary = _time_boundary(until)
        record_sets = self._record_sets()
        records = [record for record_set in record_sets for record in record_set.records]
        filtered, query_engine = _query_record_sets(
            record_sets,
            source=source,
            operator_id=operator_id,
            action=action,
            outcome=outcome,
            tenant_id=tenant_id,
            delivery_namespace=delivery_namespace,
            customer_id=customer_id,
            project_id=project_id,
            site_id=site_id,
            managed_object_id=managed_object_id,
            q=q,
            since=since_boundary,
            until=until_boundary,
        )
        filtered.sort(key=lambda item: item.get("sort_at") or 0.0, reverse=True)
        review_decisions = _review_decisions(self._paths.audit_reviews, self._config)
        public_filtered = [
            _apply_review_decision(_public_record(item), review_decisions)
            for item in filtered
        ]
        source_health = self._source_health()
        review_integrity = self._review_integrity()
        customer_report = _customer_report(
            public_filtered,
            source_health=source_health,
            review_integrity=review_integrity,
        )
        audit_readiness = _audit_readiness(
            public_filtered,
            source_health=source_health,
            review_integrity=review_integrity,
        )
        visible = public_filtered[:safe_limit]
        omitted_count = max(0, len(public_filtered) - len(visible))
        return {
            "records": visible,
            "count": len(visible),
            "total": len(records),
            "filtered_total": len(filtered),
            "limit": safe_limit,
            "truncated": bool(omitted_count),
            "omitted_record_count": omitted_count,
            "filters": {
                "source": source,
                "operator_id": operator_id,
                "action": action,
                "outcome": outcome,
                "tenant_id": tenant_id,
                "delivery_namespace": delivery_namespace,
                "customer_id": customer_id,
                "project_id": project_id,
                "site_id": site_id,
                "managed_object_id": managed_object_id,
                "q": q,
                "since": since,
                "until": until,
            },
            "time_window": _time_window_payload(since_boundary, until_boundary),
            "summary": _summary(public_filtered),
            "product_summary": _product_summary(public_filtered),
            "customer_report": customer_report,
            "delivery_dossier": _delivery_dossier(
                public_filtered,
                customer_report=customer_report,
                audit_readiness=audit_readiness,
                review_integrity=review_integrity,
            ),
            "audit_readiness": audit_readiness,
            "review_queue": _review_queue(public_filtered),
            "sources": {
                "skill_audit": str(self._paths.skill_audit),
                "field_action_audit": str(self._paths.field_action_audit or ""),
                "field_event_archive": str(self._paths.field_event_archive or ""),
                "runtime_audit": str(self._paths.runtime_audit or ""),
                "audit_reviews": str(self._paths.audit_reviews or ""),
            },
            "source_health": source_health,
            "review_integrity": review_integrity,
            "query_engine": query_engine,
        }

    def _records(self) -> list[dict[str, Any]]:
        return [record for record_set in self._record_sets() for record in record_set.records]

    def record_exists(self, record_id: str) -> bool:
        """Return whether a public audit record id exists in the current sources."""
        target = str(record_id or "").strip()
        if not target:
            return False
        return any(_record_id(record) == target for record in self._records())

    def _record_sets(self) -> list[AuditRecordSet]:
        return [
            self._skill_record_set(),
            self._field_record_set(),
            self._runtime_record_set(),
        ]

    def _skill_records(self) -> list[dict[str, Any]]:
        return list(self._skill_record_set().records)

    def _skill_record_set(self) -> AuditRecordSet:
        return _indexed_jsonl_record_set(
            self._paths.skill_audit,
            limit=500,
            normalize=_normalize_skill,
            source_name="skill_audit",
        )

    def _field_records(self) -> list[dict[str, Any]]:
        return list(self._field_record_set().records)

    def _field_record_set(self) -> AuditRecordSet:
        if self._paths.field_action_audit and self._paths.field_action_audit.is_file():
            return _indexed_jsonl_record_set(
                self._paths.field_action_audit,
                limit=1000,
                normalize=_normalize_field_action,
                source_name="field_action_audit",
            )
        if self._paths.field_event_archive and self._paths.field_event_archive.is_file():
            return _indexed_archive_record_set(self._paths.field_event_archive)
        return _empty_record_set("field_action_audit", self._paths.field_action_audit)

    def _runtime_records(self) -> list[dict[str, Any]]:
        return list(self._runtime_record_set().records)

    def _runtime_record_set(self) -> AuditRecordSet:
        path = self._paths.runtime_audit
        if not path or not path.is_file():
            return _empty_record_set("runtime_audit", path)
        return _indexed_jsonl_record_set(
            path,
            limit=1000,
            normalize=_normalize_runtime,
            source_name="runtime_audit",
        )

    def _source_health(self) -> dict[str, dict[str, Any]]:
        return {
            "skill_audit": _source_health_item(self._paths.skill_audit),
            "field_action_audit": _source_health_item(self._paths.field_action_audit),
            "field_event_archive": _source_health_item(self._paths.field_event_archive),
            "runtime_audit": _source_health_item(self._paths.runtime_audit),
            "audit_reviews": _source_health_item(self._paths.audit_reviews),
        }

    def _review_integrity(self) -> dict[str, Any]:
        if self._paths.audit_reviews is None:
            return {"valid": True, "exists": False, "checked_count": 0, "failures": []}
        return AuditReviewService(self._config, path=self._paths.audit_reviews).integrity()

    @staticmethod
    def _paths_from_config(config: dict[str, Any]) -> AuditPaths:
        field_cfg = config.get("field_operations") if isinstance(config.get("field_operations"), dict) else {}
        field_audit_cfg = field_cfg.get("action_audit") if isinstance(field_cfg.get("action_audit"), dict) else {}
        runtime_handoff_cfg = (
            config.get("runtime_handoff") if isinstance(config.get("runtime_handoff"), dict) else {}
        )
        runtime_handoff_audit_cfg = (
            runtime_handoff_cfg.get("audit")
            if isinstance(runtime_handoff_cfg.get("audit"), dict)
            else {}
        )
        runtime_cfg = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
        handoff_cfg = runtime_cfg.get("handoff") if isinstance(runtime_cfg.get("handoff"), dict) else {}
        runtime_audit_cfg = handoff_cfg.get("audit") if isinstance(handoff_cfg.get("audit"), dict) else {}
        skill_cfg = config.get("skills") if isinstance(config.get("skills"), dict) else {}
        audit_cfg = config.get("audit") if isinstance(config.get("audit"), dict) else {}
        audit_review_cfg = (
            audit_cfg.get("review") if isinstance(audit_cfg.get("review"), dict) else {}
        )
        return AuditPaths(
            skill_audit=Path(str(skill_cfg.get("audit_path") or default_skill_audit_path())),
            field_action_audit=_optional_path(field_audit_cfg.get("path") or field_cfg.get("action_audit_path")),
            field_event_archive=_optional_path(field_cfg.get("archive_path")),
            runtime_audit=_optional_path(
                runtime_handoff_audit_cfg.get("path")
                or runtime_handoff_audit_cfg.get("jsonl_path")
                or runtime_handoff_cfg.get("audit_log_path")
                or runtime_audit_cfg.get("path")
                or runtime_audit_cfg.get("jsonl_path")
                or handoff_cfg.get("audit_log_path")
            ),
            audit_reviews=(
                _optional_path(
                    audit_review_cfg.get("path")
                    or audit_review_cfg.get("jsonl_path")
                    or audit_cfg.get("review_path")
                )
                or AuditReviewService(config).path
            ),
        )


def _optional_path(value: Any) -> Path | None:
    text = str(value or "").strip()
    return Path(text) if text else None


def _indexed_jsonl_record_set(
    path: Path,
    *,
    limit: int,
    normalize: Callable[[dict[str, Any]], dict[str, Any]],
    source_name: str,
) -> AuditRecordSet:
    if not path.is_file():
        return _empty_record_set(source_name, path)
    try:
        stat = path.stat()
    except OSError:
        return _empty_record_set(source_name, path)

    normalizer_name = getattr(normalize, "__name__", source_name)
    cache_key = (
        str(path.resolve()),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        max(1, int(limit)),
        normalizer_name,
    )
    with _AUDIT_INDEX_LOCK:
        cached = _AUDIT_INDEX_CACHE.get(cache_key)
        if cached is not None:
            return AuditRecordSet(
                source_name=cached.source_name,
                path=cached.path,
                records=cached.records,
                indexes=cached.indexes,
                cache_hit=True,
                indexed=cached.indexed,
            )

    records = tuple(normalize(item) for item in _read_jsonl(path, limit=limit))
    record_set = _build_record_set(
        source_name=source_name,
        path=str(path),
        records=records,
        cache_hit=False,
    )
    with _AUDIT_INDEX_LOCK:
        _prune_stale_index_entries(path=path, limit=limit, normalizer_name=normalizer_name)
        _AUDIT_INDEX_CACHE[cache_key] = record_set
        while len(_AUDIT_INDEX_CACHE) > _AUDIT_INDEX_MAX_ENTRIES:
            _AUDIT_INDEX_CACHE.pop(next(iter(_AUDIT_INDEX_CACHE)))
    return record_set


def _indexed_archive_record_set(path: Path) -> AuditRecordSet:
    if not path.is_file():
        return _empty_record_set("field_event_archive", path)
    try:
        stat = path.stat()
    except OSError:
        return _empty_record_set("field_event_archive", path)

    cache_key = (
        str(path.resolve()),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        500,
        "_field_records_from_archive",
    )
    with _AUDIT_INDEX_LOCK:
        cached = _AUDIT_INDEX_CACHE.get(cache_key)
        if cached is not None:
            return AuditRecordSet(
                source_name=cached.source_name,
                path=cached.path,
                records=cached.records,
                indexes=cached.indexes,
                cache_hit=True,
                indexed=cached.indexed,
            )

    record_set = _build_record_set(
        source_name="field_event_archive",
        path=str(path),
        records=tuple(_field_records_from_archive(path)),
        cache_hit=False,
    )
    with _AUDIT_INDEX_LOCK:
        _prune_stale_index_entries(
            path=path,
            limit=500,
            normalizer_name="_field_records_from_archive",
        )
        _AUDIT_INDEX_CACHE[cache_key] = record_set
        while len(_AUDIT_INDEX_CACHE) > _AUDIT_INDEX_MAX_ENTRIES:
            _AUDIT_INDEX_CACHE.pop(next(iter(_AUDIT_INDEX_CACHE)))
    return record_set


def _prune_stale_index_entries(*, path: Path, limit: int, normalizer_name: str) -> None:
    resolved = str(path.resolve())
    stale = [
        key
        for key in _AUDIT_INDEX_CACHE
        if key[0] == resolved and key[3] == max(1, int(limit)) and key[4] == normalizer_name
    ]
    for key in stale:
        _AUDIT_INDEX_CACHE.pop(key, None)


def _empty_record_set(source_name: str, path: Path | None) -> AuditRecordSet:
    return AuditRecordSet(
        source_name=source_name,
        path=str(path or ""),
        records=(),
        indexes={field: {} for field in _INDEX_FIELDS},
        cache_hit=False,
        indexed=False,
    )


def _build_record_set(
    *,
    source_name: str,
    path: str,
    records: tuple[dict[str, Any], ...],
    cache_hit: bool,
) -> AuditRecordSet:
    mutable_indexes: dict[str, dict[str, set[int]]] = {field: {} for field in _INDEX_FIELDS}
    for index, record in enumerate(records):
        for field in _INDEX_FIELDS:
            value = str(record.get(field) or "").strip()
            if not value:
                continue
            mutable_indexes[field].setdefault(value, set()).add(index)
    indexes = {
        field: {value: frozenset(indexes) for value, indexes in field_index.items()}
        for field, field_index in mutable_indexes.items()
    }
    return AuditRecordSet(
        source_name=source_name,
        path=path,
        records=records,
        indexes=indexes,
        cache_hit=cache_hit,
        indexed=True,
    )


def _query_record_sets(
    record_sets: list[AuditRecordSet],
    *,
    source: str,
    operator_id: str,
    action: str,
    outcome: str,
    tenant_id: str,
    delivery_namespace: str,
    customer_id: str,
    project_id: str,
    site_id: str,
    managed_object_id: str,
    q: str,
    since: dict[str, Any],
    until: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    total_records = sum(len(record_set.records) for record_set in record_sets)
    scanned_records = 0
    candidate_records = 0
    index_used = False
    source_stats: list[dict[str, Any]] = []
    for record_set in record_sets:
        candidate_indexes, used_exact_index = _candidate_indexes(
            record_set,
            source=source,
            operator_id=operator_id,
            action=action,
            outcome=outcome,
            tenant_id=tenant_id,
            delivery_namespace=delivery_namespace,
            customer_id=customer_id,
            project_id=project_id,
            site_id=site_id,
            managed_object_id=managed_object_id,
        )
        index_used = index_used or used_exact_index
        candidate_records += len(candidate_indexes)
        scanned_records += len(candidate_indexes)
        for index in candidate_indexes:
            record = record_set.records[index]
            if _matches(
                record,
                source=source,
                operator_id=operator_id,
                action=action,
                outcome=outcome,
                tenant_id=tenant_id,
                delivery_namespace=delivery_namespace,
                customer_id=customer_id,
                project_id=project_id,
                site_id=site_id,
                managed_object_id=managed_object_id,
                q=q,
                since=since,
                until=until,
            ):
                filtered.append(record)
        source_stats.append({
            "source_name": record_set.source_name,
            "path": record_set.path,
            "records": len(record_set.records),
            "candidates": len(candidate_indexes),
            "cache_hit": record_set.cache_hit,
            "indexed": record_set.indexed,
        })

    return filtered, {
        "indexed": True,
        "index_used": index_used,
        "total_records": total_records,
        "candidate_records": candidate_records,
        "scanned_records": scanned_records,
        "scan_avoidance_records": max(0, total_records - scanned_records),
        "cache_hits": sum(1 for record_set in record_sets if record_set.cache_hit),
        "cache_misses": sum(
            1 for record_set in record_sets if record_set.indexed and not record_set.cache_hit
        ),
        "sources": source_stats,
    }


def _candidate_indexes(
    record_set: AuditRecordSet,
    *,
    source: str,
    operator_id: str,
    action: str,
    outcome: str,
    tenant_id: str,
    delivery_namespace: str,
    customer_id: str,
    project_id: str,
    site_id: str,
    managed_object_id: str,
) -> tuple[list[int], bool]:
    exact_filters = {
        "source": str(source or "").strip(),
        "operator_id": str(operator_id or "").strip(),
        "action": str(action or "").strip(),
        "outcome": str(outcome or "").strip(),
        "tenant_id": str(tenant_id or "").strip(),
        "delivery_namespace": str(delivery_namespace or "").strip(),
        "customer_id": str(customer_id or "").strip(),
        "project_id": str(project_id or "").strip(),
        "site_id": str(site_id or "").strip(),
        "managed_object_id": str(managed_object_id or "").strip(),
    }
    matching_sets: list[frozenset[int]] = []
    for field, value in exact_filters.items():
        if not value:
            continue
        matching_sets.append(record_set.indexes.get(field, {}).get(value, frozenset()))

    if not matching_sets:
        return list(range(len(record_set.records))), False
    if any(not indexes for indexes in matching_sets):
        return [], True
    smallest, *rest = sorted(matching_sets, key=len)
    candidates = set(smallest)
    for indexes in rest:
        candidates.intersection_update(indexes)
        if not candidates:
            break
    return sorted(candidates), True


def _read_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    safe_limit = max(1, limit)
    records: deque[dict[str, Any]] = deque(maxlen=safe_limit)
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    records.append(item)
    except OSError:
        return []
    return list(records)


def _normalize_skill(item: dict[str, Any]) -> dict[str, Any]:
    timestamp = str(item.get("timestamp") or "")
    scope = _scope_payload(item)
    return {
        "source": "skill",
        "category": str(item.get("event_type") or "skill"),
        "action": str(item.get("action") or item.get("skill_name") or ""),
        "outcome": str(item.get("status") or ""),
        "operator_id": str(item.get("operator_id") or ""),
        "subject": str(item.get("skill_name") or ""),
        "reason": str(item.get("reason") or ""),
        "message": str(item.get("result_preview") or item.get("user_text_preview") or ""),
        "timestamp": timestamp,
        "sort_at": _sort_time(timestamp),
        "raw": item,
        **scope,
    }


def _normalize_field_action(item: dict[str, Any]) -> dict[str, Any]:
    audit = item.get("audit") if isinstance(item.get("audit"), dict) else {}
    event = item.get("event") if isinstance(item.get("event"), dict) else {}
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    delivery = item.get("delivery") if isinstance(item.get("delivery"), dict) else {}
    runtime_delivery = (
        item.get("runtime_delivery") if isinstance(item.get("runtime_delivery"), dict) else {}
    )
    scope = _scope_payload(item, audit, event, payload, delivery, runtime_delivery)
    at = audit.get("at") or item.get("created_at")
    return {
        "source": "field",
        "category": str(item.get("kind") or "field_event_action"),
        "action": str(audit.get("action") or ""),
        "outcome": str(audit.get("outcome") or ""),
        "operator_id": str(audit.get("operator_id") or ""),
        "subject": str(item.get("event_id") or ""),
        "reason": str(audit.get("reason") or ""),
        "message": str(audit.get("note") or audit.get("authorization_reason") or ""),
        "timestamp": _iso_from_any(at),
        "sort_at": _sort_time(at),
        "raw": item,
        **scope,
    }


def _field_records_from_archive(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for event in _read_jsonl(path, limit=500):
        audits = event.get("action_audit") if isinstance(event.get("action_audit"), list) else []
        for audit in audits:
            if isinstance(audit, dict):
                records.append(_normalize_field_action({
                    "kind": "field_event_action",
                    "event_id": event.get("event_id"),
                    "created_at": audit.get("at"),
                    "audit": audit,
                    "event": event,
                    "payload": event.get("payload") if isinstance(event.get("payload"), dict) else {},
                    "delivery": event.get("delivery") if isinstance(event.get("delivery"), dict) else {},
                    "runtime_delivery": (
                        event.get("runtime_delivery")
                        if isinstance(event.get("runtime_delivery"), dict)
                        else {}
                    ),
                }))
    return records


def _normalize_runtime(item: dict[str, Any]) -> dict[str, Any]:
    action = item.get("action") if isinstance(item.get("action"), dict) else {}
    event = item.get("event") if isinstance(item.get("event"), dict) else {}
    handoff = item.get("handoff") if isinstance(item.get("handoff"), dict) else {}
    run = item.get("run") if isinstance(item.get("run"), dict) else {}
    scope = _scope_payload(item, action, event, handoff, run)
    kind = str(item.get("kind") or "runtime")
    return {
        "source": "runtime",
        "category": kind,
        "action": str(action.get("action") or event.get("event_type") or kind),
        "outcome": str(action.get("outcome") or event.get("status") or item.get("state") or ""),
        "operator_id": str(action.get("operator_id") or event.get("operator_id") or ""),
        "subject": str(item.get("run_id") or event.get("run_id") or item.get("handoff_id") or ""),
        "reason": str(action.get("reason") or event.get("reason") or ""),
        "message": str(event.get("message") or action.get("note") or ""),
        "timestamp": _iso_from_any(item.get("created_at")),
        "sort_at": _sort_time(item.get("created_at")),
        "raw": item,
        **scope,
    }


def _scope_payload(*sources: dict[str, Any]) -> dict[str, str]:
    payload: dict[str, str] = {field: "" for field in _SCOPE_FIELDS}
    for field in _SCOPE_FIELDS:
        for source in sources:
            if not isinstance(source, dict):
                continue
            value = _scope_value(source, field)
            if value:
                payload[field] = value
                break
    has_project_scope = any(
        payload.get(field)
        for field in ("customer_id", "project_id", "site_id", "managed_object_id")
    )
    if has_project_scope:
        payload["tenant_id"] = payload["tenant_id"] or "default"
        payload["delivery_namespace"] = payload["delivery_namespace"] or "default"
    return payload


def _scope_value(source: dict[str, Any], field: str) -> str:
    project_scope = source.get("project_scope") if isinstance(source.get("project_scope"), dict) else {}
    aliases = {
        "managed_object_id": ("managed_object_id", "object_id", "target_object_id"),
    }.get(field, (field,))
    for key in aliases:
        value = source.get(key)
        if value not in (None, ""):
            return str(value).strip()
    value = project_scope.get(field)
    if value not in (None, ""):
        return str(value).strip()
    return ""


def _matches(
    record: dict[str, Any],
    *,
    source: str,
    operator_id: str,
    action: str,
    outcome: str,
    tenant_id: str,
    delivery_namespace: str,
    customer_id: str,
    project_id: str,
    site_id: str,
    managed_object_id: str,
    q: str,
    since: dict[str, Any],
    until: dict[str, Any],
) -> bool:
    if source and record.get("source") != source:
        return False
    if operator_id and record.get("operator_id") != operator_id:
        return False
    if action and record.get("action") != action:
        return False
    if outcome and record.get("outcome") != outcome:
        return False
    exact_scope = {
        "tenant_id": tenant_id,
        "delivery_namespace": delivery_namespace,
        "customer_id": customer_id,
        "project_id": project_id,
        "site_id": site_id,
        "managed_object_id": managed_object_id,
    }
    for field, value in exact_scope.items():
        expected = str(value or "").strip()
        if expected and str(record.get(field) or "").strip() != expected:
            return False
    if q:
        haystack = json.dumps(_public_record(record), ensure_ascii=False)
        if q.lower() not in haystack.lower():
            return False
    sort_at = float(record.get("sort_at") or 0.0)
    since_sort = since.get("sort_at")
    if isinstance(since_sort, (int, float)) and sort_at < float(since_sort):
        return False
    until_sort = until.get("sort_at")
    if isinstance(until_sort, (int, float)) and sort_at > float(until_sort):
        return False
    return True


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_source = Counter(str(item.get("source") or "unknown") for item in records)
    by_outcome = Counter(str(item.get("outcome") or "unknown") for item in records)
    by_operator = Counter(str(item.get("operator_id") or "unknown") for item in records)
    by_severity = Counter(str(item.get("severity") or "unknown") for item in records)
    by_customer = Counter(str(item.get("customer_id") or "unscoped") for item in records)
    by_project = Counter(str(item.get("project_id") or "unscoped") for item in records)
    by_site = Counter(str(item.get("site_id") or "unscoped") for item in records)
    by_managed_object = Counter(
        str(item.get("managed_object_id") or "unscoped") for item in records
    )
    review_count = sum(1 for item in records if item.get("requires_review"))
    return {
        "by_source": dict(sorted(by_source.items())),
        "by_outcome": dict(sorted(by_outcome.items())),
        "by_severity": dict(sorted(by_severity.items())),
        "by_customer": dict(sorted(by_customer.items())),
        "by_project": dict(sorted(by_project.items())),
        "by_site": dict(sorted(by_site.items())),
        "by_managed_object": dict(sorted(by_managed_object.items())),
        "top_operators": dict(by_operator.most_common(8)),
        "requires_review_count": review_count,
    }


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    severity = _severity_for(record)
    requires_review = _requires_review(record, severity=severity)
    source = str(record.get("source") or "")
    subject = str(record.get("subject") or "")
    action = str(record.get("action") or "")
    outcome = str(record.get("outcome") or "")
    evidence_refs = _evidence_refs(record)
    return {
        "record_id": _record_id(record),
        "source": source,
        "category": record.get("category") or "",
        "action": action,
        "action_label": _action_label(action or str(record.get("category") or "")),
        "outcome": outcome,
        "outcome_label": _outcome_label(outcome),
        "operator_id": record.get("operator_id") or "",
        "tenant_id": record.get("tenant_id") or "",
        "delivery_namespace": record.get("delivery_namespace") or "",
        "customer_id": record.get("customer_id") or "",
        "project_id": record.get("project_id") or "",
        "site_id": record.get("site_id") or "",
        "managed_object_id": record.get("managed_object_id") or "",
        "actor_type": _actor_type(record),
        "resource_type": _resource_type(source),
        "resource_id": subject,
        "subject": subject,
        "customer_label": _customer_label(source),
        "display_title": _display_title(record, source=source, subject=subject),
        "customer_impact": _customer_impact(source),
        "customer_copy": _customer_copy(
            record,
            source=source,
            subject=subject,
            severity=severity,
            requires_review=requires_review,
        ),
        "governance": _governance_payload(
            severity=severity,
            requires_review=requires_review,
        ),
        "severity": severity,
        "severity_label": _severity_label(severity),
        "requires_review": requires_review,
        "review_reason": _review_reason(record, severity=severity, requires_review=requires_review),
        "recommended_action": _recommended_action(
            record,
            severity=severity,
            requires_review=requires_review,
        ),
        "acceptance": _acceptance_payload(
            severity=severity,
            requires_review=requires_review,
            evidence_refs=evidence_refs,
        ),
        "reason": record.get("reason") or "",
        "message": record.get("message") or "",
        "timestamp": record.get("timestamp") or "",
        "integrity": _integrity_summary(record),
        "evidence_refs": evidence_refs,
        "evidence_status": _evidence_status(evidence_refs),
    }


def _product_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    review_count = sum(1 for item in records if item.get("requires_review"))
    high_or_critical = sum(1 for item in records if item.get("severity") in {"critical", "high"})
    signed_count = sum(1 for item in records if item.get("integrity", {}).get("signed"))
    hash_chain_count = sum(1 for item in records if item.get("integrity", {}).get("hash_chain"))
    status = "needs_review" if review_count else "auditable"
    customer_ready = review_count == 0
    return {
        "status": status,
        "customer_status_label": "待主管复核" if review_count else "可交付审计包",
        "customer_status": (
            "存在待复核记录，暂不能交付给客户验收"
            if review_count
            else "审计记录已整理完成，可作为客户验收证据"
        ),
        "record_count": len(records),
        "requires_review_count": review_count,
        "high_or_critical_count": high_or_critical,
        "integrity": {
            "signed_record_count": signed_count,
            "hash_chained_record_count": hash_chain_count,
        },
        "handoff": {
            "customer_ready": customer_ready,
            "release_claim": (
                "可作为客户验收证据导出"
                if customer_ready
                else "不能交付给客户验收，需先完成主管复核"
            ),
            "required_action": (
                "导出审计包并随验收材料归档"
                if customer_ready
                else "处理 review_queue 中的待复核记录"
            ),
        },
        "latest_timestamp": str(records[0].get("timestamp") or "") if records else "",
    }


def _customer_report(
    records: list[dict[str, Any]],
    *,
    source_health: dict[str, dict[str, Any]],
    review_integrity: dict[str, Any],
) -> dict[str, Any]:
    review_count = sum(1 for item in records if item.get("requires_review"))
    critical_count = sum(1 for item in records if item.get("severity") == "critical")
    high_count = sum(1 for item in records if item.get("severity") == "high")
    evidence_linked = sum(
        1 for item in records if item.get("evidence_status", {}).get("status") == "linked"
    )
    source_errors = [
        name
        for name, health in source_health.items()
        if int(health.get("invalid_record_count") or 0) > 0
        or health.get("readable") is False
    ]
    ready = review_count == 0 and not source_errors and review_integrity.get("valid") is not False
    acceptance_checklist = _customer_acceptance_checklist(
        records,
        review_count=review_count,
        source_errors=source_errors,
        review_integrity=review_integrity,
    )
    return {
        "title": "AskMe 产品审计报告",
        "status": "ready" if ready else "needs_action",
        "status_label": "可交付" if ready else "待主管复核",
        "customer_ready": ready,
        "summary_sentence": _customer_report_summary_sentence(
            records,
            review_count=review_count,
            source_errors=source_errors,
        ),
        "handoff_brief": _customer_handoff_brief(
            records,
            ready=ready,
            review_count=review_count,
            source_errors=source_errors,
        ),
        "acceptance_summary": _customer_acceptance_summary(acceptance_checklist),
        "acceptance_checklist": acceptance_checklist,
        "sections": {
            "review": {
                "pending_count": review_count,
                "critical_count": critical_count,
                "high_count": high_count,
            },
            "evidence": {
                "records_with_evidence": evidence_linked,
                "records_without_evidence": max(0, len(records) - evidence_linked),
            },
            "integrity": {
                "review_log_valid": review_integrity.get("valid") is not False,
                "hash_chained_records": sum(
                    1 for item in records if item.get("integrity", {}).get("hash_chain")
                ),
                "signed_records": sum(
                    1 for item in records if item.get("integrity", {}).get("signed")
                ),
            },
            "source_health": {
                "error_sources": source_errors,
                "source_labels": {name: _audit_source_label(name) for name in source_health},
                "configured_sources": [
                    name for name, health in source_health.items() if health.get("configured")
                ],
            },
        },
    }


def _delivery_dossier(
    records: list[dict[str, Any]],
    *,
    customer_report: dict[str, Any],
    audit_readiness: dict[str, Any],
    review_integrity: dict[str, Any],
) -> dict[str, Any]:
    ready = (
        bool(customer_report.get("customer_ready"))
        and str(audit_readiness.get("status") or "") == "ready"
        and review_integrity.get("valid") is not False
    )
    source_counts = Counter(str(item.get("source") or "unknown") for item in records)
    high_or_critical = sum(1 for item in records if item.get("severity") in {"critical", "high"})
    evidence_linked = sum(
        1 for item in records if item.get("evidence_status", {}).get("status") == "linked"
    )
    blocker_labels = [
        str(item)
        for item in audit_readiness.get("blocker_labels", [])
        if str(item or "").strip()
    ]
    warning_labels = [
        str(item)
        for item in audit_readiness.get("warning_labels", [])
        if str(item or "").strip()
    ]
    return {
        "title": "客户交付审计档案",
        "decision": "ready" if ready else "blocked",
        "decision_label": "可进入客户验收" if ready else "验收前存在阻断项",
        "customer_claim": (
            "已审计的操作、证据和复核记录可用于客户验收、事件复盘和责任追溯。"
            if ready
            else "当前审计范围仍有阻断项，暂不能作为客户验收证据。"
        ),
        "allowed_uses": (
            [
                "客户验收材料",
                "试点复盘材料",
                "事件闭环材料",
                "责任追溯材料",
            ]
            if ready
            else ["内部复核材料", "问题诊断材料", "证据缺口清单"]
        ),
        "blocked_uses": [
            "无人值守生产上线声明",
            "替代现场验收结果",
            "替代安全主管或客户负责人签字",
        ],
        "handoff_owner": "交付负责人" if ready else _dossier_blocking_owner(blocker_labels),
        "must_fix": blocker_labels,
        "watch_items": warning_labels,
        "record_scope": {
            "record_count": len(records),
            "source_counts": dict(sorted(source_counts.items())),
            "source_labels": {
                source: _customer_label(source)
                for source in sorted(source_counts)
            },
            "high_or_critical_count": high_or_critical,
            "evidence_linked_count": evidence_linked,
            "review_required_count": int(audit_readiness.get("review_required_count") or 0),
        },
        "acceptance_gate": customer_report.get("acceptance_summary") or {},
        "next_step": (
            "生成审计包，并随客户验收材料归档。"
            if ready
            else "先处理阻断项，完成主管复核后重新生成审计包。"
        ),
    }


def _dossier_blocking_owner(blocker_labels: list[str]) -> str:
    joined = " ".join(blocker_labels)
    if "复核" in joined:
        return "现场主管"
    if "完整性" in joined or "格式" in joined:
        return "研发或交付工程师"
    return "交付负责人"


def _customer_report_summary_sentence(
    records: list[dict[str, Any]],
    *,
    review_count: int,
    source_errors: list[str],
) -> str:
    if review_count:
        return f"当前有 {review_count} 条记录需要主管复核，暂不能作为客户验收审计包。"
    if source_errors:
        return "审计源存在读取或格式问题，需修复后再导出客户验收材料。"
    return f"当前 {len(records)} 条审计记录可用于客户验收和问题追溯。"


def _customer_handoff_brief(
    records: list[dict[str, Any]],
    *,
    ready: bool,
    review_count: int,
    source_errors: list[str],
) -> dict[str, Any]:
    if ready:
        return {
            "claim": "本审计包可作为客户验收、问题追溯和内部复盘材料。",
            "customer_message": f"本次范围内共整理 {len(records)} 条记录，暂无待复核阻断项。",
            "delivery_owner": "交付负责人",
            "next_step": "生成审计包并随验收材料归档。",
        }
    if review_count:
        return {
            "claim": "本审计包尚不能交付客户验收。",
            "customer_message": f"仍有 {review_count} 条记录需要主管确认处理依据。",
            "delivery_owner": "现场主管",
            "next_step": "先完成待复核记录，再重新生成审计包。",
        }
    return {
        "claim": "本审计包尚不能交付客户验收。",
        "customer_message": "审计源存在读取或格式问题，可能遗漏关键记录。",
        "delivery_owner": "研发/交付工程师",
        "next_step": f"修复审计源：{', '.join(source_errors) or 'unknown'}。",
    }


def _customer_acceptance_checklist(
    records: list[dict[str, Any]],
    *,
    review_count: int,
    source_errors: list[str],
    review_integrity: dict[str, Any],
) -> list[dict[str, Any]]:
    evidence_count = sum(
        1 for item in records if item.get("evidence_status", {}).get("status") == "linked"
    )
    high_or_critical = sum(1 for item in records if item.get("severity") in {"critical", "high"})
    return [
        {
            "id": "supervisor_review",
            "label": "主管复核",
            "status": "blocked" if review_count else "passed",
            "required": True,
            "detail": (
                f"仍有 {review_count} 条记录需要复核"
                if review_count
                else "无待复核记录"
            ),
            "next_step": "处理待复核队列" if review_count else "无需处理",
        },
        {
            "id": "source_health",
            "label": "审计源完整性",
            "status": "blocked" if source_errors else "passed",
            "required": True,
            "detail": (
                f"异常来源：{', '.join(source_errors)}"
                if source_errors
                else "已配置审计源可读取"
            ),
            "next_step": "修复异常审计源" if source_errors else "无需处理",
        },
        {
            "id": "review_log_integrity",
            "label": "复核日志防篡改",
            "status": "blocked" if review_integrity.get("valid") is False else "passed",
            "required": True,
            "detail": (
                "复核日志哈希链异常"
                if review_integrity.get("valid") is False
                else "复核日志哈希链有效"
            ),
            "next_step": (
                "暂停交付并核查复核日志"
                if review_integrity.get("valid") is False
                else "无需处理"
            ),
        },
        {
            "id": "evidence_links",
            "label": "证据附件",
            "status": "passed" if evidence_count else "warning",
            "required": False,
            "detail": f"{evidence_count} 条记录已关联证据",
            "next_step": "高风险记录建议补齐图片、视频或报告证据" if not evidence_count else "随审计包导出",
        },
        {
            "id": "risk_visibility",
            "label": "高风险可见性",
            "status": "warning" if high_or_critical else "passed",
            "required": False,
            "detail": f"{high_or_critical} 条高风险或关键记录",
            "next_step": "向客户说明处理闭环" if high_or_critical else "无需处理",
        },
    ]


def _customer_acceptance_summary(checklist: list[dict[str, Any]]) -> dict[str, Any]:
    required_blockers = [
        item
        for item in checklist
        if item.get("required") and str(item.get("status") or "") == "blocked"
    ]
    warnings = [item for item in checklist if str(item.get("status") or "") == "warning"]
    passed = [item for item in checklist if str(item.get("status") or "") == "passed"]
    ready = not required_blockers
    return {
        "status": "ready" if ready else "blocked",
        "status_label": "可进入客户验收" if ready else "需先处理阻断项",
        "customer_message": (
            "审计材料已满足必要验收条件，可随客户验收或复盘材料归档。"
            if ready
            else f"仍有 {len(required_blockers)} 个必要验收项未通过，暂不能提交客户验收。"
        ),
        "required_blocker_count": len(required_blockers),
        "warning_count": len(warnings),
        "passed_count": len(passed),
        "total_count": len(checklist),
        "blocked_item_ids": [str(item.get("id") or "") for item in required_blockers],
        "warning_item_ids": [str(item.get("id") or "") for item in warnings],
        "next_step": (
            "导出审计包并随验收材料归档。"
            if ready
            else "优先处理 supervisor_review、source_health 或 review_log_integrity 阻断项。"
        ),
    }


def _audit_readiness(
    records: list[dict[str, Any]],
    *,
    source_health: dict[str, dict[str, Any]],
    review_integrity: dict[str, Any],
) -> dict[str, Any]:
    review_count = sum(1 for item in records if item.get("requires_review"))
    critical_count = sum(1 for item in records if item.get("severity") == "critical")
    invalid_sources = [
        {
            "source": name,
            "invalid_record_count": int(health.get("invalid_record_count") or 0),
            "path": str(health.get("path") or ""),
        }
        for name, health in source_health.items()
        if int(health.get("invalid_record_count") or 0) > 0
    ]
    missing_sources = [
        {
            "source": name,
            "path": str(health.get("path") or ""),
        }
        for name, health in source_health.items()
        if health.get("configured") and not health.get("exists") and name != "audit_reviews"
    ]
    unsigned_high_risk = sum(
        1
        for item in records
        if item.get("severity") in {"critical", "high"}
        and not item.get("integrity", {}).get("signed")
    )
    blockers: list[dict[str, Any]] = []
    if review_integrity.get("valid") is False:
        blockers.append({"reason": "review_log_integrity_failed"})
    if invalid_sources:
        blockers.append({"reason": "audit_source_has_invalid_records", "sources": invalid_sources})
    if review_count:
        blockers.append({"reason": "supervisor_review_required", "count": review_count})
    warnings: list[dict[str, Any]] = []
    if missing_sources:
        warnings.append({"reason": "configured_audit_source_missing", "sources": missing_sources})
    if unsigned_high_risk:
        warnings.append({"reason": "high_risk_records_not_signed", "count": unsigned_high_risk})
    if critical_count:
        warnings.append({"reason": "critical_records_present", "count": critical_count})
    return {
        "status": "blocked" if blockers else "ready",
        "status_label": "阻断交付" if blockers else "可生成验收审计包",
        "blockers": blockers,
        "blocker_labels": [_audit_reason_label(str(item.get("reason") or "")) for item in blockers],
        "warnings": warnings,
        "warning_labels": [_audit_reason_label(str(item.get("reason") or "")) for item in warnings],
        "next_actions": _audit_next_actions(blockers=blockers, warnings=warnings),
        "review_required_count": review_count,
        "critical_count": critical_count,
        "unsigned_high_risk_count": unsigned_high_risk,
    }


def _audit_next_actions(
    *,
    blockers: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> list[str]:
    actions: list[str] = []
    reasons = {str(item.get("reason") or "") for item in [*blockers, *warnings]}
    if "review_log_integrity_failed" in reasons:
        actions.append("暂停交付，先核查审计复核日志的哈希链。")
    if "audit_source_has_invalid_records" in reasons:
        actions.append("修复或归档格式异常的 JSONL 审计记录后再导出。")
    if "supervisor_review_required" in reasons:
        actions.append("主管需要处理待复核队列并给出复核结论。")
    if "configured_audit_source_missing" in reasons:
        actions.append("检查当前节点配置的审计源路径是否存在。")
    if "high_risk_records_not_signed" in reasons:
        actions.append("为高风险现场和运行审计记录启用签名。")
    if not actions:
        actions.append("审计包已可导出并随客户验收材料归档。")
    return actions


def _audit_reason_label(reason: str) -> str:
    return {
        "review_log_integrity_failed": "复核日志完整性异常",
        "audit_source_has_invalid_records": "审计源存在格式异常",
        "supervisor_review_required": "存在待主管复核记录",
        "configured_audit_source_missing": "配置的审计源不存在",
        "high_risk_records_not_signed": "高风险记录未签名",
        "critical_records_present": "存在紧急级记录",
    }.get(reason, reason or "未知问题")


def _audit_source_label(source: str) -> str:
    return {
        "skill_audit": "能力与技能日志",
        "field_action_audit": "现场处置动作日志",
        "field_event_archive": "现场事件归档",
        "runtime_audit": "机器人运行交接日志",
        "audit_reviews": "主管复核日志",
    }.get(source, source)


def _review_queue(records: list[dict[str, Any]], *, limit: int = 25) -> list[dict[str, Any]]:
    return [item for item in records if item.get("requires_review")][:limit]


def _review_decisions(path: Path | None, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return AuditReviewService(config, path=path).latest()


def _apply_review_decision(
    record: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    decision = decisions.get(str(record.get("record_id") or ""))
    if not decision:
        record["review_status"] = "pending" if record.get("requires_review") else "not_required"
        _refresh_review_dependent_fields(record)
        return record
    record["review_decision"] = decision
    if decision_clears_review(str(decision.get("decision") or "")):
        record["requires_review"] = False
        record["review_status"] = "cleared"
    else:
        record["requires_review"] = True
        record["review_status"] = str(decision.get("decision") or "pending")
    _refresh_review_dependent_fields(record)
    return record


def _refresh_review_dependent_fields(record: dict[str, Any]) -> None:
    """Keep customer-facing copy aligned after review decisions are applied."""
    severity = str(record.get("severity") or "low")
    requires_review = bool(record.get("requires_review"))
    source = str(record.get("source") or "")
    record["governance"] = _governance_payload(
        severity=severity,
        requires_review=requires_review,
    )
    record["review_reason"] = _review_reason(
        record,
        severity=severity,
        requires_review=requires_review,
    )
    record["recommended_action"] = _recommended_action(
        record,
        severity=severity,
        requires_review=requires_review,
    )
    evidence_refs = record.get("evidence_refs") if isinstance(record.get("evidence_refs"), list) else []
    record["acceptance"] = _acceptance_payload(
        severity=severity,
        requires_review=requires_review,
        evidence_refs=evidence_refs,
    )
    customer_copy = record.get("customer_copy")
    if isinstance(customer_copy, dict):
        customer_copy["next_step"] = _customer_next_step(
            record,
            severity=severity,
            requires_review=requires_review,
        )
        customer_copy["review_owner"] = _review_owner(
            source,
            severity=severity,
            requires_review=requires_review,
        )


def _record_id(record: dict[str, Any]) -> str:
    raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
    source = str(record.get("source") or "audit")
    for key in ("record_id", "audit_id", "sequence", "event_id", "run_id", "handoff_id", "id"):
        value = raw.get(key)
        if value not in {None, ""}:
            return f"{source}:{value}"
    audit = raw.get("audit") if isinstance(raw.get("audit"), dict) else {}
    for key in ("record_id", "audit_id", "id"):
        value = audit.get(key)
        if value not in {None, ""}:
            return f"{source}:{value}"
    seed = {
        "source": source,
        "category": record.get("category") or "",
        "action": record.get("action") or "",
        "outcome": record.get("outcome") or "",
        "operator_id": record.get("operator_id") or "",
        "subject": record.get("subject") or "",
        "timestamp": record.get("timestamp") or "",
    }
    digest = hashlib.sha256(
        json.dumps(seed, ensure_ascii=False, sort_keys=True).encode("utf-8"),
    ).hexdigest()[:16]
    return f"{source}:{digest}"


def _severity_for(record: dict[str, Any]) -> str:
    text = " ".join(
        str(record.get(key) or "").lower()
        for key in ("source", "category", "action", "outcome", "reason", "message", "subject")
    )
    outcome = str(record.get("outcome") or "").lower()
    if any(term in text for term in ("fire", "smoke", "fall_unrecoverable", "motor_fault", "emergency")):
        return "critical"
    if outcome in {"denied", "blocked", "failed", "rejected", "invalid", "unauthorized", "error"}:
        return "high"
    if any(term in text for term in ("not_authorized", "tamper", "mismatch", "forged", "untrusted")):
        return "high"
    if outcome in {"pending", "requested", "needs_review", "warning"}:
        return "medium"
    if any(term in text for term in ("approval", "missing", "retry", "timeout")):
        return "medium"
    return "low"


def _requires_review(record: dict[str, Any], *, severity: str) -> bool:
    if severity in {"critical", "high"}:
        return True
    outcome = str(record.get("outcome") or "").lower()
    reason = str(record.get("reason") or "").lower()
    return outcome in {"pending", "needs_review"} or "approval_required" in reason


def _customer_label(source: str) -> str:
    return {
        "skill": "能力与技能审计",
        "field": "现场事件处置审计",
        "runtime": "机器人任务运行审计",
    }.get(source, "产品审计")


def _display_title(record: dict[str, Any], *, source: str, subject: str) -> str:
    label = _customer_label(source)
    action = _action_label(str(record.get("action") or record.get("category") or "activity"))
    outcome = _outcome_label(str(record.get("outcome") or "recorded"))
    prefix = f"{label}："
    target = f"{subject} " if subject else ""
    return f"{prefix}{target}执行{action}，结果为{outcome}"


def _customer_impact(source: str) -> str:
    return {
        "skill": "capability_governance",
        "field": "field_response_and_evidence",
        "runtime": "robot_task_operation",
    }.get(source, "product_governance")


def _customer_copy(
    record: dict[str, Any],
    *,
    source: str,
    subject: str,
    severity: str,
    requires_review: bool,
) -> dict[str, str]:
    actor = str(record.get("operator_id") or "系统")
    actor_label = _actor_label(_actor_type(record))
    action = _action_label(str(record.get("action") or record.get("category") or "操作"))
    outcome = _outcome_label(str(record.get("outcome") or "recorded"))
    resource = subject or "当前对象"
    return {
        "title": _customer_copy_title(source),
        "what_happened": _customer_what_happened(
            source=source,
            actor_label=actor_label,
            actor=actor,
            action=action,
            outcome=outcome,
            resource=resource,
        ),
        "why_it_matters": _customer_why_it_matters(source, severity=severity),
        "next_step": _customer_next_step(
            record,
            severity=severity,
            requires_review=requires_review,
        ),
        "review_owner": _review_owner(source, severity=severity, requires_review=requires_review),
    }


def _customer_copy_title(source: str) -> str:
    return {
        "skill": "机器人能力变更记录",
        "field": "现场事件处置记录",
        "runtime": "机器人任务运行记录",
    }.get(source, "产品审计记录")


def _customer_what_happened(
    *,
    source: str,
    actor_label: str,
    actor: str,
    action: str,
    outcome: str,
    resource: str,
) -> str:
    actor_text = _actor_display(actor_label, actor)
    if source == "field":
        return f"{actor_text}对现场事件 {resource} 执行{action}，结果为{outcome}。"
    if source == "runtime":
        return f"{actor_text}对机器人任务 {resource} 执行{action}，结果为{outcome}。"
    if source == "skill":
        return f"{actor_text}对机器人能力 {resource} 执行{action}，结果为{outcome}。"
    return f"{actor_text}对 {resource} 执行{action}，结果为{outcome}。"


def _actor_display(actor_label: str, actor: str) -> str:
    clean_actor = str(actor or "").strip()
    if not clean_actor or clean_actor == actor_label:
        return f"{actor_label} "
    if actor_label == "系统" and clean_actor.lower() in {"system", "系统"}:
        return "系统 "
    return f"{actor_label} {clean_actor} "


def _actor_label(actor_type: str) -> str:
    return {
        "system": "系统",
        "supervisor": "主管",
        "visitor": "访客",
        "security_operator": "安保人员",
        "operator": "操作员",
    }.get(actor_type, "操作员")


def _customer_why_it_matters(source: str, *, severity: str) -> str:
    if severity == "critical":
        return "这条记录关联安全或紧急事件，必须进入复核和留痕。"
    return {
        "skill": "这条记录影响机器人可用能力、启停审批和后续责任追踪。",
        "field": "这条记录影响现场事件处理、证据闭环和客户验收。",
        "runtime": "这条记录影响机器人任务执行、暂停接管和运行安全。",
    }.get(source, "这条记录影响产品治理和客户验收。")


def _customer_next_step(
    record: dict[str, Any],
    *,
    severity: str,
    requires_review: bool,
) -> str:
    if requires_review:
        if severity == "critical":
            return "立即由主管复核，并确认现场证据、通知记录和关闭原因。"
        if severity == "high":
            return "主管需要核对操作权限、现场证据和处理说明。"
        return "主管需要确认记录是否有效，并给出复核结论。"
    if _evidence_refs(record):
        return "无需复核，保留证据链接用于客户查看。"
    return "无需复核，保留在审计时间线中备查。"


def _review_owner(source: str, *, severity: str, requires_review: bool) -> str:
    if not requires_review:
        return "无需复核"
    if severity == "critical":
        return "安全主管"
    return {
        "skill": "产品负责人",
        "field": "现场主管",
        "runtime": "运行主管",
    }.get(source, "主管")


def _action_label(action: str) -> str:
    normalized = str(action or "").strip().lower()
    return {
        "acknowledge": "确认收到",
        "approve": "审批通过",
        "request_close": "申请关闭",
        "close": "关闭",
        "notify": "通知",
        "resend_notification": "重发通知",
        "runtime_delivery": "记录运行回调",
        "handoff": "提交任务交接",
        "execute": "执行",
        "upsert_package": "更新能力包",
        "reject": "驳回",
        "profile_written": "写入配置",
        "agent_task": "执行智能体任务",
        "navigate": "导航",
        "patrol": "巡检",
        "dog_control": "机器狗控制",
        "pause": "暂停",
        "resume": "继续",
        "cancel": "取消",
        "skill_enable": "启用能力",
        "skill_disable": "停用能力",
        "activity": "记录活动",
        "操作": "操作",
    }.get(normalized, action or "操作")


def _outcome_label(outcome: str) -> str:
    normalized = str(outcome or "").strip().lower()
    return {
        "accepted": "已接受",
        "approved": "已通过",
        "acknowledged": "已确认",
        "completed": "已完成",
        "closed": "已关闭",
        "created": "已创建",
        "draft_created": "草稿已创建",
        "package_updated": "能力包已更新",
        "profile_written": "配置已写入",
        "recorded": "已记录",
        "sent": "已发送",
        "succeeded": "成功",
        "resolved": "已处理",
        "waived": "已豁免",
        "denied": "已拒绝",
        "blocked": "已阻止",
        "failed": "失败",
        "rejected": "已驳回",
        "invalid": "无效",
        "unauthorized": "未授权",
        "error": "异常",
        "pending": "待处理",
        "requested": "已申请",
        "needs_review": "待复核",
        "warning": "需关注",
    }.get(normalized, outcome or "已记录")


def _severity_label(severity: str) -> str:
    return {
        "critical": "紧急",
        "high": "高风险",
        "medium": "中风险",
        "low": "低风险",
    }.get(str(severity or "").lower(), "未分级")


def _acceptance_payload(
    *,
    severity: str,
    requires_review: bool,
    evidence_refs: list[dict[str, str]],
) -> dict[str, Any]:
    if requires_review:
        return {
            "status": "blocked",
            "label": "待复核，不可交付",
            "can_export": False,
            "reason": "该记录仍需主管复核，不能进入客户验收审计包。",
            "next_step": (
                "补齐现场证据并由主管确认"
                if severity in {"critical", "high"}
                else "由主管确认记录有效性"
            ),
        }
    return {
        "status": "ready",
        "label": "可归档",
        "can_export": True,
        "reason": "该记录无需复核，可保留在审计时间线中。",
        "next_step": "随审计包导出" if evidence_refs else "保留记录用于追溯",
    }


def _governance_payload(*, severity: str, requires_review: bool) -> dict[str, Any]:
    return {
        "handoff_status": "blocked_until_reviewed" if requires_review else "ready",
        "customer_visible": True,
        "exportable": not requires_review,
        "requires_supervisor": requires_review or severity in {"critical", "high"},
    }


def _review_reason(
    record: dict[str, Any],
    *,
    severity: str,
    requires_review: bool,
) -> str:
    if not requires_review:
        return ""
    outcome = str(record.get("outcome") or "").lower()
    reason = str(record.get("reason") or "").lower()
    if severity == "critical":
        return "critical_safety_or_emergency_record"
    if severity == "high":
        return "failed_or_high_risk_action"
    if outcome in {"pending", "needs_review"}:
        return "pending_decision"
    if "approval_required" in reason:
        return "approval_required"
    return "supervisor_review_required"


def _recommended_action(
    record: dict[str, Any],
    *,
    severity: str,
    requires_review: bool,
) -> str:
    if requires_review:
        if severity == "critical":
            return "升级给安全主管，并在关闭前补齐现场证据。"
        if severity == "high":
            return "主管需要核对操作权限、现场证据和处理说明。"
        return "复核该记录，并选择通过、已处理、豁免、升级或驳回。"
    if not _evidence_refs(record):
        return "无需复核，保留在审计时间线中备查。"
    return "无需复核，证据已关联，可供客户验收查看。"


def _resource_type(source: str) -> str:
    return {
        "skill": "skill",
        "field": "field_event",
        "runtime": "task_run",
    }.get(source, "audit_record")


def _actor_type(record: dict[str, Any]) -> str:
    operator_id = str(record.get("operator_id") or "").lower()
    if not operator_id:
        return "system"
    if "supervisor" in operator_id or "admin" in operator_id:
        return "supervisor"
    if "visitor" in operator_id:
        return "visitor"
    if "guard" in operator_id or "security" in operator_id:
        return "security_operator"
    return "operator"


def _integrity_summary(record: dict[str, Any]) -> dict[str, Any]:
    raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
    return {
        "hash_chain": bool(raw.get("record_hash")),
        "signed": bool(raw.get("record_signature") or raw.get("signature")),
        "hash_alg": str(raw.get("hash_alg") or ""),
        "signature_alg": str(raw.get("signature_alg") or raw.get("runtime_signature_alg") or ""),
        "sequence": raw.get("sequence"),
    }


def _evidence_refs(record: dict[str, Any]) -> list[dict[str, str]]:
    raw = record.get("raw") if isinstance(record.get("raw"), dict) else {}
    refs: list[dict[str, str]] = []
    for container in _evidence_containers(raw):
        for key in ("evidence_refs", "evidence_media", "evidence", "artifacts"):
            value = container.get(key) if isinstance(container, dict) else None
            refs.extend(_normalize_evidence_value(value, source_key=key))
        refs.extend(_direct_evidence_refs(container))
    return _dedupe_evidence_refs(refs)[:8]


def _evidence_containers(raw: dict[str, Any]) -> list[dict[str, Any]]:
    containers = [raw]
    for key in ("event", "payload", "delivery", "runtime_delivery"):
        value = raw.get(key)
        if isinstance(value, dict):
            containers.append(value)
    return containers


def _normalize_evidence_value(value: Any, *, source_key: str) -> list[dict[str, str]]:
    if not value:
        return []
    if isinstance(value, list):
        refs: list[dict[str, str]] = []
        for item in value:
            refs.extend(_normalize_evidence_value(item, source_key=source_key))
        return refs
    if isinstance(value, dict):
        path = str(value.get("path") or value.get("url") or value.get("uri") or "").strip()
        label = str(value.get("label") or value.get("type") or source_key).strip()
        if not path and not label:
            return []
        return [{"label": label or source_key, "path": path}]
    text = str(value).strip()
    return [{"label": source_key, "path": text}] if text else []


def _direct_evidence_refs(container: dict[str, Any]) -> list[dict[str, str]]:
    if not isinstance(container, dict):
        return []
    refs: list[dict[str, str]] = []
    for key, label in (
        ("image_path", "photo"),
        ("photo_path", "photo"),
        ("snapshot_path", "snapshot"),
        ("video_path", "video"),
        ("audio_path", "audio"),
        ("report_path", "report"),
        ("map_path", "map"),
        ("preview_url", "preview"),
    ):
        value = str(container.get(key) or "").strip()
        if value:
            refs.append({"label": label, "path": value})
    return refs


def _dedupe_evidence_refs(refs: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for ref in refs:
        path = str(ref.get("path") or "").strip()
        label = str(ref.get("label") or ref.get("type") or "evidence").strip()
        if not path:
            continue
        key = (label, path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"label": label or "evidence", "path": path})
    return deduped


def _evidence_status(refs: list[dict[str, str]]) -> dict[str, Any]:
    labels = Counter(str(ref.get("label") or "evidence") for ref in refs)
    return {
        "status": "linked" if refs else "not_linked",
        "ref_count": len(refs),
        "labels": dict(sorted(labels.items())),
    }


def _source_health_item(path: Path | None) -> dict[str, Any]:
    configured = path is not None
    exists = bool(path and path.is_file())
    health = _jsonl_health(path) if exists else {}
    return {
        "configured": configured,
        "exists": exists,
        "path": str(path or ""),
        "record_count": health.get("line_count", 0),
        "valid_record_count": health.get("valid_record_count", 0),
        "invalid_record_count": health.get("invalid_record_count", 0),
        "readable": health.get("readable", bool(not exists)),
        "error": health.get("error", ""),
    }


def _jsonl_record_count(path: Path | None) -> int:
    if path is None or not path.is_file():
        return 0
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def _jsonl_health(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {
            "readable": bool(path is None),
            "line_count": 0,
            "valid_record_count": 0,
            "invalid_record_count": 0,
            "error": "" if path is None else "file_not_found",
        }
    line_count = 0
    valid_count = 0
    invalid_count = 0
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                line_count += 1
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    invalid_count += 1
                    continue
                if isinstance(item, dict):
                    valid_count += 1
                else:
                    invalid_count += 1
    except OSError as exc:
        return {
            "readable": False,
            "line_count": 0,
            "valid_record_count": 0,
            "invalid_record_count": 0,
            "error": str(exc),
        }
    return {
        "readable": True,
        "line_count": line_count,
        "valid_record_count": valid_count,
        "invalid_record_count": invalid_count,
        "error": "",
    }


def _time_boundary(value: Any) -> dict[str, Any]:
    text = str(value or "").strip()
    if not text:
        return {"raw": "", "sort_at": None, "valid": True}
    sort_at = _sort_time_or_none(value)
    if sort_at is None:
        return {
            "raw": text,
            "sort_at": None,
            "valid": False,
            "reason": "invalid_time_boundary",
        }
    return {
        "raw": text,
        "sort_at": sort_at,
        "valid": True,
        "iso": datetime.fromtimestamp(sort_at, tz=UTC).isoformat(),
    }


def _time_window_payload(since: dict[str, Any], until: dict[str, Any]) -> dict[str, Any]:
    errors = []
    if since.get("valid") is False:
        errors.append({"field": "since", "reason": since.get("reason")})
    if until.get("valid") is False:
        errors.append({"field": "until", "reason": until.get("reason")})
    return {
        "since": since,
        "until": until,
        "valid": not errors,
        "errors": errors,
    }


def _sort_time(value: Any) -> float:
    parsed = _sort_time_or_none(value)
    return parsed if parsed is not None else 0.0


def _sort_time_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _iso_from_any(value: Any) -> str:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC).isoformat()
    text = str(value or "").strip()
    return text
