"""Backend-independent knowledge catalog for curated RAG facts."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from askme.config import project_root

_UTC = timezone(timedelta(0))
_ACTIVE_STATUSES = {"published", "approved", "active"}
_BLOCKED_STATUSES = {"draft", "pending", "rejected", "deleted", "expired", "conflicted"}
_PROMPT_AFFECTING_FIELDS = {
    "text",
    "memory_text",
    "approval_status",
    "expires_at",
    "entity_key",
    "fact_key",
    "value",
    "content_hash",
}


class KnowledgeCatalog:
    """Durable source of truth for curated knowledge lifecycle metadata."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        data_dir: str | Path | None = None,
        path: str | Path | None = None,
    ) -> None:
        cfg = config or {}
        if path is not None:
            self._path = Path(path)
        else:
            if data_dir is not None:
                resolved = Path(data_dir)
            else:
                raw = cfg.get("app", {}).get("data_dir", "data")
                resolved = Path(raw)
                if not resolved.is_absolute():
                    resolved = project_root() / resolved
            self._path = resolved / "memory" / "catalog" / "records.json"
        self._records: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._load()

    def health(self) -> dict[str, Any]:
        with self._lock:
            records = [self._public_record(record) for record in self._records.values()]
            total = len(records)
            by_state: dict[str, int] = {}
            for record in records:
                state = str(record.get("lifecycle_state") or "unknown")
                by_state[state] = by_state.get(state, 0) + 1
        return {
            "path": str(self._path),
            "total": total,
            "prompt_eligible": sum(1 for record in records if record.get("prompt_eligible")),
            "needs_review": sum(
                by_state.get(state, 0)
                for state in ("draft", "pending_review", "rejected", "unapproved")
            ),
            "needs_reindex": sum(1 for record in records if record.get("needs_reindex")),
            "expiring_soon": by_state.get("expiring_soon", 0),
            "expired": by_state.get("expired", 0),
            "conflicted": by_state.get("conflict", 0),
            "deleted": by_state.get("deleted", 0),
            "by_state": by_state,
        }

    def upsert_payloads(self, payloads: list[dict[str, Any]]) -> dict[str, Any]:
        changed_ids: set[str] = set()
        now = _now_iso()
        with self._lock:
            for payload in payloads:
                record = self._normalize_payload(payload, now=now)
                record_id = str(record.get("record_id") or "").strip()
                if not record_id:
                    continue
                previous = self._records.get(record_id, {})
                if previous.get("indexed_at"):
                    record["indexed_at"] = previous.get("indexed_at")
                if previous.get("indexed_evidence_version"):
                    record["indexed_evidence_version"] = previous.get("indexed_evidence_version")
                record["evidence_version"] = self._next_evidence_version(previous, record)
                record["metadata"]["evidence_version"] = record["evidence_version"]
                record["events"] = _record_events(previous, record)
                record["events"].append(_event("upsert", actor=record.get("owner") or "import"))
                self._records[record_id] = {**previous, **record}
                changed_ids.add(record_id)
            conflict_changed_ids = self._refresh_conflicts()
            changed_ids.update(conflict_changed_ids)
            self._save()
            records = [self._public_record(self._records[record_id]) for record_id in changed_ids]
        return {
            "records": records,
            "changed_record_ids": sorted(changed_ids),
            "indexed_candidates": [record for record in records if self.is_prompt_eligible(record)],
        }

    def list_records(self, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        safe_limit = max(1, min(int(limit), 500))
        safe_offset = max(0, int(offset))
        with self._lock:
            ordered = sorted(
                self._records.values(),
                key=lambda record: str(record.get("updated_at") or record.get("imported_at") or ""),
                reverse=True,
            )
            window = ordered[safe_offset : safe_offset + safe_limit]
            records = [self._public_record(record) for record in window]
            total = len(ordered)
        return {
            "backend": "catalog",
            "records": records,
            "total": total,
            "catalog": self.health(),
        }

    def records_for_rebuild(
        self,
        *,
        record_ids: list[str] | None = None,
        include_ineligible: bool = False,
        limit: int | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Return catalog records selected for a rebuild-index job."""
        wanted = {
            str(record_id or "").strip()
            for record_id in (record_ids or [])
            if str(record_id or "").strip()
        }
        safe_offset = max(0, int(offset))
        safe_limit = None if limit is None else max(1, int(limit))
        with self._lock:
            ordered = sorted(
                self._records.values(),
                key=lambda record: str(record.get("updated_at") or record.get("imported_at") or ""),
                reverse=True,
            )
            if wanted:
                ordered = [
                    record
                    for record in ordered
                    if str(record.get("record_id") or "").strip() in wanted
                ]
            public_records = [self._public_record(record) for record in ordered]
            selected = [
                record
                for record in public_records
                if include_ineligible or self.is_prompt_eligible(record)
            ]
            total_selected = len(selected)
            if safe_offset:
                selected = selected[safe_offset:]
            if safe_limit is not None:
                selected = selected[:safe_limit]
        return {
            "records": selected,
            "total": len(public_records),
            "eligible": total_selected,
            "skipped": max(0, len(public_records) - total_selected),
            "record_ids": [record["record_id"] for record in selected],
        }

    def update_metadata(self, record_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        target = str(record_id or "").strip()
        if not target:
            return {"updated": False, "error": "missing_record_id"}
        allowed = {
            "text",
            "memory_text",
            "approval_status",
            "category",
            "source",
            "owner",
            "updated_at",
            "expires_at",
            "entity_key",
            "fact_key",
            "value",
            "deleted_at",
            "deleted_reason",
            "restored_at",
            "conflict_set_id",
            "source_version",
            "evidence_version",
            "approved_by",
            "approved_at",
            "rejected_by",
            "rejected_at",
            "review_note",
            "updated_by",
        }
        clean_patch = {k: v for k, v in patch.items() if k in allowed}
        if not clean_patch:
            return {"updated": False, "error": "empty_patch"}
        clean_patch.setdefault("updated_at", _now_iso())
        with self._lock:
            record = self._records.get(target)
            if record is None:
                return {"updated": False, "error": "record_not_found", "record_id": target}
            if any(field in clean_patch for field in _PROMPT_AFFECTING_FIELDS):
                revisions = _record_revisions(record)
                revisions.append(_revision_from_record(record, actor=str(clean_patch.get("updated_by") or "")))
                record["revisions"] = revisions[-25:]
                clean_patch["evidence_version"] = _int_or_default(
                    record.get("evidence_version"),
                    1,
                ) + 1
            record.update(clean_patch)
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            record["metadata"] = {**metadata, **clean_patch}
            events = _record_events(record)
            actor = str(
                clean_patch.get("approved_by")
                or clean_patch.get("rejected_by")
                or clean_patch.get("updated_by")
                or ""
            )
            status = str(clean_patch.get("approval_status") or "").strip().lower()
            kind = f"status:{status}" if status else "metadata_update"
            events.append(_event(kind, actor=actor, note=str(clean_patch.get("review_note") or "")))
            record["events"] = events[-50:]
            changed_ids = self._refresh_conflicts()
            changed_ids.add(target)
            self._save()
            public = self._public_record(record)
            changed_records = [
                self._public_record(self._records[item])
                for item in changed_ids
                if item in self._records
            ]
        return {
            "updated": True,
            "record_id": target,
            "record": public,
            "changed_records": changed_records,
            "patch": clean_patch,
        }

    def update_metadata_many(
        self,
        updates: list[dict[str, Any]],
        *,
        default_patch: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Patch metadata for multiple records while preserving single-update semantics."""
        updated_records: list[dict[str, Any]] = []
        changed_records: dict[str, dict[str, Any]] = {}
        errors: list[dict[str, Any]] = []
        for index, item in enumerate(updates, start=1):
            if not isinstance(item, dict):
                errors.append({"index": index, "error": "invalid_update"})
                continue
            record_id = str(item.get("record_id") or "").strip()
            patch = dict(default_patch or {})
            item_patch = item.get("patch") if isinstance(item.get("patch"), dict) else {}
            patch.update(item_patch)
            result = self.update_metadata(record_id, patch)
            if result.get("updated"):
                record = result.get("record")
                if isinstance(record, dict):
                    updated_records.append(record)
                for changed in result.get("changed_records", []):
                    if isinstance(changed, dict):
                        changed_records[str(changed.get("record_id") or "")] = changed
            else:
                errors.append({
                    "index": index,
                    "record_id": record_id,
                    "error": result.get("error", "update_failed"),
                })
        return {
            "updated": len(updated_records),
            "failed": len(errors),
            "records": updated_records,
            "changed_records": list(changed_records.values()),
            "errors": errors,
        }

    def diff_record(self, record_id: str, revision_id: str | None = None) -> dict[str, Any]:
        target = str(record_id or "").strip()
        with self._lock:
            record = self._records.get(target)
            if record is None:
                return {"found": False, "error": "record_not_found", "record_id": target}
            revisions = _record_revisions(record)
            if not revisions:
                return {
                    "found": True,
                    "record_id": target,
                    "revision_id": "",
                    "changes": [],
                    "current": self._public_record(record),
                    "previous": {},
                }
            previous = revisions[-1]
            if revision_id:
                previous = next(
                    (item for item in revisions if item.get("revision_id") == revision_id),
                    previous,
                )
            current = self._public_record(record)
        changes = []
        for field in sorted(_PROMPT_AFFECTING_FIELDS | {"category", "source", "owner"}):
            before = previous.get("snapshot", {}).get(field, previous.get(field, ""))
            after = current.get(field, "")
            if str(before or "") != str(after or ""):
                changes.append({"field": field, "before": before, "after": after})
        return {
            "found": True,
            "record_id": target,
            "revision_id": previous.get("revision_id", ""),
            "changes": changes,
            "previous": previous,
            "current": current,
        }

    def rollback_record(
        self,
        record_id: str,
        revision_id: str | None = None,
        *,
        actor: str = "",
        note: str = "",
    ) -> dict[str, Any]:
        target = str(record_id or "").strip()
        with self._lock:
            record = self._records.get(target)
            if record is None:
                return {"updated": False, "error": "record_not_found", "record_id": target}
            revisions = _record_revisions(record)
            if not revisions:
                return {"updated": False, "error": "no_revisions", "record_id": target}
            selected = revisions[-1]
            if revision_id:
                selected = next(
                    (item for item in revisions if item.get("revision_id") == revision_id),
                    selected,
                )
            snapshot = selected.get("snapshot") if isinstance(selected.get("snapshot"), dict) else {}
            rollback_fields = _PROMPT_AFFECTING_FIELDS | {
                "category",
                "source",
                "owner",
                "approval_status",
                "source_version",
            }
            revisions.append(_revision_from_record(record, actor=actor, note="before_rollback"))
            record["revisions"] = revisions[-25:]
            for field in rollback_fields:
                if field in snapshot:
                    record[field] = snapshot[field]
            record["updated_at"] = _now_iso()
            record["updated_by"] = actor
            record["review_note"] = note or f"rollback:{selected.get('revision_id', '')}"
            record["evidence_version"] = _int_or_default(record.get("evidence_version"), 1) + 1
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            for field in rollback_fields:
                if field in record:
                    metadata[field] = record[field]
            metadata["updated_at"] = record["updated_at"]
            metadata["updated_by"] = actor
            metadata["review_note"] = record["review_note"]
            metadata["evidence_version"] = record["evidence_version"]
            record["metadata"] = metadata
            events = _record_events(record)
            events.append(_event("rollback", actor=actor, note=record["review_note"]))
            record["events"] = events[-50:]
            changed_ids = self._refresh_conflicts()
            changed_ids.add(target)
            self._save()
            changed_records = [
                self._public_record(self._records[item])
                for item in changed_ids
                if item in self._records
            ]
            public = self._public_record(record)
        return {
            "updated": True,
            "action": "rollback",
            "record_id": target,
            "revision_id": selected.get("revision_id", ""),
            "record": public,
            "changed_records": changed_records,
        }

    def mark_indexed(self, record_id: str) -> None:
        target = str(record_id or "").strip()
        if not target:
            return
        with self._lock:
            record = self._records.get(target)
            if record is None:
                return
            record["indexed_at"] = _now_iso()
            record["indexed_evidence_version"] = _int_or_default(
                record.get("evidence_version"),
                1,
            )
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            record["metadata"] = {
                **metadata,
                "indexed_at": record["indexed_at"],
                "indexed_evidence_version": record["indexed_evidence_version"],
            }
            events = _record_events(record)
            events.append(_event("indexed", actor="rag_backend"))
            record["events"] = events[-50:]
            self._save()

    def evidence_drop_reason(self, metadata: dict[str, Any]) -> str:
        """Return why backend evidence must be rejected against catalog truth."""
        record_id = str(metadata.get("record_id") or "").strip()
        if not record_id:
            return ""
        with self._lock:
            record = self._records.get(record_id)
            if record is None:
                return "catalog:missing_record"
            public = self._public_record(record)
        if not self.is_prompt_eligible(public):
            status = str(public.get("approval_status") or "").strip().lower()
            if public.get("conflict_set_id"):
                return f"catalog_conflict:{public['conflict_set_id']}"
            if status:
                return f"catalog_status:{status}"
            return "catalog:not_prompt_eligible"
        expected_version = str(public.get("evidence_version") or "").strip()
        actual_version = str(metadata.get("evidence_version") or "").strip()
        if expected_version and actual_version and expected_version != actual_version:
            return f"catalog_evidence_version:{actual_version}->{expected_version}"
        return ""

    @staticmethod
    def is_prompt_eligible(record: dict[str, Any]) -> bool:
        status = str(record.get("approval_status") or "").strip().lower()
        if status in _BLOCKED_STATUSES:
            return False
        if status and status not in _ACTIVE_STATUSES:
            return False
        if record.get("conflict_set_id"):
            return False
        expires_at = _parse_time(record.get("expires_at"))
        return not (expires_at is not None and expires_at <= datetime.now(_UTC))

    def _refresh_conflicts(self) -> set[str]:
        changed_ids: set[str] = set()
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for record in self._records.values():
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            status = str(record.get("approval_status") or metadata.get("approval_status") or "")
            if status.lower() in {"deleted", "rejected", "draft", "pending"}:
                continue
            entity_key = _norm(record.get("entity_key") or metadata.get("entity_key"))
            fact_key = _norm(record.get("fact_key") or metadata.get("fact_key"))
            value = _norm(record.get("value") or metadata.get("value"))
            if not entity_key or not fact_key or not value:
                record.pop("conflict_set_id", None)
                metadata.pop("conflict_set_id", None)
                continue
            record["entity_key"] = entity_key
            record["fact_key"] = fact_key
            record["value"] = value
            record["metadata"] = {**metadata, "entity_key": entity_key, "fact_key": fact_key, "value": value}
            groups.setdefault((entity_key, fact_key), []).append(record)

        conflicted_ids: set[str] = set()
        for (entity_key, fact_key), records in groups.items():
            values = {str(record.get("value") or "") for record in records}
            if len(values) <= 1:
                continue
            conflict_id = f"conflict:{entity_key}:{fact_key}"
            for record in records:
                record_id = str(record.get("record_id") or "")
                conflicted_ids.add(record_id)
                if record.get("conflict_set_id") != conflict_id:
                    changed_ids.add(record_id)
                record["conflict_set_id"] = conflict_id
                metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
                record["metadata"] = {
                    **metadata,
                    "conflict_set_id": conflict_id,
                }

        for record in self._records.values():
            record_id = str(record.get("record_id") or "")
            if record_id in conflicted_ids:
                continue
            if record.get("conflict_set_id"):
                changed_ids.add(record_id)
            record.pop("conflict_set_id", None)
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            metadata.pop("conflict_set_id", None)
            record["metadata"] = metadata
        return changed_ids

    @staticmethod
    def _normalize_payload(payload: dict[str, Any], *, now: str) -> dict[str, Any]:
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        record_id = str(payload.get("record_id") or metadata.get("record_id") or "").strip()
        text = str(payload.get("text") or "").strip()
        memory_text = str(payload.get("memory_text") or text).strip()
        record = {
            "record_id": record_id,
            "text": text,
            "memory_text": memory_text,
            "category": str(payload.get("category") or metadata.get("category") or "faq"),
            "source": str(payload.get("source") or metadata.get("source") or ""),
            "owner": str(payload.get("owner") or metadata.get("owner") or ""),
            "updated_at": str(payload.get("updated_at") or metadata.get("updated_at") or now),
            "imported_at": str(payload.get("imported_at") or metadata.get("imported_at") or now),
            "expires_at": str(payload.get("expires_at") or metadata.get("expires_at") or ""),
            "confidence": payload.get("confidence", metadata.get("confidence", 1.0)),
            "approval_status": str(
                payload.get("approval_status") or metadata.get("approval_status") or "published"
            ),
            "entity_key": _norm(payload.get("entity_key") or metadata.get("entity_key")),
            "fact_key": _norm(payload.get("fact_key") or metadata.get("fact_key")),
            "value": _norm(payload.get("value") or metadata.get("value")),
            "content_hash": str(payload.get("content_hash") or metadata.get("content_hash") or ""),
            "source_version": str(payload.get("source_version") or metadata.get("source_version") or "1"),
            "evidence_version": _int_or_default(
                payload.get("evidence_version") or metadata.get("evidence_version"),
                1,
            ),
            "approved_by": str(payload.get("approved_by") or metadata.get("approved_by") or ""),
            "approved_at": str(payload.get("approved_at") or metadata.get("approved_at") or ""),
            "rejected_by": str(payload.get("rejected_by") or metadata.get("rejected_by") or ""),
            "rejected_at": str(payload.get("rejected_at") or metadata.get("rejected_at") or ""),
            "review_note": str(payload.get("review_note") or metadata.get("review_note") or ""),
            "updated_by": str(payload.get("updated_by") or metadata.get("updated_by") or ""),
            "events": _record_events(payload, metadata),
            "metadata": dict(metadata),
        }
        record["metadata"].update({
            "record_id": record_id,
            "category": record["category"],
            "source": record["source"],
            "owner": record["owner"],
            "updated_at": record["updated_at"],
            "expires_at": record["expires_at"],
            "confidence": record["confidence"],
            "approval_status": record["approval_status"],
            "content_hash": record["content_hash"],
            "source_version": record["source_version"],
            "evidence_version": record["evidence_version"],
        })
        if record["entity_key"]:
            record["metadata"]["entity_key"] = record["entity_key"]
        if record["fact_key"]:
            record["metadata"]["fact_key"] = record["fact_key"]
        if record["value"]:
            record["metadata"]["value"] = record["value"]
        return record

    @staticmethod
    def _public_record(record: dict[str, Any]) -> dict[str, Any]:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        payload = {
            "record_id": record.get("record_id") or metadata.get("record_id") or "",
            "text": record.get("text") or "",
            "memory_text": record.get("memory_text") or record.get("text") or "",
            "category": record.get("category") or metadata.get("category") or "",
            "source": record.get("source") or metadata.get("source") or "",
            "owner": record.get("owner") or metadata.get("owner") or "",
            "updated_at": record.get("updated_at") or metadata.get("updated_at") or "",
            "expires_at": record.get("expires_at") or metadata.get("expires_at") or "",
            "confidence": record.get("confidence", metadata.get("confidence", 1.0)),
            "approval_status": record.get("approval_status") or metadata.get("approval_status") or "",
            "entity_key": record.get("entity_key") or metadata.get("entity_key") or "",
            "fact_key": record.get("fact_key") or metadata.get("fact_key") or "",
            "value": record.get("value") or metadata.get("value") or "",
            "content_hash": record.get("content_hash") or metadata.get("content_hash") or "",
            "source_version": record.get("source_version") or metadata.get("source_version") or "",
            "evidence_version": record.get("evidence_version") or metadata.get("evidence_version") or 1,
            "indexed_evidence_version": record.get("indexed_evidence_version")
            or metadata.get("indexed_evidence_version")
            or "",
            "approved_by": record.get("approved_by") or metadata.get("approved_by") or "",
            "approved_at": record.get("approved_at") or metadata.get("approved_at") or "",
            "rejected_by": record.get("rejected_by") or metadata.get("rejected_by") or "",
            "rejected_at": record.get("rejected_at") or metadata.get("rejected_at") or "",
            "review_note": record.get("review_note") or metadata.get("review_note") or "",
            "updated_by": record.get("updated_by") or metadata.get("updated_by") or "",
            "conflict_set_id": record.get("conflict_set_id") or metadata.get("conflict_set_id") or "",
            "indexed_at": record.get("indexed_at") or "",
            "deleted_at": record.get("deleted_at") or metadata.get("deleted_at") or "",
            "deleted_reason": record.get("deleted_reason") or metadata.get("deleted_reason") or "",
            "restored_at": record.get("restored_at") or metadata.get("restored_at") or "",
            "events": _record_events(record, metadata)[-10:],
            "revisions": _record_revisions(record)[-10:],
            "metadata": dict(metadata),
        }
        payload["metadata"].update({
            "record_id": payload["record_id"],
            "approval_status": payload["approval_status"],
            "category": payload["category"],
            "source": payload["source"],
            "updated_at": payload["updated_at"],
            "expires_at": payload["expires_at"],
            "content_hash": payload["content_hash"],
            "source_version": payload["source_version"],
            "evidence_version": payload["evidence_version"],
            "approved_by": payload["approved_by"],
            "approved_at": payload["approved_at"],
            "rejected_by": payload["rejected_by"],
            "rejected_at": payload["rejected_at"],
            "review_note": payload["review_note"],
            "updated_by": payload["updated_by"],
            "events": payload["events"],
        })
        if payload["conflict_set_id"]:
            payload["metadata"]["conflict_set_id"] = payload["conflict_set_id"]
        lifecycle = _lifecycle_snapshot(payload)
        payload.update(lifecycle)
        return payload

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            records = data.get("records") if isinstance(data, dict) else data
            if not isinstance(records, list):
                return
            self._records = {
                str(record.get("record_id")): record
                for record in records
                if isinstance(record, dict) and str(record.get("record_id") or "").strip()
            }
        except Exception:
            self._records = {}

    @staticmethod
    def _next_evidence_version(previous: dict[str, Any], record: dict[str, Any]) -> int:
        if not previous:
            return _int_or_default(record.get("evidence_version"), 1)
        current = _int_or_default(previous.get("evidence_version"), 1)
        for field in _PROMPT_AFFECTING_FIELDS:
            if str(previous.get(field) or "") != str(record.get(field) or ""):
                return current + 1
        return _int_or_default(record.get("evidence_version"), current)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "records": list(self._records.values()),
            "saved_at": _now_iso(),
        }
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._path)


def _now_iso() -> str:
    return datetime.now(_UTC).isoformat(timespec="seconds")


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _parse_time(value: Any) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    raw = str(value).strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=_UTC)
    return parsed.astimezone(_UTC)


def _lifecycle_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    status = str(record.get("approval_status") or "").strip().lower()
    expires_at = _parse_time(record.get("expires_at"))
    now = datetime.now(_UTC)
    expires_in_days: int | None = None
    if expires_at is not None:
        expires_in_days = (expires_at.date() - now.date()).days

    prompt_eligible = KnowledgeCatalog.is_prompt_eligible(record)
    indexed_version = str(record.get("indexed_evidence_version") or "").strip()
    evidence_version = str(record.get("evidence_version") or "").strip()
    needs_reindex = bool(
        prompt_eligible
        and (
            not str(record.get("indexed_at") or "").strip()
            or (indexed_version and evidence_version and indexed_version != evidence_version)
        )
    )

    state = "ready"
    label = "可回答"
    reason = ""
    if status == "deleted":
        state, label, reason = "deleted", "已删除", "deleted"
    elif record.get("conflict_set_id"):
        state, label, reason = "conflict", "知识冲突", str(record.get("conflict_set_id") or "")
    elif expires_at is not None and expires_at <= now:
        state, label, reason = "expired", "已过期", "expires_at"
    elif status in {"draft", ""}:
        state, label, reason = "draft", "草稿待审批", "approval_status"
    elif status == "pending":
        state, label, reason = "pending_review", "待审批", "approval_status"
    elif status == "rejected":
        state, label, reason = "rejected", "已驳回", "approval_status"
    elif status and status not in _ACTIVE_STATUSES:
        state, label, reason = "unapproved", "未批准", "approval_status"
    elif needs_reindex:
        state, label, reason = "needs_reindex", "需重建索引", "indexed_evidence_version"
    elif expires_in_days is not None and expires_in_days <= 7:
        state, label, reason = "expiring_soon", "即将过期", "expires_at"

    return {
        "prompt_eligible": prompt_eligible,
        "needs_reindex": needs_reindex,
        "expires_in_days": expires_in_days,
        "lifecycle_state": state,
        "lifecycle_label": label,
        "lifecycle_reason": reason,
    }


def _event(kind: str, *, actor: str = "", note: str = "") -> dict[str, Any]:
    return {
        "kind": str(kind or "event"),
        "actor": str(actor or ""),
        "note": str(note or ""),
        "at": _now_iso(),
    }


def _record_events(*sources: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for source in sources:
        raw = source.get("events") if isinstance(source, dict) else None
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            events.append({
                "kind": str(item.get("kind") or "event"),
                "actor": str(item.get("actor") or ""),
                "note": str(item.get("note") or ""),
                "at": str(item.get("at") or ""),
            })
    return events[-50:]


def _record_revisions(source: dict[str, Any]) -> list[dict[str, Any]]:
    raw = source.get("revisions") if isinstance(source, dict) else None
    if not isinstance(raw, list):
        return []
    revisions: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        snapshot = item.get("snapshot") if isinstance(item.get("snapshot"), dict) else {}
        revisions.append({
            "revision_id": str(item.get("revision_id") or ""),
            "actor": str(item.get("actor") or ""),
            "note": str(item.get("note") or ""),
            "created_at": str(item.get("created_at") or ""),
            "evidence_version": _int_or_default(item.get("evidence_version"), 1),
            "snapshot": dict(snapshot),
        })
    return revisions[-25:]


def _revision_from_record(record: dict[str, Any], *, actor: str = "", note: str = "") -> dict[str, Any]:
    fields = _PROMPT_AFFECTING_FIELDS | {
        "record_id",
        "category",
        "source",
        "owner",
        "approval_status",
        "source_version",
    }
    snapshot = {field: record.get(field, "") for field in sorted(fields)}
    version = _int_or_default(record.get("evidence_version"), 1)
    return {
        "revision_id": f"rev-{version}-{len(_record_revisions(record)) + 1}",
        "actor": str(actor or record.get("updated_by") or record.get("owner") or ""),
        "note": str(note or "before_prompt_affecting_change"),
        "created_at": _now_iso(),
        "evidence_version": version,
        "snapshot": snapshot,
    }


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
