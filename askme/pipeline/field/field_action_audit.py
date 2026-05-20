"""Append-only field action audit helpers."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

FIELD_ACTION_AUDIT_GENESIS = "GENESIS"
FIELD_ACTION_AUDIT_HASH_ALG = "sha256"
FIELD_ACTION_AUDIT_SIGNATURE_ALG = "hmac-sha256"


class FieldActionAuditIntegrityError(ValueError):
    """Raised when an append-only action audit file is not safe to append to."""


def audit_review_path_from_field_config(config: dict[str, Any]) -> Path | None:
    audit_cfg = config.get("audit") if isinstance(config.get("audit"), dict) else {}
    review_cfg = audit_cfg.get("review") if isinstance(audit_cfg.get("review"), dict) else {}
    raw = (
        review_cfg.get("path")
        or review_cfg.get("jsonl_path")
        or audit_cfg.get("review_path")
        or config.get("audit_review_path")
    )
    text = str(raw or "artifacts/audit/reviews.jsonl").strip()
    return Path(text) if text else None


def field_action_audit_hash(record: dict[str, Any]) -> str:
    """Return the stable hash for one append-only field action audit record."""
    payload = dict(record)
    payload.pop("record_hash", None)
    payload.pop("record_signature", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def field_action_audit_signature(record: dict[str, Any], *, secret: str) -> str:
    payload = dict(record)
    payload.pop("record_signature", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(secret.encode("utf-8"), encoded, hashlib.sha256).hexdigest()


def next_field_action_audit_checkpoint(path: Path) -> tuple[int, str]:
    """Return a best-effort checkpoint for read-only legacy diagnostics.

    Write paths must use ``strict_field_action_audit_checkpoint`` so corrupt
    JSONL, broken hash chains, and signature failures cannot be skipped.
    """

    if not path.exists():
        return 1, FIELD_ACTION_AUDIT_GENESIS
    latest_hash = FIELD_ACTION_AUDIT_GENESIS
    sequence = 0
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            sequence += 1
            if record.get("record_hash"):
                latest_hash = str(record["record_hash"])
    return sequence + 1, latest_hash


def strict_field_action_audit_checkpoint(path: Path, *, secret: str = "") -> tuple[int, str]:
    """Return the next checkpoint only when the existing audit chain is intact."""

    if not path.exists():
        return 1, FIELD_ACTION_AUDIT_GENESIS

    previous_hash = FIELD_ACTION_AUDIT_GENESIS
    latest_hash = previous_hash
    sequence = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FieldActionAuditIntegrityError(
                f"invalid_json at action audit line {line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise FieldActionAuditIntegrityError(
                f"record_not_object at action audit line {line_number}"
            )

        sequence += 1
        if record.get("sequence") != sequence:
            raise FieldActionAuditIntegrityError(
                f"sequence_mismatch at action audit line {line_number}: "
                f"expected {sequence}, got {record.get('sequence')!r}"
            )
        if record.get("prev_hash") != previous_hash:
            raise FieldActionAuditIntegrityError(
                f"prev_hash_mismatch at action audit line {line_number}"
            )

        expected_hash = field_action_audit_hash(record)
        actual_hash = str(record.get("record_hash") or "")
        if actual_hash != expected_hash:
            raise FieldActionAuditIntegrityError(
                f"record_hash_mismatch at action audit line {line_number}"
            )

        if secret:
            if record.get("signature_alg") != FIELD_ACTION_AUDIT_SIGNATURE_ALG:
                raise FieldActionAuditIntegrityError(
                    f"signature_alg_mismatch at action audit line {line_number}"
                )
            expected_signature = field_action_audit_signature(record, secret=secret)
            actual_signature = str(record.get("record_signature") or "")
            if not hmac.compare_digest(actual_signature, expected_signature):
                raise FieldActionAuditIntegrityError(
                    f"record_signature_mismatch at action audit line {line_number}"
                )

        latest_hash = actual_hash
        previous_hash = latest_hash

    return sequence + 1, latest_hash


def field_action_audit_counts_by_event(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        event_id = str(event.get("event_id") or "")
        audit = event.get("action_audit")
        if not event_id or not isinstance(audit, list):
            continue
        counts[event_id] = counts.get(event_id, 0) + len(audit)
    return counts


__all__ = [
    "FIELD_ACTION_AUDIT_GENESIS",
    "FIELD_ACTION_AUDIT_HASH_ALG",
    "FIELD_ACTION_AUDIT_SIGNATURE_ALG",
    "FieldActionAuditIntegrityError",
    "audit_review_path_from_field_config",
    "field_action_audit_counts_by_event",
    "field_action_audit_hash",
    "field_action_audit_signature",
    "next_field_action_audit_checkpoint",
    "strict_field_action_audit_checkpoint",
]
