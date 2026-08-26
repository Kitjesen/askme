"""Customer-project evidence inventory and file hashing helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import quote

from askme.pipeline.field.customer_project_template_support import _mapping
from askme.pipeline.field.paths import PROJECT_ROOT


def _evidence_url(path: str) -> str:
    if not path:
        return ""
    try:
        raw = Path(path)
        resolved = raw.resolve()
        rel = resolved.relative_to(PROJECT_ROOT.resolve())
        return f"/api/field/evidence?path={quote(rel.as_posix())}"
    except (OSError, ValueError):
        return ""


def _evidence_file_modified_at(path: str) -> float:
    try:
        resolved = Path(path).resolve()
        resolved.relative_to(PROJECT_ROOT.resolve())
        return float(resolved.stat().st_mtime)
    except OSError:
        return 0.0
    except ValueError:
        return 0.0


def _customer_project_evidence_inventory(report: dict[str, Any]) -> list[dict[str, Any]]:
    readiness = _mapping(report.get("field_readiness"))
    reports = (
        readiness.get("evidence_reports")
        if isinstance(readiness.get("evidence_reports"), list)
        else []
    )
    inventory: list[dict[str, Any]] = []
    records_by_path: dict[str, dict[str, Any]] = {}
    for item in reports:
        compact = _mapping(item)
        path = str(compact.get("path") or "")
        path_key = _evidence_path_key(path)
        if not path_key or path_key in records_by_path:
            continue
        record = _evidence_file_inventory(path, evidence_url=str(compact.get("evidence_url") or ""))
        records_by_path[path_key] = record
        inventory.append(record)
    archive_path = str(_mapping(readiness.get("archive")).get("path") or "")
    archive_path_key = _evidence_path_key(archive_path)
    if archive_path_key and archive_path_key not in records_by_path:
        record = _evidence_file_inventory(archive_path, evidence_url=_evidence_url(archive_path))
        records_by_path[archive_path_key] = record
        inventory.append(record)
    onsite = _mapping(report.get("onsite_acceptance_evidence"))
    onsite_receipts = onsite.get("receipts") if isinstance(onsite.get("receipts"), list) else []
    for item in onsite_receipts:
        receipt = _mapping(item)
        path = str(receipt.get("path") or "")
        path_key = _evidence_path_key(path)
        if not path_key:
            continue
        onsite_metadata = {
            "evidence_type": "onsite_acceptance",
            "onsite_evidence_type": str(receipt.get("evidence_type") or ""),
            "receipt_id": str(receipt.get("receipt_id") or ""),
        }
        if path_key in records_by_path:
            record = records_by_path[path_key]
            if not record.get("evidence_url"):
                record["evidence_url"] = str(receipt.get("evidence_url") or _evidence_url(path))
            record.update(onsite_metadata)
            continue
        record = _evidence_file_inventory(
            path, evidence_url=str(receipt.get("evidence_url") or _evidence_url(path))
        )
        record.update(onsite_metadata)
        records_by_path[path_key] = record
        inventory.append(record)
    return inventory


def _evidence_file_inventory(
    path: str,
    *,
    evidence_url: str,
    allowed_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path,
        "evidence_url": evidence_url,
        "exists": False,
        "size_bytes": 0,
        "sha256": "",
    }
    try:
        resolved = Path(path).resolve()
    except OSError as exc:
        record["error"] = str(exc)
        return record
    roots = (PROJECT_ROOT, *allowed_roots)
    if not _is_under_allowed_root(resolved, roots):
        record["error"] = "outside_project"
        return record
    if not resolved.exists() or not resolved.is_file():
        return record
    try:
        data = resolved.read_bytes()
    except OSError as exc:
        record["error"] = str(exc)
        return record
    record.update(
        {
            "exists": True,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    )
    return record


def _is_under_allowed_root(path: Path, roots: tuple[Path, ...]) -> bool:
    for root in roots:
        try:
            path.relative_to(root.resolve())
        except (OSError, ValueError):
            continue
        return True
    return False


def _evidence_path_key(path: str) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve())
    except OSError:
        return path


__all__ = [
    "_customer_project_evidence_inventory",
    "_evidence_file_inventory",
    "_evidence_file_modified_at",
    "_evidence_url",
]
