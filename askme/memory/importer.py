"""Knowledge import helpers for askme memory/RAG."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any

from askme.memory.bridge import MemoryBridge

DEFAULT_CATEGORY = "faq"
SUPPORTED_CATEGORIES = {"location", "equipment", "route", "faq", "note"}
_UTC = timezone(timedelta(0))


@dataclass(frozen=True)
class KnowledgeRecord:
    """One curated fact ready to be indexed into long-term memory."""

    text: str
    category: str = DEFAULT_CATEGORY
    source: str = ""
    owner: str = ""
    updated_at: str = ""
    expires_at: str = ""
    confidence: float = 1.0
    approval_status: str = "published"
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_category(self) -> str:
        category = str(self.category or DEFAULT_CATEGORY).strip().lower()
        return category if category in SUPPORTED_CATEGORIES else DEFAULT_CATEGORY

    def to_memory_text(self) -> str:
        prefix = self.normalized_category()
        return f"[{prefix}] {self.text.strip()}"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "category": self.normalized_category(),
            "source": self.source,
            "owner": self.owner,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "confidence": self.confidence,
            "approval_status": self.approval_status,
            **self.metadata,
        }


@dataclass(frozen=True)
class ImportResult:
    """Summary of one import operation."""

    source: str
    parsed: int
    imported: int
    skipped: int
    errors: list[str]
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "parsed": self.parsed,
            "imported": self.imported,
            "skipped": self.skipped,
            "errors": self.errors,
            "dry_run": self.dry_run,
        }


def parse_knowledge_file(
    path: str | Path,
    *,
    source: str | None = None,
    category: str | None = None,
) -> list[KnowledgeRecord]:
    """Parse Markdown, JSON, JSONL, or CSV knowledge into records."""
    file_path = Path(path)
    source_name = source or file_path.name
    suffix = file_path.suffix.lower()
    if suffix in {".json"}:
        return _records_from_json(file_path, source=source_name, category=category)
    if suffix in {".jsonl", ".ndjson"}:
        return _records_from_jsonl(file_path, source=source_name, category=category)
    if suffix in {".csv"}:
        return _records_from_csv(file_path, source=source_name, category=category)
    return _records_from_markdown(file_path, source=source_name, category=category)


def parse_knowledge_text(
    content: str,
    *,
    filename: str = "knowledge.md",
    source: str | None = None,
    category: str | None = None,
) -> list[KnowledgeRecord]:
    """Parse uploaded knowledge text without requiring multipart/temp files."""
    source_name = source or filename or "inline"
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".json"}:
        return _records_from_json_text(content, source=source_name, category=category)
    if suffix in {".jsonl", ".ndjson"}:
        return _records_from_jsonl_text(content, source=source_name, category=category)
    if suffix in {".csv"}:
        return _records_from_csv_text(content, source=source_name, category=category)
    return _records_from_markdown_text(content, source=source_name, category=category)


async def import_knowledge_file(
    path: str | Path,
    *,
    bridge: MemoryBridge | None = None,
    source: str | None = None,
    category: str | None = None,
    dry_run: bool = False,
) -> ImportResult:
    """Parse and save one knowledge file into the configured memory backend."""
    file_path = Path(path)
    errors: list[str] = []
    if not file_path.exists():
        return ImportResult(
            source=str(file_path),
            parsed=0,
            imported=0,
            skipped=0,
            errors=[f"file_not_found: {file_path}"],
            dry_run=dry_run,
        )
    try:
        records = parse_knowledge_file(file_path, source=source, category=category)
    except Exception as exc:
        return ImportResult(
            source=str(file_path),
            parsed=0,
            imported=0,
            skipped=0,
            errors=[f"parse_error: {type(exc).__name__}: {exc}"],
            dry_run=dry_run,
        )
    imported = 0
    skipped = 0
    memory = bridge or MemoryBridge()

    for index, record in enumerate(records, start=1):
        if not record.text.strip():
            skipped += 1
            continue
        if dry_run:
            imported += 1
            continue
        try:
            await memory.save_fact(record.to_memory_text(), record.to_metadata())
            imported += 1
        except Exception as exc:  # pragma: no cover - defensive, bridge should degrade
            errors.append(f"record {index}: {type(exc).__name__}: {exc}")

    return ImportResult(
        source=str(file_path),
        parsed=len(records),
        imported=imported,
        skipped=skipped,
        errors=errors,
        dry_run=dry_run,
    )


def _records_from_json(
    path: Path,
    *,
    source: str,
    category: str | None,
) -> list[KnowledgeRecord]:
    raw = json.loads(_read_text(path))
    if isinstance(raw, dict):
        items = raw.get("records") or raw.get("items") or raw.get("knowledge") or [raw]
    else:
        items = raw
    return [_record_from_mapping(item, source=source, category=category) for item in items]


def _records_from_json_text(
    content: str,
    *,
    source: str,
    category: str | None,
) -> list[KnowledgeRecord]:
    raw = json.loads(content)
    if isinstance(raw, dict):
        items = raw.get("records") or raw.get("items") or raw.get("knowledge") or [raw]
    else:
        items = raw
    return [_record_from_mapping(item, source=source, category=category) for item in items]


def _records_from_jsonl(
    path: Path,
    *,
    source: str,
    category: str | None,
) -> list[KnowledgeRecord]:
    records = []
    for line in _read_text(path).splitlines():
        line = line.strip()
        if line:
            records.append(_record_from_mapping(json.loads(line), source=source, category=category))
    return records


def _records_from_jsonl_text(
    content: str,
    *,
    source: str,
    category: str | None,
) -> list[KnowledgeRecord]:
    records = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            records.append(_record_from_mapping(json.loads(line), source=source, category=category))
    return records


def _records_from_csv(
    path: Path,
    *,
    source: str,
    category: str | None,
) -> list[KnowledgeRecord]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [
            _record_from_mapping(row, source=source, category=category)
            for row in csv.DictReader(f)
        ]


def _records_from_csv_text(
    content: str,
    *,
    source: str,
    category: str | None,
) -> list[KnowledgeRecord]:
    return [
        _record_from_mapping(row, source=source, category=category)
        for row in csv.DictReader(StringIO(content))
    ]


def _records_from_markdown(
    path: Path,
    *,
    source: str,
    category: str | None,
) -> list[KnowledgeRecord]:
    lines = _read_text(path).splitlines()
    records: list[KnowledgeRecord] = []
    current_heading = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            current_heading = stripped.lstrip("#").strip()
            continue
        if stripped.startswith(("- ", "* ")):
            text = stripped[2:].strip()
        else:
            text = stripped
        if current_heading:
            text = f"{current_heading}: {text}"
        records.append(_record_from_mapping({"text": text}, source=source, category=category))
    return records


def _records_from_markdown_text(
    content: str,
    *,
    source: str,
    category: str | None,
) -> list[KnowledgeRecord]:
    records: list[KnowledgeRecord] = []
    current_heading = ""
    for line in content.lstrip("\ufeff").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            current_heading = stripped.lstrip("#").strip()
            continue
        if stripped.startswith(("- ", "* ")):
            text = stripped[2:].strip()
        else:
            text = stripped
        if current_heading:
            text = f"{current_heading}: {text}"
        records.append(_record_from_mapping({"text": text}, source=source, category=category))
    return records


def _record_from_mapping(
    item: Any,
    *,
    source: str,
    category: str | None,
) -> KnowledgeRecord:
    if not isinstance(item, dict):
        item = {"text": str(item)}
    text = _first_non_empty(item, "text", "content", "fact", "answer", "description", "name")
    question = _first_non_empty(item, "question", "q")
    if question and text and question not in text:
        text = f"问: {question}\n答: {text}"
    metadata = {k: v for k, v in item.items() if k not in _KNOWN_FIELDS}
    updated_at = str(item.get("updated_at") or item.get("date") or _now_iso())
    text_hash = hashlib.sha256(str(text or "").strip().encode("utf-8")).hexdigest()
    metadata.setdefault("content_hash", text_hash)
    metadata.setdefault("record_id", str(item.get("record_id") or f"know_{text_hash[:16]}"))
    return KnowledgeRecord(
        text=str(text or "").strip(),
        category=str(category or item.get("category") or item.get("type") or DEFAULT_CATEGORY),
        source=str(item.get("source") or source),
        owner=str(item.get("owner") or ""),
        updated_at=updated_at,
        expires_at=str(item.get("expires_at") or ""),
        confidence=_float_or_default(item.get("confidence"), 1.0),
        approval_status=str(item.get("approval_status") or "published"),
        metadata=metadata,
    )


def _read_text(path: Path) -> str:
    """Read curated knowledge with common Chinese Windows encoding fallbacks."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="gb18030")


def _first_non_empty(mapping: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _now_iso() -> str:
    return datetime.now(_UTC).isoformat(timespec="seconds")


_KNOWN_FIELDS = {
    "text",
    "content",
    "fact",
    "answer",
    "description",
    "name",
    "question",
    "q",
    "category",
    "type",
    "source",
    "owner",
    "updated_at",
    "date",
    "expires_at",
    "confidence",
    "approval_status",
    "record_id",
    "content_hash",
}
