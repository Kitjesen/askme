"""MemoryModule 鈥?wraps the four-layer memory stack as a declarative module.

Canonical wiring::

    session_memory = SessionMemory(llm=llm)
    conversation = ConversationManager(session_memory=session_memory, metrics=ota_metrics)
    memory = MemoryBridge()
    episodic = EpisodicMemory(llm=llm)
    memory_system = MemorySystem(llm=llm, conversation=conversation, ...)

The LLMClient is obtained from LLMModule via ``In[LLMClient]`` auto-wiring.
After wiring, ``self.llm_client`` is the LLMModule instance (not the client
directly). Access the client via ``self.llm_client.client``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from askme.llm.client import LLMClient
from askme.memory.bridge import MemoryBridge
from askme.memory.catalog import KnowledgeCatalog
from askme.memory.conversation import ConversationManager
from askme.memory.episodic_memory import EpisodicMemory
from askme.memory.index_jobs import KnowledgeIndexJobStore
from askme.memory.session import SessionMemory
from askme.memory.system import MemorySystem
from askme.runtime.module import In, Module, ModuleRegistry, Out
from askme.schemas.messages import MemoryContext

logger = logging.getLogger(__name__)
_UTC = timezone(timedelta(0))


def _parse_optional_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=_UTC)
    return parsed.astimezone(_UTC)


class MemoryModule(Module):
    """Provides the four-layer memory stack to the runtime."""

    name = "memory"
    provides = ("conversation", "episodic", "vector_memory", "session_memory")

    llm_client: In[LLMClient]
    memory_context: Out[MemoryContext]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        # Get LLMClient from LLMModule (auto-wired In port gives module instance)
        llm_mod = self.llm_client
        llm: LLMClient | None = getattr(llm_mod, "client", None) if llm_mod else None

        ota_metrics = getattr(llm_mod, "ota_metrics", None) if llm_mod else None

        self._session_memory = SessionMemory(llm=llm)
        self._conversation = ConversationManager(
            session_memory=self._session_memory,
            metrics=ota_metrics,
        )
        self._knowledge_catalog = KnowledgeCatalog(config=cfg)
        self._memory_bridge = MemoryBridge(
            config=cfg,
            knowledge_catalog=self._knowledge_catalog,
        )
        self._episodic = EpisodicMemory(llm=llm)
        self._memory_system = MemorySystem(
            llm=llm,
            conversation=self._conversation,
            session_memory=self._session_memory,
            episodic=self._episodic,
            vector_memory=self._memory_bridge,
        )
        self._knowledge_job_store = KnowledgeIndexJobStore(config=cfg)
        self._warmup_task: asyncio.Task[None] | None = None
        logger.info("MemoryModule: built (llm=%s)", "wired" if llm else "none")

    # -- typed accessors ------------------------------------------------
    @property
    def conversation(self) -> ConversationManager:
        """L1 conversation manager."""
        return self._conversation

    @property
    def session_memory(self) -> SessionMemory:
        """L2 session memory."""
        return self._session_memory

    @property
    def episodic(self) -> EpisodicMemory:
        """L3 episodic memory."""
        return self._episodic

    @property
    def memory_bridge(self) -> MemoryBridge:
        """L4 vector memory bridge."""
        return self._memory_bridge

    @property
    def memory_system(self) -> MemorySystem:
        """Unified memory system."""
        return self._memory_system

    async def start(self) -> None:
        """Warm memory backends after runtime start without delaying readiness."""
        if self._warmup_task is None or self._warmup_task.done():
            self._warmup_task = asyncio.create_task(
                self._memory_bridge.warmup(),
                name="memory_warmup",
            )

    async def stop(self) -> None:
        """Consolidate memories on shutdown (best-effort, non-blocking)."""
        warmup_task = getattr(self, "_warmup_task", None)
        if warmup_task is not None and not warmup_task.done():
            warmup_task.cancel()
            await asyncio.gather(warmup_task, return_exceptions=True)
        try:
            robotmem = getattr(self._memory_bridge, "_robotmem", None)
            if robotmem and robotmem.available:
                llm_mod = self.llm_client
                llm = getattr(llm_mod, "client", None) if llm_mod else None
                if llm:
                    n = await robotmem.consolidate(llm, batch_size=20)
                    if n:
                        logger.info("MemoryModule: consolidated %d facts on shutdown.", n)
        except Exception as exc:
            logger.debug("MemoryModule: consolidation on shutdown failed: %s", exc)

    def health(self) -> dict[str, Any]:
        bridge_health = self._memory_bridge.health()
        return {
            "status": "ok",
            "conversation_len": len(self._conversation.history),
            "episodic_buffer_len": len(self._episodic._buffer),
            "rag": bridge_health,
            "knowledge_catalog": self._knowledge_catalog.health(),
            "knowledge_index_jobs": self._knowledge_job_store.health(),
            **bridge_health,
        }

    async def search_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """HTTP-facing memory/RAG search contract for dashboards and readiness checks."""
        query = str(payload.get("query") or payload.get("text") or "").strip()
        if not query:
            return {
                "query": query,
                "results": [],
                "rag": self._memory_bridge.health(),
                "warnings": ["empty_query"],
            }

        await self._memory_bridge.retrieve(query)
        health = self._memory_bridge.health()
        evidence = health.get("last_evidence")
        dropped = health.get("last_dropped_evidence")
        answer_policy = health.get("last_answer_policy")
        return {
            "query": query,
            "results": evidence if isinstance(evidence, list) else [],
            "rag": {
                "enabled": health.get("enabled", False),
                "backend": health.get("backend", ""),
                "available": health.get("available", False),
                "last_backend": health.get("last_backend", ""),
                "last_retrieve_ms": health.get("last_retrieve_ms"),
                "last_retrieved_items": health.get("last_retrieved_items", 0),
                "fallback_count": health.get("fallback_count", 0),
                "last_fallback_reason": health.get("last_fallback_reason", ""),
                "dropped_evidence": dropped if isinstance(dropped, list) else [],
                "answer_policy": answer_policy if isinstance(answer_policy, dict) else {},
            },
            "warnings": [],
        }

    async def preview_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Parse uploaded knowledge content and return records without indexing."""
        from askme.memory.importer import parse_knowledge_text

        content = str(payload.get("content") or "")
        filename = str(payload.get("filename") or payload.get("source") or "knowledge.md")
        source = str(payload.get("source") or filename)
        category = payload.get("category")
        if not content.strip():
            return {
                "source": source,
                "parsed": 0,
                "records": [],
                "errors": ["empty_content"],
                "dry_run": True,
            }
        try:
            records = parse_knowledge_text(
                content,
                filename=filename,
                source=source,
                category=str(category) if category else None,
            )
        except Exception as exc:
            return {
                "source": source,
                "parsed": 0,
                "records": [],
                "errors": [f"parse_error: {type(exc).__name__}: {exc}"],
                "dry_run": True,
            }
        return {
            "source": source,
            "parsed": len(records),
            "records": [self._knowledge_record_payload(record) for record in records[:100]],
            "errors": [],
            "dry_run": True,
        }

    async def import_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Parse and publish uploaded knowledge content into the configured RAG backend."""
        preview = await self.preview_payload(payload)
        if preview.get("errors"):
            return {
                "source": preview.get("source", ""),
                "parsed": 0,
                "imported": 0,
                "skipped": 0,
                "errors": preview["errors"],
                "rag": self._memory_bridge.health(),
            }
        imported = 0
        errors: list[str] = []
        preview_records = [
            record
            for record in preview.get("records", [])
            if isinstance(record, dict)
        ]
        catalog_result = self._knowledge_catalog.upsert_payloads(preview_records)
        sync_result = await self._sync_catalog_records(catalog_result.get("records", []))
        imported += int(sync_result.get("indexed", 0) or 0)
        errors.extend(sync_result.get("errors", []))
        skipped = max(0, int(preview.get("parsed", 0) or 0) - imported)
        return {
            "source": preview.get("source", ""),
            "parsed": int(preview.get("parsed", 0) or 0),
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
            "catalog": self._knowledge_catalog.health(),
            "rag": self._memory_bridge.health(),
        }

    async def list_knowledge_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """List knowledge records for Knowledge Console."""
        limit = _int_or_default(payload.get("limit"), 100)
        offset = _int_or_default(payload.get("offset"), 0)
        catalog = self._knowledge_catalog.list_records(limit=limit, offset=offset)
        return {
            "records": catalog.get("records", []),
            "total": catalog.get("total", 0),
            "backend": catalog.get("backend", ""),
            "catalog": catalog.get("catalog", {}),
            "operations": self._knowledge_operations_payload(catalog.get("records", [])),
            "index_jobs": self._knowledge_job_store.list_jobs(limit=_int_or_default(
                payload.get("job_limit"),
                5,
            )),
            "rag": self._memory_bridge.health(),
        }

    def _knowledge_operations_payload(self, records: Any) -> dict[str, Any]:
        """Return product-facing operating queues for Knowledge Console."""

        if not isinstance(records, list):
            records = []
        now = datetime.now(_UTC)
        approval_queue: list[dict[str, Any]] = []
        conflict_queue: list[dict[str, Any]] = []
        expiry_queue: list[dict[str, Any]] = []
        reindex_queue: list[dict[str, Any]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            state = str(record.get("lifecycle_state") or "").lower()
            status = str(record.get("approval_status") or "").lower()
            item = {
                "record_id": record.get("record_id", ""),
                "source": record.get("source", ""),
                "owner": record.get("owner", ""),
                "category": record.get("category", ""),
                "updated_at": record.get("updated_at", ""),
                "evidence_version": record.get("evidence_version", ""),
            }
            if state in {"unapproved", "pending_review"} or status in {"draft", "pending", "pending_review"}:
                approval_queue.append({**item, "reason": "needs_approval"})
            if record.get("conflict_set_id") or state in {"conflict", "conflicted"}:
                conflict_queue.append({
                    **item,
                    "reason": record.get("conflict_set_id") or "conflict",
                })
            if record.get("needs_reindex") or state == "needs_reindex":
                reindex_queue.append({**item, "reason": "evidence_version_not_indexed"})
            expires_at = _parse_optional_time(record.get("expires_at"))
            if expires_at is not None:
                days = (expires_at.date() - now.date()).days
                if days <= 7:
                    expiry_queue.append({
                        **item,
                        "expires_at": record.get("expires_at", ""),
                        "expires_in_days": days,
                        "reason": "expired" if days < 0 else "expiring_soon",
                    })
        return {
            "approval_queue": approval_queue,
            "conflict_queue": conflict_queue,
            "expiry_queue": expiry_queue,
            "reindex_queue": reindex_queue,
            "release_cadence": {
                "mode": "manual",
                "next_release_window": self._next_release_window(),
                "next_release_requires": [
                    "approval_queue_empty",
                    "conflict_queue_empty",
                    "expiry_queue_reviewed",
                    "reindex_queue_empty",
                ],
                "blocked": bool(approval_queue or conflict_queue or expiry_queue or reindex_queue),
                "blockers": {
                    "approval": len(approval_queue),
                    "conflict": len(conflict_queue),
                    "expiry": len(expiry_queue),
                    "reindex": len(reindex_queue),
                },
            },
            "missing_product_capabilities": [
                "scheduled_expiry_reminders",
                "scheduled_release_automation",
            ],
        }

    @staticmethod
    def _next_release_window() -> str:
        now = datetime.now(_UTC)
        return (now + timedelta(days=1)).replace(hour=10, minute=0, second=0, microsecond=0).isoformat()

    async def update_knowledge_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Patch approval/deletion metadata for a knowledge record."""
        record_id = str(payload.get("record_id") or "").strip()
        action = str(payload.get("action") or "").strip().lower()
        if action in {"rebuild", "rebuild_index", "reindex"}:
            return await self.rebuild_knowledge_index_payload(payload)
        if action in {"diff", "compare"}:
            return await self.diff_knowledge_payload(payload)
        if action == "rollback":
            return await self.rollback_knowledge_payload(payload)
        if action in {"resolve_conflict", "resolve"}:
            return await self.resolve_knowledge_conflict_payload(payload)
        if action in {"bulk", "bulk_update", "bulk_metadata"} or isinstance(payload.get("updates"), list):
            return await self.bulk_update_knowledge_payload(payload)
        if isinstance(payload.get("record_ids"), list) and not record_id:
            return await self.bulk_update_knowledge_payload(payload)

        patch = self._metadata_patch_for_action(payload)
        if not record_id:
            return {"updated": False, "error": "missing_record_id"}
        result = self._knowledge_catalog.update_metadata(record_id, patch)
        if result.get("updated"):
            changed = result.get("changed_records")
            records = changed if isinstance(changed, list) else [result.get("record", {})]
            sync = await self._sync_catalog_records([r for r in records if isinstance(r, dict)])
            return {
                **result,
                "sync": sync,
                "catalog": self._knowledge_catalog.health(),
                "rag": self._memory_bridge.health(),
            }
        return {**result, "catalog": self._knowledge_catalog.health(), "rag": self._memory_bridge.health()}

    async def bulk_update_knowledge_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Patch metadata for multiple knowledge records and sync changed records."""
        updates = self._bulk_updates_from_payload(payload)
        if not updates:
            return {
                "updated": 0,
                "failed": 0,
                "records": [],
                "errors": [{"error": "missing_updates"}],
                "catalog": self._knowledge_catalog.health(),
                "rag": self._memory_bridge.health(),
            }

        result = self._knowledge_catalog.update_metadata_many(updates)
        changed = [
            record
            for record in result.get("changed_records", [])
            if isinstance(record, dict)
        ]
        sync = await self._sync_catalog_records(changed)
        return {
            **result,
            "sync": sync,
            "catalog": self._knowledge_catalog.health(),
            "rag": self._memory_bridge.health(),
        }

    async def diff_knowledge_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a field-level diff between current knowledge and a prior revision."""
        record_id = str(payload.get("record_id") or "").strip()
        if not record_id:
            return {"found": False, "error": "missing_record_id"}
        result = self._knowledge_catalog.diff_record(
            record_id,
            str(payload.get("revision_id") or "").strip() or None,
        )
        return {
            **result,
            "catalog": self._knowledge_catalog.health(),
            "rag": self._memory_bridge.health(),
        }

    async def rollback_knowledge_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Rollback a knowledge record to a catalog revision and re-sync eligible evidence."""
        record_id = str(payload.get("record_id") or "").strip()
        if not record_id:
            return {"updated": False, "error": "missing_record_id"}
        result = self._knowledge_catalog.rollback_record(
            record_id,
            str(payload.get("revision_id") or "").strip() or None,
            actor=str(payload.get("operator_id") or payload.get("updated_by") or "").strip(),
            note=str(payload.get("review_note") or payload.get("reason") or "").strip(),
        )
        changed = [
            record
            for record in result.get("changed_records", [])
            if isinstance(record, dict)
        ]
        sync = await self._sync_catalog_records(changed)
        return {
            **result,
            "sync": sync,
            "catalog": self._knowledge_catalog.health(),
            "rag": self._memory_bridge.health(),
        }

    async def resolve_knowledge_conflict_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Resolve a conflict by keeping one record and rejecting its conflicting peers."""
        keep_record_id = str(
            payload.get("keep_record_id") or payload.get("record_id") or ""
        ).strip()
        if not keep_record_id:
            return {
                "updated": 0,
                "failed": 1,
                "error": "missing_keep_record_id",
                "catalog": self._knowledge_catalog.health(),
                "rag": self._memory_bridge.health(),
            }
        now = datetime.now(_UTC).isoformat(timespec="seconds")
        operator_id = str(payload.get("operator_id") or payload.get("updated_by") or "").strip()
        note = str(payload.get("review_note") or payload.get("reason") or "resolve_conflict")
        conflict_id = str(payload.get("conflict_set_id") or "").strip()
        records = self._knowledge_catalog.list_records(limit=500).get("records", [])
        keep = next(
            (
                record
                for record in records
                if isinstance(record, dict) and record.get("record_id") == keep_record_id
            ),
            None,
        )
        if not isinstance(keep, dict):
            return {
                "updated": 0,
                "failed": 1,
                "error": "record_not_found",
                "record_id": keep_record_id,
                "catalog": self._knowledge_catalog.health(),
                "rag": self._memory_bridge.health(),
            }
        if not conflict_id:
            conflict_id = str(keep.get("conflict_set_id") or "").strip()
        reject_ids = [
            str(record_id or "").strip()
            for record_id in payload.get("reject_record_ids", [])
            if str(record_id or "").strip()
        ] if isinstance(payload.get("reject_record_ids"), list) else []
        if not reject_ids and conflict_id:
            reject_ids = [
                str(record.get("record_id") or "")
                for record in records
                if isinstance(record, dict)
                and record.get("conflict_set_id") == conflict_id
                and record.get("record_id") != keep_record_id
            ]
        updates = [{
            "record_id": keep_record_id,
            "patch": {
                "approval_status": "published",
                "conflict_set_id": "",
                "approved_by": operator_id,
                "approved_at": now,
                "updated_by": operator_id,
                "review_note": note,
            },
        }]
        updates.extend({
            "record_id": record_id,
            "patch": {
                "approval_status": "rejected",
                "rejected_by": operator_id,
                "rejected_at": now,
                "updated_by": operator_id,
                "review_note": note,
            },
        } for record_id in reject_ids)
        result = self._knowledge_catalog.update_metadata_many(updates)
        changed = [
            record
            for record in result.get("changed_records", [])
            if isinstance(record, dict)
        ]
        sync = await self._sync_catalog_records(changed)
        return {
            **result,
            "action": "resolve_conflict",
            "keep_record_id": keep_record_id,
            "rejected_record_ids": reject_ids,
            "sync": sync,
            "catalog": self._knowledge_catalog.health(),
            "rag": self._memory_bridge.health(),
        }

    async def rebuild_knowledge_index_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Re-index catalog-approved records into the configured RAG backend."""
        started = datetime.now(_UTC)
        started_at = started.isoformat(timespec="seconds")
        job_id = f"knowledge_rebuild_{started.strftime('%Y%m%dT%H%M%S%fZ')}"
        operator_id = str(payload.get("operator_id") or payload.get("updated_by") or "").strip()
        record_ids = [
            str(record_id or "").strip()
            for record_id in payload.get("record_ids", [])
            if str(record_id or "").strip()
        ] if isinstance(payload.get("record_ids"), list) else None
        limit = payload.get("limit")
        selected = self._knowledge_catalog.records_for_rebuild(
            record_ids=record_ids,
            include_ineligible=bool(payload.get("include_ineligible", False)),
            limit=_int_or_default(limit, 0) if limit is not None else None,
            offset=_int_or_default(payload.get("offset"), 0),
        )
        records = [
            record
            for record in selected.get("records", [])
            if isinstance(record, dict)
        ]
        sync = await self._sync_catalog_records(records)
        status = "completed" if not sync.get("errors") else "completed_with_errors"
        completed = datetime.now(_UTC)
        bridge_health = self._memory_bridge.health()
        job = self._knowledge_job_store.record({
            "job_id": job_id,
            "type": "knowledge_rebuild_index",
            "status": status,
            "operator_id": operator_id,
            "started_at": started_at,
            "completed_at": completed.isoformat(timespec="seconds"),
            "duration_ms": int((completed - started).total_seconds() * 1000),
            "requested_record_ids": record_ids or [],
            "scanned": selected.get("total", 0),
            "eligible": selected.get("eligible", 0),
            "selected": len(records),
            "indexed": sync.get("indexed", 0),
            "skipped": sync.get("skipped", 0) + selected.get("skipped", 0),
            "errors": sync.get("errors", []),
            "record_ids": selected.get("record_ids", []),
            "backend": bridge_health.get("last_backend") or bridge_health.get("backend") or "",
            "fallback_reason": bridge_health.get("last_fallback_reason") or "",
            "include_ineligible": bool(payload.get("include_ineligible", False)),
        })
        return {
            "job": job,
            "scanned": selected.get("total", 0),
            "eligible": selected.get("eligible", 0),
            "selected": len(records),
            "indexed": sync.get("indexed", 0),
            "skipped": sync.get("skipped", 0) + selected.get("skipped", 0),
            "errors": sync.get("errors", []),
            "record_ids": selected.get("record_ids", []),
            "index_jobs": self._knowledge_job_store.list_jobs(limit=5),
            "catalog": self._knowledge_catalog.health(),
            "rag": bridge_health,
        }

    @staticmethod
    def _metadata_patch_for_action(payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action") or "").strip().lower()
        patch = payload.get("patch") if isinstance(payload.get("patch"), dict) else {}
        now = datetime.now(_UTC).isoformat(timespec="seconds")
        actor = str(payload.get("operator_id") or payload.get("updated_by") or "").strip()
        note = str(payload.get("review_note") or payload.get("reason") or "").strip()
        if action == "publish":
            return {
                **patch,
                "approval_status": "published",
                "approved_by": actor,
                "approved_at": now,
                "updated_by": actor,
                "review_note": note,
                "updated_at": now,
            }
        if action == "approve":
            return {
                **patch,
                "approval_status": "approved",
                "approved_by": actor,
                "approved_at": now,
                "updated_by": actor,
                "review_note": note,
                "updated_at": now,
            }
        if action == "reject":
            return {
                **patch,
                "approval_status": "rejected",
                "rejected_by": actor,
                "rejected_at": now,
                "updated_by": actor,
                "review_note": note,
                "updated_at": now,
            }
        if action == "delete":
            return {
                **patch,
                "approval_status": "deleted",
                "deleted_at": now,
                "deleted_reason": str(payload.get("reason") or "knowledge_console"),
                "updated_by": actor,
                "review_note": note,
            }
        if action == "restore":
            return {
                **patch,
                "approval_status": str(payload.get("approval_status") or "published"),
                "restored_at": now,
                "updated_by": actor,
                "review_note": note,
                "updated_at": now,
            }
        return dict(patch)

    def _bulk_updates_from_payload(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        updates_payload = payload.get("updates")
        if isinstance(updates_payload, list):
            updates: list[dict[str, Any]] = []
            for item in updates_payload:
                if not isinstance(item, dict):
                    updates.append({})
                    continue
                item_payload = {**payload, **item}
                item_payload["patch"] = {
                    **(payload.get("patch") if isinstance(payload.get("patch"), dict) else {}),
                    **(item.get("patch") if isinstance(item.get("patch"), dict) else {}),
                }
                updates.append({
                    "record_id": item.get("record_id"),
                    "patch": self._metadata_patch_for_action(item_payload),
                })
            return updates

        record_ids = payload.get("record_ids")
        if not isinstance(record_ids, list):
            return []
        patch = self._metadata_patch_for_action(payload)
        return [
            {"record_id": record_id, "patch": patch}
            for record_id in record_ids
            if str(record_id or "").strip()
        ]

    @staticmethod
    def _knowledge_record_payload(record: Any) -> dict[str, Any]:
        metadata = record.to_metadata()
        return {
            "text": record.text,
            "memory_text": record.to_memory_text(),
            "category": record.normalized_category(),
            "source": record.source,
            "owner": record.owner,
            "updated_at": record.updated_at,
            "expires_at": record.expires_at,
            "confidence": record.confidence,
            "approval_status": record.approval_status,
            "metadata": metadata,
        }

    async def _sync_catalog_records(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        indexed = 0
        skipped = 0
        errors: list[str] = []
        for index, record in enumerate(records, start=1):
            record_id = str(record.get("record_id") or "").strip()
            metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
            patch = {
                "approval_status": record.get("approval_status") or metadata.get("approval_status") or "",
                "category": record.get("category") or metadata.get("category") or "",
                "source": record.get("source") or metadata.get("source") or "",
                "owner": record.get("owner") or metadata.get("owner") or "",
                "updated_at": record.get("updated_at") or metadata.get("updated_at") or "",
                "expires_at": record.get("expires_at") or metadata.get("expires_at") or "",
                "source_version": record.get("source_version") or metadata.get("source_version") or "",
                "evidence_version": record.get("evidence_version") or metadata.get("evidence_version") or "",
                "conflict_set_id": record.get("conflict_set_id") or metadata.get("conflict_set_id") or "",
                "deleted_at": record.get("deleted_at") or metadata.get("deleted_at") or "",
                "deleted_reason": record.get("deleted_reason") or metadata.get("deleted_reason") or "",
                "restored_at": record.get("restored_at") or metadata.get("restored_at") or "",
            }
            if record_id:
                try:
                    result = self._memory_bridge.update_knowledge_metadata(record_id, patch)
                    if inspect.isawaitable(result):
                        await result
                except Exception as exc:  # pragma: no cover - bridge should degrade
                    errors.append(f"catalog sync metadata {index}: {type(exc).__name__}: {exc}")

            if not self._knowledge_catalog.is_prompt_eligible(record):
                skipped += 1
                continue
            text = str(record.get("memory_text") or record.get("text") or "").strip()
            if not text:
                skipped += 1
                continue
            try:
                await self._memory_bridge.save_fact(text, metadata)
                if record_id:
                    self._knowledge_catalog.mark_indexed(record_id)
                indexed += 1
            except Exception as exc:  # pragma: no cover - bridge should degrade
                errors.append(f"catalog sync record {index}: {type(exc).__name__}: {exc}")
        return {"indexed": indexed, "skipped": skipped, "errors": errors}


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
