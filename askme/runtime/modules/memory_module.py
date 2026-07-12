"""MemoryModule -wraps the four-layer memory stack as a declarative module.

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
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

from askme.llm.core.client import LLMClient
from askme.memory.core.conversation import ConversationManager
from askme.memory.core.episodic_memory import EpisodicMemory
from askme.memory.core.session import SessionMemory
from askme.memory.core.system import MemorySystem
from askme.memory.retrieval.bridge import MemoryBridge
from askme.memory.retrieval.catalog import KnowledgeCatalog
from askme.memory.retrieval.index_jobs import KnowledgeIndexJobStore
from askme.memory.retrieval.taxonomy import (
    knowledge_category_metadata,
    knowledge_category_taxonomy_payload,
)
from askme.runtime.core.module import In, Module, ModuleRegistry, Out
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


def _knowledge_document_profile(filename: str, content: str = "") -> dict[str, Any]:
    suffix = str(filename or "").rsplit(".", 1)
    ext = f".{suffix[-1].lower()}" if len(suffix) > 1 else ""
    supported = {
        ".md": ("markdown", "line_records"),
        ".markdown": ("markdown", "line_records"),
        ".txt": ("text", "line_records"),
        ".csv": ("csv", "table_records"),
        ".json": ("json", "structured_records"),
        ".jsonl": ("jsonl", "structured_records"),
        ".ndjson": ("jsonl", "structured_records"),
        "": ("text", "line_records"),
    }
    if ext not in supported:
        return {
            "filename": filename,
            "extension": ext,
            "supported": False,
            "document_type": ext.lstrip(".") or "unknown",
            "preview_mode": "unsupported",
            "reason": f"unsupported_file_type:{ext or 'unknown'}",
            "guidance": "Convert PDF, DOCX, XLSX, image, or binary files to Markdown, CSV, JSON, or plain text before importing.",
            "bytes": len(str(content or "").encode("utf-8", errors="ignore")),
        }
    document_type, preview_mode = supported[ext]
    return {
        "filename": filename,
        "extension": ext,
        "supported": True,
        "document_type": document_type,
        "preview_mode": preview_mode,
        "reason": "",
        "guidance": "Preview parses records before publishing; only approved, current, non-conflicting external records can answer customers.",
        "bytes": len(str(content or "").encode("utf-8", errors="ignore")),
    }


def _knowledge_governance_fields(payload: dict[str, Any], *, filename: str = "") -> dict[str, str]:
    profile = _knowledge_document_profile(filename)
    return {
        "quality_status": str(payload.get("quality_status") or "").strip(),
        "visibility": str(payload.get("visibility") or "").strip(),
        "customer_id": str(payload.get("customer_id") or payload.get("customer") or "").strip(),
        "project_id": str(payload.get("project_id") or payload.get("project") or "").strip(),
        "product_area": str(payload.get("product_area") or payload.get("product") or "").strip(),
        "workstream": str(payload.get("workstream") or payload.get("initiative") or "").strip(),
        "linked_object_type": str(
            payload.get("linked_object_type") or payload.get("object_type") or ""
        ).strip(),
        "linked_object_id": str(payload.get("linked_object_id") or payload.get("object_id") or "").strip(),
        "document_type": str(payload.get("document_type") or profile["document_type"] or "").strip(),
    }


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
        self._config = cfg
        self._memory_cfg = cfg.get("memory", {}) if isinstance(cfg.get("memory"), dict) else {}

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

    def replace_llm(self, llm: LLMClient) -> None:
        """Route future summaries and reflections to a replacement LLM client."""

        self._session_memory.set_llm(llm)
        self._episodic._llm = llm
        self._memory_system._llm = llm

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
        catalog_health = self._knowledge_catalog.health()
        index_jobs_health = self._knowledge_job_store.health()
        return {
            "status": "ok",
            "conversation_len": len(self._conversation.history),
            "episodic_buffer_len": len(self._episodic._buffer),
            "rag": bridge_health,
            "memory_strategy": self._memory_strategy_payload(bridge_health),
            "knowledge_catalog": catalog_health,
            "knowledge_index_jobs": index_jobs_health,
            **bridge_health,
        }

    async def health_payload(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return product-facing memory backend readiness for Dashboard/API."""

        _ = payload
        bridge_health = self._memory_bridge.health()
        catalog_health = self._knowledge_catalog.health()
        index_jobs_health = self._knowledge_job_store.health()
        strategy = self._memory_strategy_payload(bridge_health)
        bridge_enabled = bool(bridge_health.get("enabled", True))
        catalog_prompt_eligible = int(catalog_health.get("prompt_eligible") or 0)
        catalog_answer_ready = catalog_prompt_eligible > 0
        bridge_ready = bool(bridge_enabled and bridge_health.get("available"))
        ready = bool(bridge_ready or catalog_answer_ready)
        degraded = bool(
            bridge_ready
            and not bridge_health.get("selected_backend_ready")
            and bridge_health.get("fallback_ready")
        )
        warnings = self._memory_health_warnings(
            bridge_health,
            strategy=strategy,
            ready=ready,
            degraded=degraded,
            catalog_answer_ready=catalog_answer_ready,
        )
        if bridge_ready:
            status = "degraded" if degraded else "ready"
        elif catalog_answer_ready:
            status = "catalog_only"
        elif not bridge_enabled:
            status = "disabled"
        else:
            status = "not_ready"
        answer_contract = self._memory_answer_contract(bridge_health)
        return {
            "status": status,
            "ready": ready,
            "customer_status": self._memory_customer_status(
                status,
                warnings=warnings,
                catalog_answer_ready=catalog_answer_ready,
            ),
            "customer_next_step": self._memory_customer_next_step(
                status,
                warnings=warnings,
                bridge_health=bridge_health,
                catalog_health=catalog_health,
            ),
            "catalog_answer_ready": catalog_answer_ready,
            "retrieval_runtime_ready": bridge_ready,
            "current_backend": bridge_health.get("last_backend")
            or bridge_health.get("backend", ""),
            "configured_backend": bridge_health.get("configured_backend", ""),
            "selected_backend": bridge_health.get("backend", ""),
            "selected_backend_ready": bool(bridge_health.get("selected_backend_ready")),
            "selected_backend_installed": bool(bridge_health.get("selected_backend_installed")),
            "selected_backend_dependency": bridge_health.get("selected_backend_dependency", {}),
            "fallback_backend": bridge_health.get("fallback_backend", ""),
            "fallback_ready": bool(bridge_health.get("fallback_ready")),
            "fallback_backend_dependency": bridge_health.get("fallback_backend_dependency", {}),
            "backend_dependencies": bridge_health.get("backend_dependencies", {}),
            "memory_strategy": strategy,
            "paths": {
                "catalog": catalog_health.get("path", ""),
                "index_jobs": index_jobs_health.get("path", ""),
                "vector_store": bridge_health.get("vector_store_path", ""),
                "mempalace": bridge_health.get("mempalace_path", ""),
            },
            "counts": {
                "catalog_total": catalog_health.get("total", 0),
                "prompt_eligible": catalog_health.get("prompt_eligible", 0),
                "needs_review": catalog_health.get("needs_review", 0),
                "expired": catalog_health.get("expired", 0),
                "conflicted": catalog_health.get("conflicted", 0),
                "vector_size": bridge_health.get("vector_size", 0),
                "index_jobs": index_jobs_health.get("total", 0),
            },
            "rag": bridge_health,
            "knowledge_catalog": catalog_health,
            "knowledge_index_jobs": index_jobs_health,
            "answer_contract": answer_contract,
            "warnings": warnings,
        }

    def _memory_strategy_payload(self, bridge_health: dict[str, Any]) -> dict[str, Any]:
        cfg = getattr(self, "_memory_cfg", {})
        customer_backend = str(
            cfg.get("customer_knowledge_backend")
            or cfg.get("backend")
            or bridge_health.get("configured_backend")
            or bridge_health.get("backend")
            or "vector"
        ).strip().lower()
        robot_backend = str(cfg.get("robot_behavior_memory_backend") or "robotmem").strip().lower()
        robot_enabled = bool(cfg.get("robot_behavior_memory_enabled", False))
        return {
            "product_default": "customer_knowledge_first",
            "customer_knowledge": {
                "purpose": "customer_rag_evidence",
                "backend": customer_backend,
                "active_backend": bridge_health.get("backend", ""),
                "ready": bool(bridge_health.get("available")),
                "data_scope": "routes_sop_devices_faq",
                "enters_prompt": True,
                "expiry_enforced": bool(bridge_health.get("rag_enforce_expiry", True)),
            },
            "robot_behavior_memory": {
                "purpose": "long_term_robot_behavior",
                "backend": robot_backend,
                "enabled": robot_enabled,
                "ready": bool(bridge_health.get("robotmem_ready"))
                if robot_backend == "robotmem"
                else False,
                "enters_prompt": False,
                "notes": "Keep robot behavior memory separate from customer RAG evidence.",
            },
        }

    @staticmethod
    def _memory_health_warnings(
        bridge_health: dict[str, Any],
        *,
        strategy: dict[str, Any],
        ready: bool,
        degraded: bool,
        catalog_answer_ready: bool,
    ) -> list[str]:
        warnings: list[str] = []
        if not bridge_health.get("enabled", True):
            warnings.append("memory_runtime_disabled_catalog_only")
        if not ready:
            warnings.append("memory_backend_not_ready")
        if degraded:
            warnings.append("selected_backend_not_ready_using_fallback")
        if catalog_answer_ready and not bridge_health.get("available"):
            warnings.append("customer_knowledge_catalog_only")
        customer = strategy.get("customer_knowledge", {})
        if customer.get("backend") != bridge_health.get("backend"):
            warnings.append("customer_backend_config_differs_from_active_backend")
        robot = strategy.get("robot_behavior_memory", {})
        if robot.get("enabled") and not robot.get("ready"):
            warnings.append("robot_behavior_memory_not_ready")
        if bridge_health.get("rag_enforce_expiry") is False:
            warnings.append("rag_expiry_not_enforced")
        return warnings

    @staticmethod
    def _memory_answer_contract(bridge_health: dict[str, Any]) -> dict[str, Any]:
        expiry_enforced = bool(bridge_health.get("rag_enforce_expiry", True))
        return {
            "contract_type": "askme.customer_knowledge_answer_contract.v1",
            "evidence_required": True,
            "approved_knowledge_only": True,
            "current_knowledge_only": expiry_enforced,
            "conflict_free_knowledge_only": True,
            "show_evidence_in_answer": True,
            "refuse_when_no_evidence": True,
            "refuse_when_expired": expiry_enforced,
            "refuse_when_conflicting": True,
            "robot_behavior_memory_enters_customer_prompt": False,
        }

    @staticmethod
    def _memory_customer_status(
        status: str,
        *,
        warnings: list[str],
        catalog_answer_ready: bool,
    ) -> str:
        if "rag_expiry_not_enforced" in warnings:
            return "知识过期拦截未启用，不能作为客户回答依据。"
        if status == "ready":
            return "客户知识库可用于有证据回答。"
        if status == "degraded":
            return "客户知识库可回答，但正在使用降级或备用检索。"
        if status == "catalog_only" and catalog_answer_ready:
            return "仅使用已发布知识目录回答，检索后端未完全就绪。"
        if status == "disabled":
            return "记忆检索已关闭，只能依赖已发布目录或拒答。"
        return "客户知识库未就绪，不能直接回答客户问题。"

    @staticmethod
    def _memory_customer_next_step(
        status: str,
        *,
        warnings: list[str],
        bridge_health: dict[str, Any],
        catalog_health: dict[str, Any],
    ) -> str:
        if "rag_expiry_not_enforced" in warnings:
            return "先启用知识过期拦截，再允许知识进入回答。"
        if "robot_behavior_memory_not_ready" in warnings:
            return "机器人长期行为记忆未就绪；先保持它与客户知识库隔离。"
        if status == "ready":
            return "继续维护已发布知识，并在回答气泡展示引用证据。"
        if status == "degraded":
            fallback = str(bridge_health.get("fallback_backend") or "fallback")
            return f"修复主检索后端，当前由 {fallback} 承接回答。"
        if status == "catalog_only":
            return "完成向量或 MemPalace 检索后端配置，并重建知识索引。"
        if status == "disabled":
            return "启用 memory.backend，或只允许已发布目录中的固定证据回答。"
        if int(catalog_health.get("needs_review") or 0):
            return "先审批待复核知识，只有已发布知识可以进入回答。"
        return "上传并发布可回答知识，然后重建索引。"

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
        results = evidence if isinstance(evidence, list) else []
        dropped_results = dropped if isinstance(dropped, list) else []
        fallback = {}
        if not results and not dropped_results:
            fallback = self._knowledge_catalog.search_records(
                query,
                limit=_int_or_default(payload.get("limit"), 5),
            )
            fallback_records = fallback.get("records")
            fallback_dropped = fallback.get("dropped_records")
            if isinstance(fallback_records, list) and fallback_records:
                results = fallback_records
                health = {**health, "last_backend": "catalog"}
            if isinstance(fallback_dropped, list) and fallback_dropped:
                dropped_results = [*dropped_results, *fallback_dropped]
        answer_policy = self._search_answer_policy(results, dropped_results, answer_policy)
        return {
            "query": query,
            "results": results,
            "rag": {
                "enabled": health.get("enabled", False),
                "backend": health.get("backend", ""),
                "configured_backend": health.get("configured_backend", ""),
                "backend_selection": health.get("backend_selection", {}),
                "available": health.get("available", False),
                "last_backend": health.get("last_backend", ""),
                "last_retrieve_ms": health.get("last_retrieve_ms"),
                "last_retrieved_items": health.get("last_retrieved_items", 0),
                "fallback_count": health.get("fallback_count", 0),
                "last_fallback_reason": health.get("last_fallback_reason", ""),
                "dropped_evidence": dropped_results,
                "answer_policy": answer_policy,
                "catalog_fallback": fallback if fallback else {},
            },
            "warnings": [],
        }

    @staticmethod
    def _search_answer_policy(
        results: list[Any],
        dropped_results: list[Any],
        current_policy: Any,
    ) -> dict[str, Any]:
        """Return a product-facing answer policy after bridge and catalog filtering."""

        policy = current_policy if isinstance(current_policy, dict) else {}
        if results:
            if policy.get("state") == "grounded":
                return policy
            return {
                "state": "grounded",
                "action": "answer_with_evidence",
                "reason": "eligible_evidence_found",
            }
        reasons = {
            str(item.get("drop_reason") or "")
            for item in dropped_results
            if isinstance(item, dict)
        }
        if reasons:
            if any("conflict" in reason for reason in reasons):
                if policy.get("state") == "conflict":
                    return policy
                return {
                    "state": "conflict",
                    "action": "clarify_or_escalate",
                    "reason": "conflicting_knowledge",
                    "drop_reasons": sorted(reasons),
                }
            if any(
                reason == "expired" or "evidence_version" in reason or "stale" in reason
                for reason in reasons
            ):
                if policy.get("state") == "stale":
                    return policy
                return {
                    "state": "stale",
                    "action": "refuse_and_request_update",
                    "reason": "knowledge_expired_or_outdated",
                    "drop_reasons": sorted(reasons),
                }
            if policy.get("state") == "unapproved":
                return policy
            return {
                "state": "unapproved",
                "action": "refuse_and_request_approval",
                "reason": "knowledge_not_prompt_eligible",
                "drop_reasons": sorted(reasons),
            }
        if policy:
            return policy
        return {
            "state": "no_evidence",
            "action": "clarify_or_refuse",
            "reason": "no_eligible_evidence",
        }

    async def preview_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Parse uploaded knowledge content and return records without indexing."""
        from askme.memory.retrieval.importer import parse_knowledge_text

        content = str(payload.get("content") or "")
        filename = str(payload.get("filename") or payload.get("source") or "knowledge.md")
        source = str(payload.get("source") or filename)
        owner = str(payload.get("owner") or "").strip()
        category = payload.get("category")
        governance = _knowledge_governance_fields(payload, filename=filename)
        if not content.strip():
            return {
                "source": source,
                "parsed": 0,
                "records": [],
                "errors": ["empty_content"],
                "dry_run": True,
                "document_profile": _knowledge_document_profile(filename, content),
                "category_taxonomy": knowledge_category_taxonomy_payload(),
            }
        profile = _knowledge_document_profile(filename, content)
        if not profile["supported"]:
            return {
                "source": source,
                "parsed": 0,
                "records": [],
                "errors": [profile["reason"]],
                "dry_run": True,
                "document_profile": profile,
                "category_taxonomy": knowledge_category_taxonomy_payload(),
            }
        try:
            records = parse_knowledge_text(
                content,
                filename=filename,
                source=source,
                category=str(category) if category else None,
            )
            if owner:
                records = [replace(record, owner=record.owner or owner) for record in records]
            records = [
                replace(
                    record,
                    quality_status=governance["quality_status"] or record.quality_status,
                    visibility=governance["visibility"] or record.visibility,
                    customer_id=governance["customer_id"] or record.customer_id,
                    project_id=governance["project_id"] or record.project_id,
                    product_area=governance["product_area"] or record.product_area,
                    workstream=governance["workstream"] or record.workstream,
                    linked_object_type=governance["linked_object_type"] or record.linked_object_type,
                    linked_object_id=governance["linked_object_id"] or record.linked_object_id,
                    document_type=governance["document_type"] or record.document_type,
                )
                for record in records
            ]
        except Exception as exc:
            return {
                "source": source,
                "parsed": 0,
                "records": [],
                "errors": [f"parse_error: {type(exc).__name__}: {exc}"],
                "dry_run": True,
                "document_profile": profile,
                "category_taxonomy": knowledge_category_taxonomy_payload(),
            }
        return {
            "source": source,
            "parsed": len(records),
            "records": [self._knowledge_record_payload(record) for record in records[:100]],
            "errors": [],
            "dry_run": True,
            "document_profile": profile,
            "category_taxonomy": knowledge_category_taxonomy_payload(),
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
                "document_profile": preview.get("document_profile", {}),
                "rag": self._memory_bridge.health(),
                "category_taxonomy": knowledge_category_taxonomy_payload(),
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
        cataloged = len([
            record for record in catalog_result.get("records", []) if isinstance(record, dict)
        ])
        return {
            "source": preview.get("source", ""),
            "parsed": int(preview.get("parsed", 0) or 0),
            "cataloged": cataloged,
            "indexed": imported,
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
            "document_profile": preview.get("document_profile", {}),
            "catalog": self._knowledge_catalog.health(),
            "rag": self._memory_bridge.health(),
            "category_taxonomy": knowledge_category_taxonomy_payload(),
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
            "category_taxonomy": knowledge_category_taxonomy_payload(),
            "index_jobs": self._knowledge_job_store.list_jobs(limit=_int_or_default(
                payload.get("job_limit"),
                5,
            )),
            "rag": self._memory_bridge.health(),
            "memory_health": await self.health_payload({}),
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
                "category_label": record.get("category_label", ""),
                "quality_status": record.get("quality_status", ""),
                "visibility": record.get("visibility", ""),
                "customer_id": record.get("customer_id", ""),
                "project_id": record.get("project_id", ""),
                "product_area": record.get("product_area", ""),
                "workstream": record.get("workstream", ""),
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
        category_meta = knowledge_category_metadata(record.normalized_category())
        return {
            "text": record.text,
            "memory_text": record.to_memory_text(),
            "category": record.normalized_category(),
            "category_label": category_meta["label"],
            "category_group": category_meta["group"],
            "category_description": category_meta["description"],
            "source": record.source,
            "owner": record.owner,
            "updated_at": record.updated_at,
            "expires_at": record.expires_at,
            "confidence": record.confidence,
            "approval_status": record.approval_status,
            "quality_status": record.quality_status,
            "visibility": record.visibility,
            "customer_id": record.customer_id,
            "project_id": record.project_id,
            "product_area": record.product_area,
            "workstream": record.workstream,
            "linked_object_type": record.linked_object_type,
            "linked_object_id": record.linked_object_id,
            "document_type": record.document_type,
            "metadata": metadata,
        }

    async def _sync_catalog_records(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        indexed = 0
        skipped = 0
        errors: list[str] = []
        bridge_health = self._memory_bridge.health()
        if bridge_health.get("enabled") is False:
            return {
                "indexed": 0,
                "skipped": len(records),
                "errors": [],
                "backend": bridge_health.get("backend", "disabled"),
                "reason": "memory_backend_disabled",
            }
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
                "quality_status": record.get("quality_status") or metadata.get("quality_status") or "",
                "visibility": record.get("visibility") or metadata.get("visibility") or "",
                "customer_id": record.get("customer_id") or metadata.get("customer_id") or "",
                "project_id": record.get("project_id") or metadata.get("project_id") or "",
                "product_area": record.get("product_area") or metadata.get("product_area") or "",
                "workstream": record.get("workstream") or metadata.get("workstream") or "",
                "linked_object_type": (
                    record.get("linked_object_type") or metadata.get("linked_object_type") or ""
                ),
                "linked_object_id": (
                    record.get("linked_object_id") or metadata.get("linked_object_id") or ""
                ),
                "document_type": record.get("document_type") or metadata.get("document_type") or "",
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
