"""HealthModule — wraps AskmeHealthServer as a declarative module.

Canonical wiring::

    health_server = AskmeHealthServer(
        cfg.get("health_server", {}),
        snapshot_provider=runtime.health_snapshot,
        metrics_provider=runtime.metrics_snapshot,
    )
"""

from __future__ import annotations

import logging
from typing import Any

from askme.runtime.module import Module, ModuleRegistry

logger = logging.getLogger(__name__)


class HealthModule(Module):
    """Provides the AskmeHealthServer to the runtime."""

    name = "health"
    depends_on = ("memory", "pipeline", "skill", "text", "mission")
    provides = ("health_http", "http_chat", "capabilities", "missions")

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        from askme.health_server import AskmeHealthServer

        health_cfg = cfg.get("health_server", {})

        # Collect health from all registered modules in the runtime.
        def _runtime_health_provider() -> dict[str, Any]:
            result: dict[str, Any] = {"status": "ok", "service": "askme"}
            for mod_name, mod in registry.items():
                try:
                    result[mod_name] = mod.health()
                except Exception:
                    result[mod_name] = {"status": "error"}
            return result

        self.server = AskmeHealthServer(
            health_cfg,
            snapshot_provider=_runtime_health_provider,
        )
        self._wire_runtime_handlers(cfg, registry)

        logger.info(
            "HealthModule: built (enabled=%s, port=%d)",
            self.server.enabled,
            self.server.port,
        )

    def _wire_runtime_handlers(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        """Connect HTTP surfaces to built runtime modules when they exist."""
        chat_handler = self._chat_handler(registry)
        if chat_handler is not None and hasattr(self.server, "set_chat_handler"):
            self.server.set_chat_handler(chat_handler)

        if hasattr(self.server, "set_capabilities_provider"):
            self.server.set_capabilities_provider(
                lambda: self._capabilities_snapshot(cfg, registry)
            )

        if hasattr(self.server, "set_conversation_provider"):
            self.server.set_conversation_provider(
                lambda: self._conversation_snapshot(registry)
            )

        mission_handler = self._mission_handler(registry)
        if mission_handler is not None and hasattr(self.server, "set_mission_handler"):
            self.server.set_mission_handler(mission_handler)

    def _chat_handler(self, registry: ModuleRegistry) -> Any | None:
        text_mod = registry.get("text")
        text_loop = getattr(text_mod, "text_loop", None) if text_mod else None
        process_turn = getattr(text_loop, "process_turn", None)
        if callable(process_turn):
            return process_turn

        pipeline_mod = registry.get("pipeline")
        pipeline = getattr(pipeline_mod, "brain_pipeline", None) if pipeline_mod else None
        process = getattr(pipeline, "process", None)
        if callable(process):
            return process
        return None

    def _conversation_snapshot(self, registry: ModuleRegistry) -> list[dict[str, Any]]:
        conversation = self._conversation(registry)
        history = getattr(conversation, "history", None)
        if not isinstance(history, list):
            return []
        return [dict(msg) for msg in history if isinstance(msg, dict)]

    def _conversation(self, registry: ModuleRegistry) -> Any | None:
        mem_mod = registry.get("memory")
        conversation = getattr(mem_mod, "conversation", None) if mem_mod else None
        if conversation is not None:
            return conversation

        pipeline_mod = registry.get("pipeline")
        pipeline = getattr(pipeline_mod, "brain_pipeline", None) if pipeline_mod else None
        return getattr(pipeline, "_conversation", None)

    def _mission_handler(self, registry: ModuleRegistry) -> Any | None:
        mission_mod = registry.get("mission")
        if mission_mod is None:
            return None
        return getattr(mission_mod, "mission_service", None)

    def _capabilities_snapshot(
        self,
        cfg: dict[str, Any],
        registry: ModuleRegistry,
    ) -> dict[str, Any]:
        from askme import __version__ as ASKME_VERSION

        profile = self._runtime_profile(registry)
        skill_mod = registry.get("skill")
        skill_manager = getattr(skill_mod, "skill_manager", None) if skill_mod else None
        contracts = skill_manager.get_contracts() if skill_manager else []
        openapi_doc = (
            skill_manager.openapi_document()
            if skill_manager
            else {"info": {"title": "", "version": ""}, "paths": {}}
        )

        components: dict[str, dict[str, Any]] = {}
        for name, mod in registry.items():
            try:
                health = mod.health()
            except Exception:
                health = {"status": "error"}
            try:
                capabilities = mod.capabilities()
            except Exception:
                capabilities = {}
            components[name] = {
                "health": health,
                "capabilities": capabilities,
            }

        return {
            "app": {
                "name": cfg.get("app", {}).get("name", "askme"),
                "version": cfg.get("app", {}).get("version") or ASKME_VERSION,
                "voice_mode": profile.voice_io,
                "robot_mode": profile.robot_api,
            },
            "profile": profile.snapshot(),
            "components": components,
            "mission_adapter": components.get("mission", {}).get("capabilities", {}),
            "skills": {
                "count": len(skill_manager.get_all()) if skill_manager else 0,
                "enabled_count": len(skill_manager.get_enabled()) if skill_manager else 0,
                "contract_count": len(contracts),
                "code_contract_count": sum(
                    1 for contract in contracts
                    if getattr(contract, "source", None) == "code"
                ),
                "legacy_contract_count": sum(
                    1 for contract in contracts
                    if getattr(contract, "source", None) != "code"
                ),
                "catalog": (
                    skill_manager.get_contract_catalog()
                    if skill_manager else []
                ),
            },
            "openapi": {
                "title": openapi_doc.get("info", {}).get("title", ""),
                "version": openapi_doc.get("info", {}).get("version", ""),
                "path_count": len(openapi_doc.get("paths", {})),
            },
        }

    def _runtime_profile(self, registry: ModuleRegistry) -> Any:
        from askme.runtime.profiles import MCP_PROFILE, legacy_profile_for

        has_voice = "voice" in registry
        has_text = "text" in registry
        has_robot = any(
            name in registry
            for name in ("control", "executor", "led", "perception", "safety")
        )
        if has_voice and has_robot and not has_text:
            return MCP_PROFILE
        return legacy_profile_for(voice_mode=has_voice, robot_mode=has_robot)

    async def start(self) -> None:
        if self.server.enabled:
            await self.server.start()

    async def stop(self) -> None:
        if self.server.enabled:
            await self.server.stop()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "enabled": self.server.enabled,
            "port": self.server.port,
        }
