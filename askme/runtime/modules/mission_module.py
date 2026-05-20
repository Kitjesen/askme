"""MissionModule - safe mission draft and dry-run adapter for askme."""

from __future__ import annotations

import logging
from typing import Any

from askme.runtime.core.module import Module, ModuleRegistry
from askme.runtime.task.mission import MissionService

logger = logging.getLogger(__name__)


class MissionModule(Module):
    """Expose industrial mission drafting without direct hardware control."""

    name = "mission"
    provides = ("mission_adapter", "mission_draft", "inspection_report")

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        self.service = MissionService(cfg)
        logger.info(
            "MissionModule: built (enabled=%s, submit_enabled=%s, arbiter=%s)",
            self.service.enabled,
            self.service.submit_enabled,
            bool(self.service.base_url),
        )

    @property
    def mission_service(self) -> MissionService:
        return self.service

    def health(self) -> dict[str, Any]:
        return self.service.health()

    def capabilities(self) -> dict[str, Any]:
        return self.service.capabilities()
