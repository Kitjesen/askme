"""LLMModule — wraps LLMClient construction as a declarative module.

Mirrors the LLMClient creation logic from ``assembly.py`` lines 414-415::

    ota_metrics = OTABridgeMetrics()
    llm = LLMClient(metrics=ota_metrics)
"""

from __future__ import annotations

import logging
from typing import Any

from askme.runtime.module import Module, ModuleRegistry, Out
from askme.llm.client import LLMClient  # TODO: migrate to llm_registry.create(cfg) when LLMClient inherits LLMBackend
from askme.robot.ota_bridge import OTABridgeMetrics

logger = logging.getLogger(__name__)


class LLMModule(Module):
    """Provides LLMClient and OTABridgeMetrics to the runtime."""

    name = "llm"
    provides = ("llm",)

    llm_client: Out[LLMClient]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        self.ota_metrics = OTABridgeMetrics()
        self.client = LLMClient(metrics=self.ota_metrics)
        logger.info("LLMModule: built (model=%s)", self.client.model)

    # -- typed accessors ------------------------------------------------
    @property
    def llm_client(self) -> LLMClient:  # type: ignore[override]
        """The LLM client instance."""
        return self.client

    @property
    def metrics(self) -> Any:
        """OTA bridge metrics collector."""
        return self.ota_metrics

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "model": self.client.model}
