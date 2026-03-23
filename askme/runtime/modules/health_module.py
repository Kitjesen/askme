"""HealthModule — wraps AskmeHealthServer as a declarative module.

Mirrors the health server creation from ``assembly.py`` lines 660-694::

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


def _noop_health_provider() -> dict[str, Any]:
    """Fallback health provider until the real one is wired."""
    return {"status": "ok", "service": "askme"}


class HealthModule(Module):
    """Provides the AskmeHealthServer to the runtime."""

    name = "health"
    provides = ("health_http", "http_chat", "capabilities")

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        from askme.health_server import AskmeHealthServer

        health_cfg = cfg.get("health_server", {})

        # Build with a no-op provider; real providers are set post-build
        self.server = AskmeHealthServer(
            health_cfg,
            snapshot_provider=_noop_health_provider,
        )

        logger.info(
            "HealthModule: built (enabled=%s, port=%d)",
            self.server.enabled,
            self.server.port,
        )

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
