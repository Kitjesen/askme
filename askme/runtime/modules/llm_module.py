"""Runtime module for the product LLM gateway."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from askme.llm.core.client import LLMClient
from askme.llm.core.config import LLMConfig
from askme.runtime.core.module import Module, ModuleRegistry, Out
from askme.telemetry.ota_bridge import OTABridgeMetrics

logger = logging.getLogger(__name__)


class LLMModule(Module):
    """Provide a configured LLM gateway with background warmup.

    This is the runtime boundary that reads the ``brain`` section from
    config.yaml.  Downstream modules receive a configured LLM object and should
    not construct provider SDK clients directly.
    """

    name = "llm"
    provides = ("llm",)

    llm_client: Out[LLMClient]
    llm_config_out: Out[LLMConfig]

    def build(self, cfg: dict[str, Any], registry: ModuleRegistry) -> None:
        self.ota_metrics = OTABridgeMetrics()
        self._llm_config = LLMConfig.from_cfg(cfg.get("brain", {}))
        self._llm_config.validate_and_warn()
        self.client = LLMClient(llm_config=self._llm_config, metrics=self.ota_metrics)
        self._switch_lock = threading.RLock()
        self._warmup_task: asyncio.Task | None = None
        logger.info("LLMModule: built (model=%s)", self.client.model)

    async def start(self) -> None:
        """Fire a background warmup request to pre-heat the LLM connection."""

        self._warmup_task = asyncio.create_task(self._warmup())

    async def stop(self) -> None:
        if self._warmup_task and not self._warmup_task.done():
            self._warmup_task.cancel()

    async def _warmup(self) -> None:
        """Silent background request to warm up API connection and model cache."""

        try:
            warmup_messages = [
                {"role": "system", "content": "Reply with one Chinese character."},
                {"role": "user", "content": "好"},
            ]
            t0 = asyncio.get_running_loop().time()
            async for _ in self.client.chat_stream(warmup_messages):
                break
            elapsed = (asyncio.get_running_loop().time() - t0) * 1000
            logger.info("LLM warmup: %.0fms (connection pre-heated)", elapsed)
        except Exception as e:
            logger.debug("LLM warmup failed (non-critical): %s", e)

    @property
    def llm_client(self) -> LLMClient:  # type: ignore[override]
        return self.client

    @property
    def llm_config_out(self) -> LLMConfig:  # type: ignore[override]
        """Expose the resolved LLMConfig so downstream modules can read it."""

        return self._llm_config

    @property
    def metrics(self) -> Any:
        return self.ota_metrics

    def replace_config(self, brain_cfg: dict[str, Any]) -> LLMClient:
        """Atomically route subsequent requests to a newly configured gateway."""

        next_client = self.prepare_client(brain_cfg)
        self.commit_client(next_client)
        return next_client

    def prepare_client(self, brain_cfg: dict[str, Any]) -> LLMClient:
        """Construct and validate a candidate without changing live routing."""

        next_config = LLMConfig.from_cfg(brain_cfg)
        errors = next_config.validate()
        if errors:
            raise ValueError("; ".join(errors))
        return LLMClient(llm_config=next_config, metrics=self.ota_metrics)

    def commit_client(self, next_client: LLMClient) -> None:
        """Publish a prepared client for subsequent requests."""

        with self._switch_lock:
            self._llm_config = next_client.config
            self.client = next_client
        logger.info(
            "LLMModule: hot switched provider=%s model=%s",
            next_client.provider_name,
            next_client.model,
        )

    async def validate_client(self, client: LLMClient, *, timeout_s: float = 10.0) -> None:
        """Run a minimal provider probe before committing a requested switch."""

        async def _probe() -> None:
            async for _ in client.chat_stream(
                [{"role": "user", "content": "只回复好"}],
                max_tokens=2,
                temperature=0.0,
            ):
                return

        await asyncio.wait_for(_probe(), timeout=max(1.0, float(timeout_s)))

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "provider": getattr(
                self.client,
                "provider_name",
                getattr(self._llm_config, "provider", "unknown"),
            ),
            "model": getattr(
                self.client,
                "model",
                getattr(self._llm_config, "model", "unknown"),
            ),
        }
