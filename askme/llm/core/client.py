"""Backward-compatible LLM client.

``LLMClient`` is kept as the public compatibility class used by existing
runtime modules and tests.  New product code should prefer ``LLMGateway`` or
the ``LLMBackend`` contract so provider details stay behind one boundary.
"""

from __future__ import annotations

from collections.abc import Callable

from askme.config import get_config
from askme.llm.core.config import LLMConfig
from askme.llm.core.factory import create_llm_provider
from askme.llm.core.gateway import LLMGateway
from askme.llm.streaming.retry import default_backoff as _backoff
from askme.telemetry.ota_bridge import OTABridgeMetrics


class LLMClient(LLMGateway):
    """Compatibility wrapper around :class:`askme.llm.core.gateway.LLMGateway`.

    Preferred construction:

        cfg = LLMConfig.from_cfg(brain_section)
        client = LLMClient(llm_config=cfg, metrics=ota_metrics)

    Legacy construction still works and reads the ``brain`` section from
    ``config.yaml``:

        client = LLMClient(api_key="...", model="MiniMax-M2.7-highspeed")
    """

    def __init__(
        self,
        *,
        llm_config: LLMConfig | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        metrics: OTABridgeMetrics | None = None,
    ) -> None:
        resolved_config = llm_config or _legacy_config(api_key=api_key, base_url=base_url, model=model)
        provider = create_llm_provider(resolved_config, backoff_func=_dynamic_backoff())
        super().__init__(llm_config=resolved_config, metrics=metrics, provider=provider)


def _legacy_config(
    *,
    api_key: str | None,
    base_url: str | None,
    model: str | None,
) -> LLMConfig:
    cfg = get_config().get("brain", {})
    return LLMConfig(
        provider=cfg.get("provider", cfg.get("backend", "")),
        api_key=api_key or cfg.get("api_key", ""),
        base_url=base_url or cfg.get("base_url", "https://api.minimax.chat/v1"),
        model=model or cfg.get("model", "MiniMax-M2.7-highspeed"),
        max_tokens=cfg.get("max_tokens", 0),
        temperature=cfg.get("temperature", 0.7),
        timeout=cfg.get("timeout", 30.0),
        max_retries=cfg.get("max_retries", 2),
        fallback_models=cfg.get("fallback_models", []),
        minimax_api_key=cfg.get("minimax_api_key", ""),
        minimax_base_url=cfg.get("minimax_base_url", "https://api.minimax.chat/v1"),
        provider_options=cfg.get("provider_options", {}),
    )


def _dynamic_backoff() -> Callable[[int], float]:
    """Return a backoff function that respects monkeypatching of _backoff."""

    return lambda attempt: _backoff(attempt)
