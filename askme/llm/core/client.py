"""Backward-compatible LLM client.

``LLMClient`` is kept as the public compatibility class used by existing
runtime modules and tests.  New product code should prefer ``LLMGateway`` or
the ``LLMBackend`` contract so provider details stay behind one boundary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from askme.config import get_config
from askme.llm.core.config import LLMConfig
from askme.llm.core.factory import create_llm_provider
from askme.llm.core.gateway import LLMGateway
from askme.llm.providers.profiles import normalize_provider_name
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
        resolved_config = llm_config or _legacy_config(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        validation_errors = resolved_config.validate()
        if normalize_provider_name(resolved_config.provider) == "fake":
            validation_errors = [
                error
                for error in validation_errors
                if "api_key" not in error and "base_url" not in error
            ]
        if validation_errors:
            raise ValueError("Invalid LLM configuration: " + "; ".join(validation_errors))
        provider = create_llm_provider(resolved_config, backoff_func=_dynamic_backoff())
        super().__init__(llm_config=resolved_config, metrics=metrics, provider=provider)


def _legacy_config(
    *,
    api_key: str | None,
    base_url: str | None,
    model: str | None,
) -> LLMConfig:
    cfg = get_config().get("brain", {})
    resolved = LLMConfig.from_cfg(cfg)
    if api_key is not None:
        resolved = replace(resolved, api_key=api_key)
    if base_url is not None:
        resolved = replace(resolved, base_url=base_url)
    if model is not None:
        resolved = replace(resolved, model=model)
    return resolved


def _dynamic_backoff() -> Callable[[int], float]:
    """Return a backoff function that respects monkeypatching of _backoff."""

    return lambda attempt: _backoff(attempt)
