"""LLM configuration.

Separates config values from where they came from.  Runtime modules should read
``config.yaml`` once, build this dataclass, and pass it into the LLM gateway.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """All configuration needed to construct an LLM gateway/provider."""

    provider: str = ""
    api_key: str = ""
    base_url: str = "https://api.minimaxi.com/v1"
    model: str = "MiniMax-M2.7-highspeed"
    max_tokens: int = 0
    temperature: float = 0.7
    timeout: float = 30.0
    max_retries: int = 2
    fallback_models: list[str] = field(default_factory=list)

    # Optional secondary MiniMax client for mixed relay + direct MiniMax setup.
    minimax_api_key: str = ""
    minimax_base_url: str = "https://api.minimaxi.com/v1"
    provider_options: dict = field(default_factory=dict)

    @staticmethod
    def _first(cfg: dict, *keys: str, default: str = "") -> str:
        for key in keys:
            value = cfg.get(key, "")
            if str(value).strip():
                return str(value)
        return default

    def validate(self) -> list[str]:
        """Return validation errors. Empty list means valid enough to start."""

        errors: list[str] = []
        if not self.api_key:
            errors.append("LLMConfig.api_key is empty; LLM calls will fail")
        if not self.model:
            errors.append("LLMConfig.model is empty")
        if self.temperature < 0.0 or self.temperature > 2.0:
            errors.append(f"LLMConfig.temperature={self.temperature} is outside [0, 2]")
        if self.timeout <= 0:
            errors.append(f"LLMConfig.timeout={self.timeout} must be positive")
        if self.max_retries < 0:
            errors.append(f"LLMConfig.max_retries={self.max_retries} must be >= 0")
        if not self.base_url:
            errors.append("LLMConfig.base_url is empty")
        return errors

    def validate_and_warn(self) -> bool:
        """Validate and log warnings for each error."""

        errors = self.validate()
        for err in errors:
            logger.warning("[LLMConfig] %s", err)
        return len(errors) == 0

    @classmethod
    def from_cfg(cls, brain_cfg: dict) -> LLMConfig:
        """Construct from the ``brain`` sub-dict of config.yaml."""

        return cls(
            provider=brain_cfg.get("provider", brain_cfg.get("backend", "")),
            api_key=cls._first(
                brain_cfg,
                "api_key",
                "llm_api_key",
                "LLM_API_KEY",
                default="",
            ),
            base_url=cls._first(
                brain_cfg,
                "base_url",
                "llm_base_url",
                "LLM_BASE_URL",
                default="https://api.minimaxi.com/v1",
            ),
            model=brain_cfg.get("model", "MiniMax-M2.7-highspeed"),
            max_tokens=brain_cfg.get("max_tokens", 0),
            temperature=brain_cfg.get("temperature", 0.7),
            timeout=brain_cfg.get("timeout", 30.0),
            max_retries=brain_cfg.get("max_retries", 2),
            fallback_models=brain_cfg.get("fallback_models", []),
            minimax_api_key=brain_cfg.get("minimax_api_key", ""),
            minimax_base_url=brain_cfg.get("minimax_base_url", "https://api.minimaxi.com/v1"),
            provider_options=brain_cfg.get("provider_options", {}),
        )
