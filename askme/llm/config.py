"""LLMConfig — pure data class carrying all LLMClient configuration.

Separates *what* (config values) from *where* (config.yaml / env vars).
``LLMModule.build()`` reads config.yaml and constructs an ``LLMConfig``.
``LLMClient`` only accepts ``LLMConfig`` — it never reads config.yaml itself.

This inversion makes LLMClient trivially testable: pass an LLMConfig with
known values and the client behaves deterministically regardless of the
environment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """All configuration needed to construct an LLMClient.

    Fields mirror the ``brain:`` section of config.yaml.

    Example::

        cfg = LLMConfig(
            api_key="sk-...",
            base_url="https://api.minimax.chat/v1",
            model="MiniMax-M2.7-highspeed",
        )
        client = LLMClient(llm_config=cfg)
    """

    api_key: str = ""
    base_url: str = "https://api.minimax.chat/v1"
    model: str = "MiniMax-M2.7-highspeed"
    max_tokens: int = 0
    temperature: float = 0.7
    timeout: float = 30.0
    max_retries: int = 2
    fallback_models: list[str] = field(default_factory=list)

    # Optional secondary MiniMax client (enabled when minimax_api_key is set)
    minimax_api_key: str = ""
    minimax_base_url: str = "https://api.minimax.chat/v1"

    def validate(self) -> list[str]:
        """Return a list of validation error messages (empty = valid).

        Call at startup (e.g. in LLMModule.build()) to surface misconfiguration
        early rather than at the first user turn.
        """
        errors: list[str] = []
        if not self.api_key:
            errors.append("LLMConfig.api_key is empty — LLM calls will fail")
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
        """Validate and log warnings for each error. Returns True if valid."""
        errors = self.validate()
        for err in errors:
            logger.warning("[LLMConfig] %s", err)
        return len(errors) == 0

    @classmethod
    def from_cfg(cls, brain_cfg: dict) -> LLMConfig:
        """Construct from the ``brain`` sub-dict of config.yaml.

        Intended for use in ``LLMModule.build()`` only.
        """
        return cls(
            api_key=brain_cfg.get("api_key", ""),
            base_url=brain_cfg.get("base_url", "https://api.minimax.chat/v1"),
            model=brain_cfg.get("model", "MiniMax-M2.7-highspeed"),
            max_tokens=brain_cfg.get("max_tokens", 0),
            temperature=brain_cfg.get("temperature", 0.7),
            timeout=brain_cfg.get("timeout", 30.0),
            max_retries=brain_cfg.get("max_retries", 2),
            fallback_models=brain_cfg.get("fallback_models", []),
            minimax_api_key=brain_cfg.get("minimax_api_key", ""),
            minimax_base_url=brain_cfg.get("minimax_base_url", "https://api.minimax.chat/v1"),
        )
