"""Model selection and provider-specific request policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelPolicy:
    """Choose models and request extensions for each LLM turn."""

    primary_model: str
    fallback_models: list[str] = field(default_factory=list)

    def model_chain(self, override: str | None = None) -> list[str]:
        """Return primary/override followed by compatible fallbacks.

        MiniMax and non-MiniMax models are not mixed in one fallback chain.  In
        voice scenarios, trying a different provider after the selected one is
        rate-limited often adds seconds of latency and creates inconsistent
        behavior.
        """

        primary = override or self.primary_model
        chain = [primary]
        primary_is_minimax = _is_minimax_family(primary)
        for fallback in self.fallback_models:
            if fallback == primary:
                continue
            if _is_minimax_family(fallback) == primary_is_minimax:
                chain.append(fallback)
        return chain

    @staticmethod
    def extra_body_for_model(model: str, *, thinking: bool) -> dict[str, Any] | None:
        """Return provider-specific request body extensions."""

        if _is_deepseek_v4(model):
            return {"thinking": {"type": "enabled" if thinking else "disabled"}}
        if thinking:
            return None
        if _is_minimax_family(model):
            return {"reasoning_split": True}
        return None


def _is_minimax_family(model: str) -> bool:
    return str(model or "").lower().startswith("minimax")


def _is_deepseek_v4(model: str) -> bool:
    return str(model or "").lower().startswith("deepseek-v4-")
