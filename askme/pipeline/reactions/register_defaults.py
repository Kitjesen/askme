"""Default reaction-engine registrations for product orchestration."""

from __future__ import annotations


def register_default_reactions() -> None:
    """Register product-layer reaction implementations."""

    from askme.interfaces.reaction import reaction_registry
    from askme.pipeline.reactions.reaction_engine import (
        HybridReaction,
        RuleBasedReaction,
    )

    reaction_registry.register("hybrid")(HybridReaction)
    reaction_registry.register("rules")(RuleBasedReaction)


__all__ = ["register_default_reactions"]
