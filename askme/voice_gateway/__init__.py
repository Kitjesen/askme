"""Voice gateway layer public API.

This package owns the unified voice runtime boundary. It hides whether a turn is
served by the local pipeline, an upstream edge service, or a later provider.

New code should construct provider-owned runtime bridge implementations through
``askme.providers.build_voice_runtime_bridge`` and pass them into
``VoiceGatewayService``. The package-root ``VoiceRuntimeBridge`` name remains
available only for legacy imports.
"""

from __future__ import annotations

from askme.voice_gateway.service import VoiceGatewayService
from askme.voice_gateway.session import (
    ConversationSession,
    ConversationSessionManager,
    ConversationTurn,
    InMemorySessionStore,
    SessionSnapshot,
)

__all__ = [
    "ConversationSession",
    "ConversationSessionManager",
    "ConversationTurn",
    "InMemorySessionStore",
    "SessionSnapshot",
    "VoiceGatewayService",
]


def __getattr__(name: str) -> object:
    if name == "VoiceRuntimeBridge":
        from askme.voice_gateway.runtime_bridge import VoiceRuntimeBridge

        return VoiceRuntimeBridge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
