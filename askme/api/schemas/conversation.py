"""Conversation, chat, and voice-turn API response contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatResponse(BaseModel):
    """Product chat response with evidence, voice metadata, and optional runtime data."""

    model_config = ConfigDict(extra="allow")

    reply: Any = ""
    text: str = ""
    spoken: bool | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    voice_turn: dict[str, Any] | None = None
    rag: dict[str, Any] = Field(default_factory=dict)
    cognition: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    chat_backend: dict[str, Any] = Field(default_factory=dict)
    answer_policy: dict[str, Any] = Field(default_factory=dict)
    scenario_preview: dict[str, Any] = Field(default_factory=dict)
    space_resolution: dict[str, Any] = Field(default_factory=dict)


class ConversationDiagnosticsResponse(BaseModel):
    """Non-sensitive chat execution diagnostics for operations dashboards."""

    model_config = ConfigDict(extra="allow")

    chat: dict[str, Any] = Field(default_factory=dict)


class RuntimeVoiceTurnResponse(BaseModel):
    """Runtime voice-turn routing response."""

    model_config = ConfigDict(extra="allow")

    handled: bool | None = None
    reason: str = ""
    voice_turn: dict[str, Any] = Field(default_factory=dict)
    reply: Any = ""
    runtime: dict[str, Any] = Field(default_factory=dict)


class ConversationHistoryResponse(BaseModel):
    """Conversation history used by monitor and dashboard pages."""

    model_config = ConfigDict(extra="allow")

    messages: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    error: str | None = None
