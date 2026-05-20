"""Neutral product-facing entry point for autonomous agent execution."""

from __future__ import annotations

from .thunder_agent_shell import ThunderAgentShell


AgentShell = ThunderAgentShell

__all__ = ["AgentShell"]
