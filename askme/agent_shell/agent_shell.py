"""DEPRECATED — neutral product-facing entry point (replaced by ZeroClaw MCP Agent)."""

from __future__ import annotations

from .thunder_agent_shell import ThunderAgentShell

AgentShell = ThunderAgentShell

__all__ = ["AgentShell"]
