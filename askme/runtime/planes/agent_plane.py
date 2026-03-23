"""Backward-compatible wrapper for :mod:`askme.runtime.planes.executive_plane`."""

from .executive_plane import build_agent_plane, build_executive_plane

__all__ = ["build_agent_plane", "build_executive_plane"]
