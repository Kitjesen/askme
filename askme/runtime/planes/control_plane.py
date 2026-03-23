"""Backward-compatible wrapper for :mod:`askme.runtime.planes.diagnostics_plane`."""

from .diagnostics_plane import build_control_plane, build_diagnostics_plane

__all__ = ["build_control_plane", "build_diagnostics_plane"]
