"""Backward-compatible wrapper for :mod:`askme.runtime.planes.platform_plane`."""

from .platform_plane import build_platform_plane, build_robot_plane

__all__ = ["build_platform_plane", "build_robot_plane"]
