"""Compatibility facade for backend registries.

The canonical registry implementation lives in
``askme.interfaces.core.registry``.
This module keeps historical ``askme.runtime.core.registry`` and
``askme.runtime.registry`` imports working.
"""

from __future__ import annotations

from askme.interfaces.core.registry import BackendRegistry

__all__ = ["BackendRegistry"]
