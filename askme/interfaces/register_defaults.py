"""Compatibility entry point for default backend registrations.

Import this module once at startup to populate the registries. Concrete
provider-backed implementations are registered by ``askme.providers``; product
reaction engines are registered by ``askme.pipeline.reactions``. This file stays
as the stable startup import used by legacy entry points.

Usage::

    import askme.interfaces.register_defaults  # noqa: F401
"""

from __future__ import annotations

from askme.pipeline.reactions.register_defaults import register_default_reactions
from askme.providers.register_defaults import register_default_provider_backends

register_default_provider_backends()
register_default_reactions()
