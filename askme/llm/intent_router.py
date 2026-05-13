"""Compatibility alias for :mod:`askme.interaction.intent_router`."""

from __future__ import annotations

import sys

from askme.interaction import intent_router as _intent_router

sys.modules[__name__] = _intent_router
