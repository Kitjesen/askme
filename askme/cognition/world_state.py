"""Compatibility alias for :mod:`askme.cognition.world.world_state`."""

from __future__ import annotations

import sys

from askme.cognition.world import world_state as _world_state

sys.modules[__name__] = _world_state
