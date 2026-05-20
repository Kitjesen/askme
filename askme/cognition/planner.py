"""Compatibility alias for :mod:`askme.cognition.planning.planner`."""

from __future__ import annotations

import sys

from askme.cognition.planning import planner as _planner

sys.modules[__name__] = _planner
