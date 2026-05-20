"""Compatibility alias for :mod:`askme.cognition.planning.planning_session`."""

from __future__ import annotations

import sys

from askme.cognition.planning import planning_session as _planning_session

sys.modules[__name__] = _planning_session
