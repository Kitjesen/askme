"""Compatibility alias for :mod:`askme.cognition.perception.active_perception`."""

from __future__ import annotations

import sys

from askme.cognition.perception import active_perception as _active_perception

sys.modules[__name__] = _active_perception
