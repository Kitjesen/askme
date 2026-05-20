"""Compatibility alias for :mod:`askme.cognition.perception.perception_sync`."""

from __future__ import annotations

import sys

from askme.cognition.perception import perception_sync as _perception_sync

sys.modules[__name__] = _perception_sync
