"""Compatibility alias for :mod:`askme.cognition.memory.working_memory`."""

from __future__ import annotations

import sys

from askme.cognition.memory import working_memory as _working_memory

sys.modules[__name__] = _working_memory
