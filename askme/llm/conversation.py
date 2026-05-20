"""Compatibility alias for :mod:`askme.memory.core.conversation`."""

from __future__ import annotations

import sys

from askme.memory import conversation as _conversation

sys.modules[__name__] = _conversation
