"""Compatibility facade for the data bus interface.

The canonical pub/sub contract is ``askme.interfaces.bus.BusBackend``. This
module keeps the legacy ``askme.robot.telemetry.pubsub.PubSubBase`` import path
working for robot-side adapters and tests.
"""

from __future__ import annotations

from askme.interfaces.bus import BusBackend as PubSubBase

__all__ = ["PubSubBase"]
