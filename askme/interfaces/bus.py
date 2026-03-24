"""Data bus backend interface."""

from __future__ import annotations

from askme.robot.pubsub import PubSubBase
from askme.runtime.registry import BackendRegistry


class BusBackend(PubSubBase):
    """Abstract data bus — subscribe/publish typed topics.

    Extends :class:`~askme.robot.pubsub.PubSubBase` which already defines
    the abstract methods: ``start``, ``stop``, ``on``, ``get_latest``,
    ``publish``, ``connected``, and ``health``.

    Implementations: Pulse (CycloneDDS), MockPulse (memory), future ZeroMQ/LCM.
    """


bus_registry = BackendRegistry("bus", BusBackend, default="pulse")
