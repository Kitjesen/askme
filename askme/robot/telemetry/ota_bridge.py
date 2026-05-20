"""Compatibility facade for OTA telemetry helpers.

The canonical module is ``askme.telemetry.ota_bridge``. This path remains for
legacy imports such as ``askme.robot.ota_bridge``, including historical
underscore-prefixed test helpers.
"""

from __future__ import annotations

from askme.telemetry import ota_bridge as _canonical

for _name in dir(_canonical):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_canonical, _name)

__all__ = [
    _name
    for _name in globals()
    if not (_name.startswith("__") and _name.endswith("__"))
]
