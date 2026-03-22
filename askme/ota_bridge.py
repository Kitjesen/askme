"""Backward-compat — real module at askme.robot.ota_bridge."""
import importlib as _il, sys as _sys
_mod = _il.import_module("askme.robot.ota_bridge")
_sys.modules[__name__] = _mod
