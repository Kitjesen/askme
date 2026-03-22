"""Backward-compat — real module at askme.robot.led_controller."""
import importlib as _il, sys as _sys
_mod = _il.import_module("askme.robot.led_controller")
_sys.modules[__name__] = _mod
