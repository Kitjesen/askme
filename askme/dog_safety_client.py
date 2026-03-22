"""Backward-compat — real module at askme.robot.safety_client."""
import importlib as _il, sys as _sys
_mod = _il.import_module("askme.robot.safety_client")
_sys.modules[__name__] = _mod
