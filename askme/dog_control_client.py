"""Backward-compat — real module at askme.robot.control_client."""
import importlib as _il, sys as _sys
_mod = _il.import_module("askme.robot.control_client")
_sys.modules[__name__] = _mod
