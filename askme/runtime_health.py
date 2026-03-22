"""Backward-compat — real module at askme.robot.runtime_health."""
import importlib as _il, sys as _sys
_mod = _il.import_module("askme.robot.runtime_health")
_sys.modules[__name__] = _mod
