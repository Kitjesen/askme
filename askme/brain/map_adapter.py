"""Backward-compat — real module at askme.memory.map_adapter."""
import importlib as _il, sys as _sys
_mod = _il.import_module("askme.memory.map_adapter")
_sys.modules[__name__] = _mod
