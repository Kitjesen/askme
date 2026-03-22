"""Backward-compat — real module at askme.memory.system."""
import importlib as _il, sys as _sys
_mod = _il.import_module("askme.memory.system")
_sys.modules[__name__] = _mod
