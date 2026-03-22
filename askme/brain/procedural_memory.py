"""Backward-compat — real module at askme.memory.procedural."""
import importlib as _il, sys as _sys
_mod = _il.import_module("askme.memory.procedural")
_sys.modules[__name__] = _mod
