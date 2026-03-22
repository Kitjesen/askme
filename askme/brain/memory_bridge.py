"""Backward-compat — real module at askme.memory.bridge."""
import importlib as _il, sys as _sys
_mod = _il.import_module("askme.memory.bridge")
_sys.modules[__name__] = _mod
