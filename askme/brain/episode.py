"""Backward-compat — real module at askme.memory.episode."""
import importlib as _il, sys as _sys
_mod = _il.import_module("askme.memory.episode")
_sys.modules[__name__] = _mod
