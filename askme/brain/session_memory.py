"""Backward-compat — real module at askme.memory.session."""
import importlib as _il, sys as _sys
_mod = _il.import_module("askme.memory.session")
_sys.modules[__name__] = _mod
