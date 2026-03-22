"""Backward-compat — real module at askme.memory.site_knowledge."""
import importlib as _il, sys as _sys
_mod = _il.import_module("askme.memory.site_knowledge")
_sys.modules[__name__] = _mod
