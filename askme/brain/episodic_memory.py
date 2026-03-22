"""Backward-compat — real module at askme.memory.episodic_memory."""
import importlib as _il, sys as _sys
_mod = _il.import_module("askme.memory.episodic_memory")
_sys.modules[__name__] = _mod
