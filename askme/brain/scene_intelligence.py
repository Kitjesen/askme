"""Backward-compat — real module at askme.perception.scene_intelligence."""
import importlib as _il, sys as _sys
_mod = _il.import_module("askme.perception.scene_intelligence")
_sys.modules[__name__] = _mod
