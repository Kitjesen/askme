"""Backward-compat — real module at askme.perception.vision_bridge."""
import importlib as _il, sys as _sys
_mod = _il.import_module("askme.perception.vision_bridge")
_sys.modules[__name__] = _mod
