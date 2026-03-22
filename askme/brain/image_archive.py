"""Backward-compat — real module at askme.perception.image_archive."""
import importlib as _il, sys as _sys
_mod = _il.import_module("askme.perception.image_archive")
_sys.modules[__name__] = _mod
