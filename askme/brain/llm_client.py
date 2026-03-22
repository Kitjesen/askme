"""Backward-compat — real module at askme.llm.client."""
import importlib as _il, sys as _sys
_mod = _il.import_module("askme.llm.client")
_sys.modules[__name__] = _mod
