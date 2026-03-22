"""Backward-compat — real module at askme.llm.conversation."""
import importlib as _il, sys as _sys
_mod = _il.import_module("askme.llm.conversation")
_sys.modules[__name__] = _mod
