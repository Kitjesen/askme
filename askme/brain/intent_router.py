"""Backward-compat — real module at askme.llm.intent_router."""
import importlib as _il, sys as _sys
_mod = _il.import_module("askme.llm.intent_router")
_sys.modules[__name__] = _mod
