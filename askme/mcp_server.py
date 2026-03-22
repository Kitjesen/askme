"""Backward-compat — real module at askme.mcp.server."""
import importlib as _il, sys as _sys
_mod = _il.import_module("askme.mcp.server")
_sys.modules[__name__] = _mod
