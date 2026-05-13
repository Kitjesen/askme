"""MCP resources for askme product I/O contracts."""

from __future__ import annotations

import json

from askme.contracts.catalog import contract_catalog, contract_examples
from askme.mcp.server import mcp


@mcp.resource("askme://contracts/io")
def contracts_io() -> str:
    """Product I/O contracts for perception, intent, action, and UI output."""
    return json.dumps(contract_catalog(), ensure_ascii=False)


@mcp.resource("askme://contracts/examples")
def contracts_examples() -> str:
    """Example payloads for external agents and integration tests."""
    return json.dumps(contract_examples(), ensure_ascii=False)
