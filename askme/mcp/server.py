"""
Askme MCP Server; exposes robot control, voice I/O, and skills via MCP.

Usage::

    python -m askme.mcp.server           # stdio (for Claude Desktop/Code)
    askme-mcp                            # via pyproject.toml scripts
"""

from __future__ import annotations

import logging
import sys

from askme.mcp.context import AppContext
from askme.mcp.registration import mcp, register_mcp_modules

__all__ = ["AppContext", "main", "mcp"]

# Logging MUST go to stderr; stdout is the JSON-RPC channel in stdio mode
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# When this file is launched with ``python -m askme.mcp.server``, Python executes
# it as ``__main__``. Keep the canonical module alias for compatibility with
# external imports of ``askme.mcp.server``.
if __name__ == "__main__":
    sys.modules["askme.mcp.server"] = sys.modules[__name__]

register_mcp_modules()


def main() -> None:
    """Entry point for ``askme-mcp`` command."""
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        logger.info(
            "askme MCP server\n\n"
            "Usage:\n"
            "  askme-mcp\n"
            "  python -m askme.mcp\n"
            "  python -m askme.mcp.server\n\n"
            "Options:\n"
            "  -h, --help   Show this help without starting MCP services.\n\n"
            "Use the askme CLI for MCP transport options:\n"
            "  python -m askme mcp serve --help",
        )
        return
    mcp.run()


if __name__ == "__main__":
    main()
