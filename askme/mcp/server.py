"""
Askme MCP Server; exposes robot control, voice I/O, and skills via MCP.

Usage::

    python -m askme.mcp.server           # stdio (for Claude Desktop/Code)
    askme-mcp                            # via pyproject.toml scripts
"""

from __future__ import annotations

import logging
import sys

_HELP_FLAGS = frozenset({"-h", "--help"})
_HELP_TEXT = """askme MCP server

Usage:
  askme-mcp
  python -m askme.mcp
  python -m askme.mcp.server

Options:
  -h, --help   Show this help without starting MCP services.

Use the askme CLI for MCP transport options:
  python -m askme mcp serve --help"""


def _help_requested(argv: list[str]) -> bool:
    return any(arg in _HELP_FLAGS for arg in argv)


# ``python -m askme.mcp.server --help`` is a CLI metadata query. Handle it
# before importing FastMCP and the runtime tool surface so it remains reliable
# under the startup command's bounded preflight deadline.
if __name__ == "__main__" and _help_requested(sys.argv[1:]):
    sys.stderr.write(f"{_HELP_TEXT}\n")
    raise SystemExit(0)

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
    if _help_requested(sys.argv[1:]):
        logger.info("%s", _HELP_TEXT)
        return
    mcp.run()


if __name__ == "__main__":
    main()
