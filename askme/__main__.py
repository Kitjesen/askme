"""
CLI entry point for askme.

    python -m askme                                  # MCP server (stdio)
    python -m askme --transport sse --port 8080      # MCP server (SSE)
    python -m askme --config /path/to/config.yaml    # custom config
    python -m askme --legacy                         # legacy CLI voice mode
    python -m askme --legacy --text                  # legacy CLI text mode
    python -m askme --legacy --robot                 # legacy CLI with robot arm
"""

import argparse
import asyncio
import logging
import os


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="askme",
        description="Askme — Voice AI + Robot MCP Server",
    )

    # MCP server options
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="MCP transport mode (default: stdio)",
    )
    parser.add_argument(
        "--host", default="localhost",
        help="Host for SSE transport (default: localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Port for SSE transport (default: 8080)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.yaml (overrides default)",
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None, help="Override log level",
    )

    # Legacy CLI options
    parser.add_argument(
        "--legacy", action="store_true",
        help="Run legacy interactive CLI instead of MCP server",
    )
    parser.add_argument(
        "--text", action="store_true",
        help="(Legacy) Text input mode",
    )
    parser.add_argument(
        "--robot", action="store_true",
        help="(Legacy) Enable robot arm control",
    )
    args = parser.parse_args()

    # Apply config path override before anything imports config
    if args.config:
        os.environ["ASKME_CONFIG_PATH"] = args.config

    # Apply log level override
    if args.log_level:
        logging.getLogger().setLevel(getattr(logging, args.log_level))

    if args.legacy:
        from askme.app import AskmeApp

        app = AskmeApp(
            voice_mode=not args.text,
            robot_mode=args.robot,
        )
        asyncio.run(app.run())
    else:
        from askme.mcp_server import mcp

        if args.transport == "sse":
            mcp.run(transport="sse", host=args.host, port=args.port)
        else:
            mcp.run()


if __name__ == "__main__":
    main()
