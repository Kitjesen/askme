#!/bin/bash
# Start askme MCP server
# Usage:
#   ./scripts/start.sh                             # stdio (default)
#   ./scripts/start.sh --transport sse --port 8080 # SSE mode
#   ./scripts/start.sh --legacy --text             # legacy CLI
set -e
cd "$(dirname "$0")/.."
source .env 2>/dev/null || true
exec python -m askme "$@"
