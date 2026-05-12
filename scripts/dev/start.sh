#!/bin/bash
# Start askme MCP server
# Usage:
#   ./scripts/dev/start.sh                             # stdio (default)
#   ./scripts/dev/start.sh --transport sse --port 8080 # SSE mode
#   ./scripts/dev/start.sh --legacy --text             # legacy CLI
set -e
cd "$(dirname "$0")/../.."
source .env 2>/dev/null || true
exec python -m askme "$@"
