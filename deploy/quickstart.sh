#!/bin/bash
# Askme + ZeroClaw quick deployment helper for Linux/macOS.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

case "${1:-menu}" in
  setup)
    echo "[1/3] Configuring ZeroClaw API key..."
    python scripts/dev/setup_zeroclaw.py

    echo "[2/3] Creating Docker .env..."
    python -c "
from askme.config import get_config
c = get_config()
b = c['brain'] if isinstance(c, dict) else c.brain
k = b.get('minimax_api_key','') if isinstance(b, dict) else getattr(b, 'minimax_api_key','')
open('docker/.env','w', encoding='utf-8').write(f'MINIMAX_API_KEY={k}\n')
"
    echo "[3/3] Done. Run: bash deploy/quickstart.sh docker"
    ;;

  docker)
    docker compose --env-file docker/.env -f docker/docker-compose.yml up -d
    echo "Askme:    http://localhost:8765"
    echo "ZeroClaw: http://localhost:8080"
    ;;

  local)
    python -m askme.blueprints.presets.edge_robot &
    sleep 5
    zeroclaw gateway --host 127.0.0.1 --port 8080 &
    echo "Services started locally"
    ;;

  stop)
    docker compose --env-file docker/.env -f docker/docker-compose.yml down 2>/dev/null || true
    pkill -f zeroclaw 2>/dev/null || true
    pkill -f askme.blueprints 2>/dev/null || true
    echo "Stopped"
    ;;

  *)
    echo "Usage: bash deploy/quickstart.sh [setup|docker|local|stop]"
    ;;
esac
