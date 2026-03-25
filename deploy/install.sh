#!/bin/bash
# Askme deployment script for S100P (sunrise)
# Usage: bash deploy/install.sh
set -e

ASKME_DIR="/home/sunrise/askme"
SERVICE_FILE="/etc/systemd/system/askme.service"

echo "=== Askme Deploy ==="

# 1. Install systemd service
if [ -f deploy/askme.service ]; then
    sudo cp deploy/askme.service "$SERVICE_FILE"
    sudo systemctl daemon-reload
    echo "[OK] systemd service installed"
else
    echo "[SKIP] askme.service not found"
fi

# 2. Ensure venv and deps
if [ ! -d "$ASKME_DIR/.venv" ]; then
    python3 -m venv "$ASKME_DIR/.venv"
    echo "[OK] venv created"
fi

source "$ASKME_DIR/.venv/bin/activate"
pip install -q -e "$ASKME_DIR"
pip install -q robotmem jieba
echo "[OK] dependencies installed"

# 3. Enable and start
sudo systemctl enable askme
echo "[OK] service enabled (auto-start on boot)"

echo ""
echo "Commands:"
echo "  sudo systemctl start askme    # start"
echo "  sudo systemctl stop askme     # stop"
echo "  sudo systemctl restart askme  # restart"
echo "  journalctl -u askme -f        # logs"
echo ""
echo "Deploy complete!"
