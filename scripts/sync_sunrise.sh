#!/usr/bin/env bash
# Sync askme source code to sunrise robot.
# Usage: bash scripts/sync_sunrise.sh
#
# Uses tar-over-ssh for fast, reliable sync on Windows (no rsync needed).
# One SSH connection, handles new directories automatically, ~5s total.

set -euo pipefail

REMOTE="sunrise@192.168.66.190"
REMOTE_DIR="~/askme"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Syncing askme to $REMOTE:$REMOTE_DIR ==="
echo "Local: $LOCAL_DIR"

# ── Fast sync via tar pipe (one SSH connection) ──────────────────────

sync_dir() {
    local src="$1" dst="$2" name="$3"
    echo "  Syncing $name..."
    # tar from local, extract on remote — handles new dirs, fast, reliable
    tar -C "$src" \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.pytest_cache' \
        --exclude='tmp' \
        -cf - . | ssh "$REMOTE" "mkdir -p $dst && tar -C $dst -xf -"
}

# Core package
sync_dir "$LOCAL_DIR/askme" "$REMOTE_DIR/askme" "askme/"

# Tests
sync_dir "$LOCAL_DIR/tests" "$REMOTE_DIR/tests" "tests/"

# Scripts (selective — only deploy-relevant files)
ssh "$REMOTE" "mkdir -p $REMOTE_DIR/scripts"
tar -C "$LOCAL_DIR/scripts" \
    --exclude='__pycache__' \
    --exclude='test_data' \
    --exclude='mock_scene.jpg' \
    -cf - . | ssh "$REMOTE" "tar -C $REMOTE_DIR/scripts -xf -"
echo "  Syncing scripts/"

# Root config files (NOT config.yaml — sunrise has its own config)
for f in pyproject.toml requirements.txt SOUL.md README.md; do
    if [ -f "$LOCAL_DIR/$f" ]; then
        scp -q "$LOCAL_DIR/$f" "$REMOTE:$REMOTE_DIR/$f"
    fi
done
echo "  Syncing root configs (config.yaml excluded — sunrise-specific)"

# Shared python (inovxio_llm, qp_memory)
if [ -d "$LOCAL_DIR/shared_python" ]; then
    sync_dir "$LOCAL_DIR/shared_python" "$REMOTE_DIR/shared_python" "shared_python/"
fi

# ── Verify ───────────────────────────────────────────────────────────

echo ""
echo "=== Verifying on sunrise ==="
ssh "$REMOTE" "cd $REMOTE_DIR && source .venv/bin/activate && python -c 'import askme; print(f\"askme imported OK\")'"

echo ""
echo "=== Done ==="
echo "  Restart: ssh $REMOTE 'tmux kill-session -t askme; tmux new-session -d -s askme bash -c \"cd $REMOTE_DIR && source .venv/bin/activate && export \\\$(grep -v ^# .env | xargs) && python -m askme --legacy\"'"
