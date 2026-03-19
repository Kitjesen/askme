#!/usr/bin/env bash
# Sync askme source code to sunrise robot.
# Usage: bash scripts/sync_sunrise.sh
#
# Prerequisites: SSH key auth configured for sunrise@192.168.66.190
# (already set up — no password needed)

set -euo pipefail

REMOTE="sunrise@192.168.66.190"
REMOTE_DIR="~/askme"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Syncing askme to $REMOTE:$REMOTE_DIR ==="
echo "Local: $LOCAL_DIR"

# Sync entire askme package + config + tests
rsync -avz --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.venv' \
    --exclude 'data/' \
    --exclude 'models/' \
    --exclude 'logs/' \
    --exclude '.env' \
    --exclude '.git' \
    --exclude '.omc' \
    --exclude '.tmp' \
    --exclude '.claude' \
    --exclude 'dev-loop-*.log' \
    --exclude 'docs/' \
    "$LOCAL_DIR/askme/" "$REMOTE:$REMOTE_DIR/askme/" \
    2>/dev/null || {
    # rsync not available on Windows — fall back to scp
    echo "rsync not available, using scp..."
    find "$LOCAL_DIR/askme" \( -name '__pycache__' -o -name 'tmp' \) -prune -o \( -name '*.py' -o -name '*.md' -o -name '*.yaml' \) -print | while read -r f; do
        rel="${f#$LOCAL_DIR/}"
        dir="$(dirname "$rel")"
        ssh "$REMOTE" "mkdir -p $REMOTE_DIR/$dir" 2>/dev/null
        scp -q "$f" "$REMOTE:$REMOTE_DIR/$rel"
    done
}

# Always sync config, requirements, and key root files
scp -q "$LOCAL_DIR/config.yaml" "$REMOTE:$REMOTE_DIR/config.yaml"
scp -q "$LOCAL_DIR/pyproject.toml" "$REMOTE:$REMOTE_DIR/pyproject.toml"
scp -q "$LOCAL_DIR/requirements.txt" "$REMOTE:$REMOTE_DIR/requirements.txt"
scp -q "$LOCAL_DIR/SOUL.md" "$REMOTE:$REMOTE_DIR/SOUL.md"

# Sync scripts/
ssh "$REMOTE" "mkdir -p $REMOTE_DIR/scripts"
find "$LOCAL_DIR/scripts" \( -name '__pycache__' -o -name 'test_data' \) -prune -o \( -name '*.py' -o -name '*.sh' -o -name '*.bat' -o -name '*.service' \) -print | while read -r f; do
    rel="${f#$LOCAL_DIR/}"
    scp -q "$f" "$REMOTE:$REMOTE_DIR/$rel"
done
rsync -avz --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    "$LOCAL_DIR/tests/" "$REMOTE:$REMOTE_DIR/tests/" \
    2>/dev/null || {
    echo "Syncing tests via scp..."
    find "$LOCAL_DIR/tests" \( -name 'tmp' -o -name '__pycache__' \) -prune -o -name '*.py' -print | while read -r f; do
        rel="${f#$LOCAL_DIR/}"
        scp -q "$f" "$REMOTE:$REMOTE_DIR/$rel"
    done
}

echo ""
echo "=== Verifying on sunrise ==="
ssh "$REMOTE" "cd $REMOTE_DIR && source .venv/bin/activate && python -c 'import askme; print(f\"askme imported OK\")'"

echo ""
echo "=== Done. To run tests: ==="
echo "  ssh $REMOTE 'cd $REMOTE_DIR && source .venv/bin/activate && python -m pytest tests/ -q'"
echo ""
echo "=== To restart service: ==="
echo "  ssh $REMOTE 'screen -S askme -X quit; screen -S askme -dm bash -c \"cd $REMOTE_DIR && ./scripts/sunrise-voice-service.sh\"'"
