#!/usr/bin/env bash
# Deploy askme to Sunrise (192.168.66.190)
# Usage: bash scripts/deploy_agentic_shell.sh [--no-restart] [--no-tests]
#
# Environment overrides:
#   REMOTE=user@host   (default: sunrise@192.168.66.190)
#   RPATH=remote/path  (default: ~/askme)

set -euo pipefail

REMOTE="${REMOTE:-sunrise@192.168.66.190}"
RPATH="${RPATH:-~/askme}"
RESTART=1
RUN_TESTS=1

for arg in "$@"; do
    case $arg in
        --no-restart) RESTART=0 ;;
        --no-tests)   RUN_TESTS=0 ;;
    esac
done

echo "=== Deploying askme to ${REMOTE}:${RPATH} ==="

# ── 1. Sync Python package via rsync (only changed files, no config.yaml) ─────
# config.yaml is intentionally excluded — Sunrise has device-specific audio
# settings (input_device, output_device, noise_gate_peak) that differ from
# the developer's Windows config. Overwriting it would break voice I/O.
echo "[1/4] Syncing Python package..."
rsync -az --checksum \
    --exclude="__pycache__/" \
    --exclude="*.pyc" \
    --exclude=".git/" \
    --exclude="data/" \
    --exclude="logs/" \
    --exclude="models/" \
    --exclude=".env" \
    --exclude="config.yaml" \
    askme/ \
    ${REMOTE}:${RPATH}/askme/

echo "[1/4] Syncing tests and skills..."
rsync -az --checksum \
    --exclude="__pycache__/" --exclude="*.pyc" --exclude="tmp/" \
    tests/ \
    ${REMOTE}:${RPATH}/tests/

rsync -az --checksum \
    askme/skills/builtin/ \
    ${REMOTE}:${RPATH}/askme/skills/builtin/

# ── 2. Sync requirements and install if changed ────────────────────────────────
echo "[2/4] Checking pip requirements..."
rsync -az requirements*.txt ${REMOTE}:${RPATH}/ 2>/dev/null || true
ssh ${REMOTE} "cd ${RPATH} && source .venv/bin/activate && \
    pip install -q -r requirements.txt 2>&1 | grep -E 'Installing|Requirement already' | head -20 || true"

# ── 3. Restart service ─────────────────────────────────────────────────────────
if [ "${RESTART}" = "1" ]; then
    echo "[3/4] Restarting askme service..."
    ssh ${REMOTE} "
        if systemctl is-active --quiet askme 2>/dev/null; then
            sudo systemctl restart askme
            sleep 3
            echo '--- systemd status ---'
            systemctl status askme --no-pager -l | tail -8
        elif screen -ls 2>/dev/null | grep -q 'askme'; then
            screen -S askme -X quit 2>/dev/null || true
            sleep 1
            mkdir -p ${RPATH}/logs
            screen -dmS askme bash -c \
                'cd ${RPATH} && source .venv/bin/activate && \
                 python -m askme --legacy --voice >> logs/askme.log 2>&1'
            sleep 2
            echo 'Screen session restarted: ' \$(screen -ls | grep askme || echo 'not found')
        else
            echo 'No active service — start with: screen -S askme ./scripts/sunrise-voice-service.sh'
        fi
    "
else
    echo "[3/4] Skipping service restart (--no-restart)"
fi

# ── 4. Health check ────────────────────────────────────────────────────────────
echo "[4/4] Running health check on remote..."
ssh ${REMOTE} "cd ${RPATH} && source .venv/bin/activate && \
    python -c '
from askme.agent_shell.thunder_agent_shell import ThunderAgentShell
from askme.pipeline.skill_dispatcher import SkillDispatcher
from askme.tools.builtin_tools import SandboxedBashTool, WriteFileTool
print(\"ThunderAgentShell    ✓\")
print(\"SkillDispatcher      ✓\")
print(\"SandboxedBashTool    ✓\")
print(\"WriteFileTool        ✓\")
'"

if [ "${RUN_TESTS}" = "1" ]; then
    echo ""
    echo "=== Remote test run ==="
    ssh ${REMOTE} "cd ${RPATH} && source .venv/bin/activate && \
        python -m pytest tests/test_thunder_agent_shell.py \
                         tests/test_sandboxed_bash.py \
                         tests/test_skill_dispatcher_real.py \
                         -q --tb=short 2>&1 | tail -10"
fi

echo ""
echo "=== Deploy complete. ==="
echo "    Voice: '帮我研究...' / '写个脚本...' / '帮我分析...'"
echo "    Text:  python -m askme --legacy"
