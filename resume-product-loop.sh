#!/bin/bash
# Resume product loop from Round 2
WORKDIR="D:/inovxio/tools/askme"
OTA_DIR="D:/inovxio/infra/ota"
LOGDIR="$WORKDIR/dev-loop-logs"
MAINLOG="$WORKDIR/dev-loop-product-main.log"
mkdir -p "$LOGDIR"

TASKS=(
  ""  # Round 1 done
  "HEALTH_ENDPOINT: Add /health and /metrics HTTP endpoints to askme. Read the existing app.py and ota_bridge.py first. Expose via a lightweight aiohttp or fastapi server: uptime, model name, last LLM latency, total conversations, active skills, voice pipeline status, OTA bridge connection state. OTA server will poll these. Port 8765. Start as background asyncio task in main.py."
  "REMOTE_CONFIG: Add remote config hot-reload to askme. Read ota_bridge.py and config.yaml first. Add config_watcher.py that polls OTA /api/devices/{device_id}/configs endpoint, detects version changes, and hot-reloads brain.model/brain.system_prompt/voice params without restart. Track config version in a local file."
  "DOCKER_COMPOSE: Create docker-compose.product.yml for Thunder deployment. Read D:/inovxio/infra/ota/docker-compose.yml for patterns. Combine askme + ota-agent sidecar. askme: build from ./Dockerfile, env_file, volumes, depends_on ota-agent. Include healthchecks. Also create a basic Dockerfile for askme if one doesn't exist."
  "ROBOT_STATUS_SKILL: Add robot_status skill to askme/skills/. Read existing skill files first for patterns. Implement: when user says '系统状态' or 'system status', query ota_bridge for OTA connection state, last heartbeat time, pending upgrades, return formatted response. Add to skills registry."
  "ALERT_HANDLER: Add alert_listener.py to askme — subscribe to OTA SSE stream /api/alerts/stream. On critical alerts (device offline, upgrade failed), proactively trigger TTS announcement. Read existing voice pipeline code first. Start as background asyncio task."
  "DEPLOYMENT_DOCS: Create docs/DEPLOYMENT.md for field engineers. Cover: prerequisites, OTA server setup, Thunder device provisioning (nova-provision.py), docker-compose.product.yml deployment, config management, OTA upgrade workflow, troubleshooting. Practical, step-by-step."
  "INTEGRATION_TEST: Write tests/test_ota_integration.py. Read existing test files and ota_bridge.py first. Mock OTA endpoints with aiohttp test server. Test: device registration, heartbeat, config pull, health endpoint, alert subscription. Use pytest-asyncio. All tests must pass."
)

START_ROUND=2
MAX_ROUNDS=8

cd "$WORKDIR"

for ROUND in $(seq $START_ROUND $MAX_ROUNDS); do
  LOGFILE="$LOGDIR/product-round-$ROUND.log"
  TASK="${TASKS[$((ROUND-1))]}"

  echo "=============================" | tee -a "$MAINLOG" | tee "$LOGFILE"
  echo "Product Round $ROUND / $MAX_ROUNDS" | tee -a "$MAINLOG" | tee -a "$LOGFILE"
  echo "Time: $(date)" | tee -a "$MAINLOG" | tee -a "$LOGFILE"
  echo "Task: ${TASK%%:*}" | tee -a "$MAINLOG" | tee -a "$LOGFILE"
  echo "=============================" | tee -a "$MAINLOG" | tee -a "$LOGFILE"

  HISTORY=$(git log --oneline -8 2>/dev/null || echo "no commits yet")

  PROMPT="You are a senior AI engineer building the NOVA Thunder industrial robot product.

CONTEXT:
- askme (D:/inovxio/tools/askme) = Thunder's AI brain (voice AI, LLM, skills, robot control, v4.0.0)
- OTA Platform (D:/inovxio/infra/ota) = remote management backend (v3.2.0, FastAPI, SQLite)
- ota_bridge.py already exists in askme — reads OTA config from config.yaml

Recent git history:
$HISTORY

YOUR TASK (Round $ROUND): $TASK

Instructions:
1. ALWAYS read existing source files before writing anything
2. Implement completely — no TODOs, no placeholders
3. Follow code patterns and style of existing files
4. git add -A && git commit -m 'feat: [description] (Product Round $ROUND)'
5. Verify syntax before committing

Be thorough. This goes into a real industrial robot."

  echo "Running Codex Round $ROUND..." | tee -a "$MAINLOG" | tee -a "$LOGFILE"
  codex exec --full-auto "$PROMPT" >> "$LOGFILE" 2>&1
  EXIT_CODE=$?

  LAST_COMMIT=$(git log --oneline -1 2>/dev/null || echo "no commit")
  echo "Round $ROUND done. Exit: $EXIT_CODE. Commit: $LAST_COMMIT" | tee -a "$MAINLOG"

  openclaw system event --text "Product Round $ROUND 完成: $LAST_COMMIT" --mode now 2>/dev/null || true

  sleep 3
done

echo "Product loop Round 2-8 complete!" | tee -a "$MAINLOG"
git log --oneline -10 >> "$MAINLOG" 2>/dev/null
openclaw system event --text "askme 产品化 Round 2-8 全部完成！查看 dev-loop-product-main.log" --mode now 2>/dev/null || true
