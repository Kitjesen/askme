#!/bin/bash
# askme Product Loop — Codex-only iterative product integration
# Goal: integrate askme with OTA backend to form complete Thunder product

set -e
WORKDIR="D:/inovxio/tools/askme"
OTA_DIR="D:/inovxio/infra/ota"
LOGDIR="$WORKDIR/dev-loop-logs"
mkdir -p "$LOGDIR"

MAX_ROUNDS=${1:-8}
ROUND=${2:-1}

cd "$WORKDIR"

TASKS=(
  "OTA_BRIDGE: Add ota_bridge.py module to askme — Thunder registers with OTA server on startup, sends periodic heartbeat+telemetry (LLM call latency, conversation count, skill success rate, voice pipeline status) via OTA agent API. Read D:/inovxio/infra/ota/agent/ for the existing agent code patterns. Add config section in config.yaml for OTA endpoint. The bridge should run as a background asyncio task."
  "HEALTH_ENDPOINT: Add /health and /metrics HTTP endpoints to askme app.py (or a new health_server.py). Expose: uptime, model name, last LLM latency, total conversations, active skills, voice pipeline status. OTA server will poll these to track device health. Use fastapi or aiohttp — whatever fits the existing stack."
  "REMOTE_CONFIG: Add remote config hot-reload to askme. OTA server's /configs API can push JSON config updates to devices. In askme, add a config_watcher.py that polls OTA /configs endpoint periodically, detects changes, and hot-reloads brain model/system_prompt/voice params without restart. Add config version tracking."
  "DOCKER_COMPOSE: Create unified docker-compose.product.yml combining askme + ota-agent in one stack for Thunder deployment. Read D:/inovxio/infra/ota/docker-compose.yml for OTA patterns. askme service: build from ./Dockerfile, depends on ota-agent sidecar. Include env_file references, volume mounts, network config, healthchecks."
  "ROBOT_STATUS_SKILL: Add robot_status skill to askme/skills/ that reports system health to the user via voice. 'Thunder，系统状态如何？' should return: OTA connection status, last sync time, pending upgrades, hardware health score. Integrates with the ota_bridge module."
  "ALERT_HANDLER: Add alert handling to askme — subscribe to OTA SSE /api/alerts/stream. When OTA sends critical alerts (upgrade failed, device offline warning), askme should proactively announce via TTS/voice pipeline. Add alert_listener.py as background task."
  "DEPLOYMENT_DOCS: Create docs/DEPLOYMENT.md explaining the complete Thunder software stack deployment: OTA server setup, Thunder device provisioning flow, askme+ota-agent docker compose, config management, upgrade workflow. Write for a field engineer deploying to a new robot."
  "INTEGRATION_TEST: Write tests/test_ota_integration.py — mock OTA server endpoints, test: registration, heartbeat, config pull, remote config apply, health endpoint, alert subscription. Full async test suite using pytest-asyncio."
)

while [ $ROUND -le $MAX_ROUNDS ]; do
  LOGFILE="$LOGDIR/product-round-$ROUND.log"
  TASK_IDX=$((ROUND - 1))
  TASK="${TASKS[$TASK_IDX]}"

  echo "=============================" | tee "$LOGFILE"
  echo "🏭 Product Round $ROUND / $MAX_ROUNDS" | tee -a "$LOGFILE"
  echo "Time: $(date)" | tee -a "$LOGFILE"
  echo "Task: ${TASK%%:*}" | tee -a "$LOGFILE"
  echo "=============================" | tee -a "$LOGFILE"

  HISTORY=$(git log --oneline -8 2>/dev/null || echo "no commits yet")

  PROMPT="You are a senior AI engineer building the NOVA Thunder industrial robot product.

CONTEXT:
- askme (D:/inovxio/tools/askme) = Thunder's AI brain (voice AI, LLM, skills, robot control)
- OTA Platform (D:/inovxio/infra/ota) = remote management backend (v3.2.0, FastAPI, SQLite)
- Goal: integrate these two into one deployable product for field deployment

Recent git history (askme repo):
$HISTORY

YOUR TASK (Round $ROUND): $TASK

Instructions:
1. Read the relevant existing source files FIRST before writing anything
2. Implement completely — no TODOs, no placeholders
3. Follow existing code patterns and style in each repo
4. Add to git: git add -A && git commit -m 'feat: [description] (Product Round $ROUND)'
5. Verify your code is syntactically correct before committing

Be thorough. This goes into a real industrial robot."

  echo "🤖 Running Codex..." | tee -a "$LOGFILE"
  codex exec --full-auto "$PROMPT" 2>&1 | tee -a "$LOGFILE"
  EXIT_CODE=${PIPESTATUS[0]}

  echo "" | tee -a "$LOGFILE"
  echo "Round $ROUND exit code: $EXIT_CODE" | tee -a "$LOGFILE"

  LAST_COMMIT=$(git log --oneline -1 2>/dev/null || echo "no commit")
  echo "Latest commit: $LAST_COMMIT" | tee -a "$LOGFILE"

  if [ $ROUND -eq $MAX_ROUNDS ]; then
    openclaw system event --text "askme 产品化 loop 完成！$MAX_ROUNDS 轮。最后: $LAST_COMMIT" --mode now 2>/dev/null || true
  else
    openclaw system event --text "Product Round $ROUND 完成 → Round $((ROUND+1)) 开始。$LAST_COMMIT" --mode now 2>/dev/null || true
  fi

  ROUND=$((ROUND + 1))
  sleep 3
done

echo ""
echo "✅ Product loop complete!"
git log --oneline -"$MAX_ROUNDS" 2>/dev/null
