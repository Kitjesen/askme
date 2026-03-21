#!/usr/bin/env bash
# End-to-end integration test for askme on sunrise.
# Usage: bash scripts/e2e_test.sh
# Requires: askme running on sunrise via tmux

set -euo pipefail

REMOTE="sunrise@192.168.66.190"
URL="http://localhost:8765"
PASS=0
FAIL=0

test_case() {
    local name="$1" endpoint="$2" body="$3" expect="$4"
    local result
    result=$(ssh "$REMOTE" "curl -s -m 15 -X POST $URL$endpoint -H 'Content-Type: application/json' -d '$body'" 2>/dev/null || echo "TIMEOUT")

    if echo "$result" | grep -q "$expect"; then
        echo "  ✅ $name"
        PASS=$((PASS + 1))
    else
        echo "  ❌ $name — expected '$expect' in: ${result:0:100}"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Askme E2E Test Suite ==="
echo ""

# 1. Health check
echo "[Health]"
health=$(ssh "$REMOTE" "curl -s -m 5 $URL/health" 2>/dev/null)
if echo "$health" | grep -q '"status":"ok"'; then
    echo "  ✅ Health endpoint"
    PASS=$((PASS + 1))
else
    echo "  ❌ Health endpoint — askme may not be running"
    FAIL=$((FAIL + 1))
fi

# 2. Quick reply (no LLM)
echo "[Quick Reply]"
test_case "你好" "/api/chat" '{"text":"你好"}' "你好"
test_case "在吗" "/api/chat" '{"text":"在吗"}' "在的"
test_case "谢谢" "/api/chat" '{"text":"谢谢"}' "不客气"

# 3. Skill routing
echo "[Skill Routing]"
test_case "几点了 (get_time)" "/api/chat" '{"text":"几点了"}' "202"
test_case "故障记录 (recall_memory)" "/api/chat" '{"text":"有什么故障记录"}' "故障"

# 4. Disabled skills should not trigger
echo "[Disabled Skills]"
result=$(ssh "$REMOTE" "curl -s -m 15 -X POST $URL/api/chat -H 'Content-Type: application/json' -d '{\"text\":\"站起来\"}'" 2>/dev/null)
if echo "$result" | grep -qv "dog_control"; then
    echo "  ✅ dog_control disabled (not triggered)"
    PASS=$((PASS + 1))
else
    echo "  ❌ dog_control should be disabled"
    FAIL=$((FAIL + 1))
fi

# 5. Services status
echo "[Services]"
daemon=$(ssh "$REMOTE" "cat /tmp/askme_frame_daemon.heartbeat 2>/dev/null" || echo "")
if [ -n "$daemon" ]; then
    echo "  ✅ frame_daemon alive"
    PASS=$((PASS + 1))
else
    echo "  ❌ frame_daemon not running"
    FAIL=$((FAIL + 1))
fi

orbbec=$(ssh "$REMOTE" "systemctl is-active orbbec-camera 2>/dev/null" || echo "inactive")
if [ "$orbbec" = "active" ]; then
    echo "  ✅ orbbec-camera service"
    PASS=$((PASS + 1))
else
    echo "  ❌ orbbec-camera service: $orbbec"
    FAIL=$((FAIL + 1))
fi

# Summary
echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
exit $FAIL
