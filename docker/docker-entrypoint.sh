#!/bin/bash
set -e

# ─────────────────────────────────────────────────────────────
# Askme Edge Runtime — Docker Entrypoint
# ─────────────────────────────────────────────────────────────
# 职责:
#   1. 等依赖就绪 (可选)
#   2. 运行数据迁移/初始化
#   3. exec 主进程 (信号透传)
# ─────────────────────────────────────────────────────────────

# ── 日志前缀 ────────────────────────────────────────────────
log() {
    echo "[entrypoint] $(date '+%Y-%m-%d %H:%M:%S') $*"
}

# ── 可选: 等健康端点就绪 ──────────────────────────────────
wait_for_service() {
    local url="$1"
    local label="$2"
    local timeout="${3:-30}"
    local interval=2
    local elapsed=0

    if [ -z "$url" ]; then
        return 0
    fi

    log "等待 $label 就绪: $url (最长 ${timeout}s)"
    while [ $elapsed -lt $timeout ]; do
        if wget -q --spider "$url" 2>/dev/null || \
           curl -sf -o /dev/null "$url" 2>/dev/null; then
            log "$label 就绪"
            return 0
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    log "警告: $label 未在 ${timeout}s 内就绪，继续启动"
    return 0
}

# ── 可选依赖等待 ────────────────────────────────────────────
if [ -n "${WAIT_FOR_HEALTH:-}" ]; then
    wait_for_service "$WAIT_FOR_HEALTH" "health-endpoint" "${WAIT_FOR_TIMEOUT:-30}"
fi

# ── 确保数据目录存在 ──────────────────────────────────────
if [ -n "${ASKME_DATA_DIR:-}" ]; then
    mkdir -p "$ASKME_DATA_DIR" 2>/dev/null || true
fi

# Fail closed before the edge runtime opens hardware or provider sessions.
log "验证 edge_robot 模型与音频设备"
if ! python -m askme.runtime.deployment_preflight; then
    log "错误: edge_robot 前置检查失败；运行时未启动"
    exit 78
fi

# ── exec 主进程 (替换 shell，信号直达 python) ──────────────
log "启动 askme edge_robot blueprint"
exec python -m askme.blueprints.presets.edge_robot "$@"
