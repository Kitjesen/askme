#!/bin/bash
# askme 语音对话持久化服务启动脚本 — sunrise aarch64
#
# 用法（在 sunrise 机器上）:
#   screen -S askme ./scripts/sunrise-voice-service.sh   # 后台 screen
#   ./scripts/sunrise-voice-service.sh --text            # 文字模式调试
#   ./scripts/sunrise-voice-service.sh --voice           # 语音模式（默认）
#   ./scripts/sunrise-voice-service.sh --voice --robot   # 语音 + 机器人控制
#
# systemd 一键安装:
#   sudo cp scripts/askme.service /etc/systemd/system/
#   sudo systemctl enable askme && sudo systemctl start askme

set -e

ASKME_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ASKME_DIR/.venv"
LOG_DIR="$ASKME_DIR/logs"
RESTART_DELAY=3
MAX_RESTARTS=100

mkdir -p "$LOG_DIR"

# 激活 venv（如果存在）
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi

# 加载 .env
if [ -f "$ASKME_DIR/.env" ]; then
    set -a
    source "$ASKME_DIR/.env"
    set +a
fi

cd "$ASKME_DIR"

# 默认语音模式
MODE="--voice"
if [ "$1" = "--text" ]; then
    MODE=""
elif [ "$1" = "--voice" ]; then
    MODE="--voice"
    shift
fi
EXTRA_ARGS="$@"

echo "=============================="
echo "askme 语音服务启动"
echo "模式: python -m askme --legacy $MODE $EXTRA_ARGS"
echo "日志: $LOG_DIR/askme.log"
echo "=============================="

RESTART_COUNT=0
while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$TIMESTAMP] 启动 askme (第 $((RESTART_COUNT+1)) 次)" | tee -a "$LOG_DIR/askme.log"

    # 运行主程序，stdout/stderr 同时输出到终端和日志
    python -m askme --legacy $MODE $EXTRA_ARGS 2>&1 | tee -a "$LOG_DIR/askme.log" || true

    EXIT_CODE=${PIPESTATUS[0]}
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$TIMESTAMP] askme 退出 (code=$EXIT_CODE)，${RESTART_DELAY}s 后重启..." | tee -a "$LOG_DIR/askme.log"

    RESTART_COUNT=$((RESTART_COUNT+1))
    sleep $RESTART_DELAY
done

echo "最大重启次数 $MAX_RESTARTS 已达到，退出。" | tee -a "$LOG_DIR/askme.log"
exit 1
