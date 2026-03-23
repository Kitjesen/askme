#!/bin/bash
# Start Foxglove WebSocket bridge on S100P
#
# Foxglove Studio (laptop) → ws://192.168.66.190:8765
#
# Prerequisites (first-time, run as sudo):
#   sudo apt update
#   sudo apt install ros-humble-foxglove-bridge
#
# Usage:
#   bash scripts/foxglove_bridge.sh
#
# Then open Foxglove Studio on your laptop and connect to:
#   ws://192.168.66.190:8765

set -euo pipefail

PORT="${FOXGLOVE_PORT:-8765}"

# Source ROS2 environment
if [[ -f /opt/ros/humble/setup.bash ]]; then
    source /opt/ros/humble/setup.bash
else
    echo "[foxglove_bridge] ERROR: /opt/ros/humble/setup.bash not found" >&2
    exit 1
fi

# Check if package is available
if ! ros2 pkg list 2>/dev/null | grep -q foxglove_bridge; then
    echo "[foxglove_bridge] foxglove_bridge package not found."
    echo ""
    echo "Install with:"
    echo "  sudo apt update"
    echo "  sudo apt install ros-humble-foxglove-bridge"
    echo ""
    exit 1
fi

echo "[foxglove_bridge] Starting on port ${PORT} ..."
echo "[foxglove_bridge] Connect from laptop: Foxglove Studio → ws://192.168.66.190:${PORT}"
echo "[foxglove_bridge] All /thunder/* topics will be available."
echo ""

exec ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:="${PORT}"
