"""Shared constants — paths, defaults, magic numbers.

All hardcoded paths and daemon file locations centralized here.
Import from this module instead of scattering "/tmp/askme_*" everywhere.
"""

# Frame daemon shared files
DAEMON_HEARTBEAT_PATH = "/tmp/askme_frame_daemon.heartbeat"
DAEMON_COLOR_FRAME_PATH = "/tmp/askme_frame_color.bin"
DAEMON_DEPTH_FRAME_PATH = "/tmp/askme_frame_depth.bin"
DAEMON_DETECTIONS_PATH = "/tmp/askme_frame_detections.json"
DAEMON_ROS2_FRAME_PATH = "/tmp/askme_ros2_frame.bin"

# ChangeDetector event output
CHANGE_EVENTS_PATH = "/tmp/askme_events.jsonl"

# Default BPU model path (Horizon RDK X5)
DEFAULT_BPU_MODEL_PATH = "/home/sunrise/data/models/yolo11s_seg_nashe_640x640_nv12.hbm"

# Daemon staleness threshold (seconds)
DAEMON_MAX_STALENESS = 3.0
