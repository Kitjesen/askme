#!/usr/bin/env python3
"""Persistent frame capture + BPU YOLO daemon for askme vision.

Runs as a systemd service (system Python, NOT venv).
Continuously:
  1. Captures latest RGB + Depth frames from ROS2 topics
  2. Runs BPU YOLO detection on each new color frame (~3ms)
  3. Writes frames + detections to /tmp/askme_frame_*.bin|json

askme reads results instantly — zero subprocess overhead, zero hobot_dnn dependency in venv.

Usage:
    python3 scripts/frame_daemon.py
    python3 scripts/frame_daemon.py --color /camera/color/image_raw --rate 5 --bpu-model /path/to.hbm

Install as service:
    sudo cp scripts/askme-frame-daemon.service /etc/systemd/system/
    sudo systemctl enable --now askme-frame-daemon
"""

import argparse
import json
import struct
import time
import os
import signal
import sys

import numpy as np

# Paths where frames/detections are written (atomically via rename)
COLOR_PATH = "/tmp/askme_frame_color.bin"
DEPTH_PATH = "/tmp/askme_frame_depth.bin"
DETECTIONS_PATH = "/tmp/askme_frame_detections.json"
HEARTBEAT_PATH = "/tmp/askme_frame_daemon.heartbeat"

# Default BPU model
DEFAULT_BPU_MODEL = "/home/sunrise/data/models/yolo11s_seg_nashe_640x640_nv12.hbm"

# COCO 80 class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def rgb_to_nv12(rgb, h=640, w=640):
    """Convert RGB numpy array to NV12 1D for BPU input."""
    from PIL import Image as PILImage
    img = PILImage.fromarray(rgb).resize((w, h), PILImage.BILINEAR)
    arr = np.asarray(img, dtype=np.int32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    y = np.clip(((66*r + 129*g + 25*b + 128) >> 8) + 16, 0, 255).astype(np.uint8)
    u = np.clip(((-38*r - 74*g + 112*b + 128) >> 8) + 128, 0, 255).astype(np.uint8)
    v = np.clip(((112*r - 94*g - 18*b + 128) >> 8) + 128, 0, 255).astype(np.uint8)
    uv = np.empty((h//2, w), dtype=np.uint8)
    uv[:, 0::2] = u[0::2, 0::2]
    uv[:, 1::2] = v[0::2, 0::2]
    return np.concatenate([y.ravel(), uv.ravel()])


def decode_bpu_outputs(outputs, conf=0.35, iou=0.45, orig_h=720, orig_w=1280):
    """Decode BPU YOLO v11 seg outputs into detection list."""
    all_boxes, all_scores, all_cls = [], [], []
    strides = [8, 16, 32]

    for si in range(3):
        cls_out = outputs[si*3].buffer[0].astype(np.float32)
        bbox_out = outputs[si*3+1].buffer[0].astype(np.float32)
        gh, gw, nc = cls_out.shape
        stride = strides[si]

        cls_scores = 1.0 / (1.0 + np.exp(-cls_out))
        max_s = cls_scores.max(axis=-1)
        ys, xs = np.where(max_s > conf)
        for y, x in zip(ys, xs):
            scores = cls_scores[y, x]
            cid = int(scores.argmax())
            sc = float(scores[cid])
            if sc < conf:
                continue
            raw = bbox_out[y, x].reshape(4, 16)
            exp = np.exp(raw - raw.max(axis=1, keepdims=True))
            soft = exp / exp.sum(axis=1, keepdims=True)
            dfl = (soft * np.arange(16, dtype=np.float32)).sum(axis=1)
            cx = (x + 0.5) * stride
            cy = (y + 0.5) * stride
            x1 = max(0, cx - dfl[0]*stride) * orig_w / 640
            y1 = max(0, cy - dfl[1]*stride) * orig_h / 640
            x2 = min(640, cx + dfl[2]*stride) * orig_w / 640
            y2 = min(640, cy + dfl[3]*stride) * orig_h / 640
            all_boxes.append([x1,y1,x2,y2])
            all_scores.append(sc)
            all_cls.append(cid)

    if not all_boxes:
        return []

    # NMS
    boxes = np.array(all_boxes)
    scores = np.array(all_scores)
    order = scores.argsort()[::-1]
    areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(int(i))
        xx1 = np.maximum(boxes[i,0], boxes[order[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[order[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[order[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[order[1:],3])
        inter = np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        ious = inter/(areas[i]+areas[order[1:]]-inter+1e-6)
        order = order[np.where(ious <= iou)[0] + 1]

    return [{
        "class_id": COCO_CLASSES[all_cls[i]] if all_cls[i] < 80 else f"class_{all_cls[i]}",
        "confidence": round(all_scores[i], 3),
        "bbox": [round(v,1) for v in all_boxes[i]],
    } for i in keep]


def main():
    parser = argparse.ArgumentParser(description="Askme frame+BPU daemon")
    parser.add_argument("--color", default="/camera/color/image_raw")
    parser.add_argument("--depth", default="/camera/depth/image_raw")
    parser.add_argument("--rate", type=float, default=5.0, help="Max rate (Hz)")
    parser.add_argument("--bpu-model", default=DEFAULT_BPU_MODEL, help="BPU .hbm model path")
    parser.add_argument("--no-bpu", action="store_true", help="Disable BPU detection")
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool, String

    # Init BPU model
    bpu_model = None
    if not args.no_bpu and os.path.exists(args.bpu_model):
        try:
            from hobot_dnn import pyeasy_dnn
            models = pyeasy_dnn.load(args.bpu_model)
            bpu_model = models[0]
            print(f"[frame_daemon] BPU model loaded: {args.bpu_model.split('/')[-1]}")
        except Exception as e:
            print(f"[frame_daemon] BPU init failed: {e}, running without detection")

    rclpy.init()
    node = Node("askme_frame_daemon")

    _be_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    _pub_detections = node.create_publisher(String, "/thunder/detections", _be_qos)
    _pub_heartbeat = node.create_publisher(Bool, "/thunder/heartbeat", _be_qos)

    color_frame = [None]
    depth_frame = [None]
    node.create_subscription(Image, args.color, lambda m: color_frame.__setitem__(0, m), 1)
    node.create_subscription(Image, args.depth, lambda m: depth_frame.__setitem__(0, m), 1)

    min_interval = 1.0 / args.rate
    running = True
    signal.signal(signal.SIGTERM, lambda *_: running or None)
    signal.signal(signal.SIGINT, lambda *_: running or None)

    # Mutable flag for signal handler
    stop = [False]
    def on_signal(*_):
        stop[0] = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    print(f"[frame_daemon] Started: color={args.color} depth={args.depth} "
          f"rate={args.rate}Hz bpu={'ON' if bpu_model else 'OFF'}")
    sys.stdout.flush()

    last_write = 0
    while not stop[0]:
        executor.spin_once(timeout_sec=0.1)
        now = time.monotonic()
        if now - last_write < min_interval:
            continue

        # Write color frame
        if color_frame[0] is not None:
            msg = color_frame[0]
            tmp = COLOR_PATH + ".tmp"
            try:
                with open(tmp, "wb") as f:
                    f.write(struct.pack("II", msg.width, msg.height))
                    f.write(bytes(msg.data))
                os.rename(tmp, COLOR_PATH)
            except Exception:
                pass

            # BPU detection on color frame
            if bpu_model is not None:
                try:
                    rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                    nv12 = rgb_to_nv12(rgb)
                    t0 = time.monotonic()
                    outputs = bpu_model.forward(nv12)
                    infer_ms = (time.monotonic() - t0) * 1000
                    dets = decode_bpu_outputs(outputs, orig_h=msg.height, orig_w=msg.width)
                    det_json = {
                        "timestamp": time.time(),
                        "infer_ms": round(infer_ms, 1),
                        "detections": dets,
                    }
                    tmp_d = DETECTIONS_PATH + ".tmp"
                    with open(tmp_d, "w") as f:
                        json.dump(det_json, f)
                    os.rename(tmp_d, DETECTIONS_PATH)
                    _pub_detections.publish(String(data=json.dumps(det_json)))
                except Exception as e:
                    pass  # silent — don't crash daemon on detection error

        # Write depth frame
        if depth_frame[0] is not None:
            msg = depth_frame[0]
            tmp = DEPTH_PATH + ".tmp"
            try:
                with open(tmp, "wb") as f:
                    f.write(struct.pack("II", msg.width, msg.height))
                    f.write(bytes(msg.data))
                os.rename(tmp, DEPTH_PATH)
            except Exception:
                pass

        # Heartbeat
        try:
            with open(HEARTBEAT_PATH, "w") as f:
                f.write(f"{time.time():.3f}\n")
            _pub_heartbeat.publish(Bool(data=True))
        except Exception:
            pass

        last_write = now

    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()
    print("[frame_daemon] Stopped")


if __name__ == "__main__":
    main()
