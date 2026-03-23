# Thunder 可视化调试指南

两种方案可用于远程可视化 S100P 机器人的传感器数据：

| 方案 | 工具 | 适合场景 |
|------|------|----------|
| **Foxglove Studio** | 浏览器 / 桌面 App | 快速查看所有 ROS2 话题，零代码 |
| **Rerun** | Python SDK | 自定义布局，检测框叠加，历史回放 |

---

## 方案 A — Foxglove Studio

### 1. S100P 端：安装并启动桥接

```bash
# 首次安装（需要 sudo）
sudo apt update
sudo apt install ros-humble-foxglove-bridge

# 启动桥接（每次使用前运行）
bash ~/askme/scripts/foxglove_bridge.sh
```

输出正常时会显示：

```
[foxglove_bridge] Starting on port 8765 ...
[foxglove_bridge] Connect from laptop: Foxglove Studio → ws://192.168.66.190:8765
```

### 2. 笔记本端：安装 Foxglove Studio

- 下载地址：https://foxglove.dev/download
- 或使用浏览器版：https://app.foxglove.dev

### 3. 连接

1. 打开 Foxglove Studio
2. 点击 **Open connection**
3. 选择 **Rosbridge WebSocket**
4. 输入：`ws://192.168.66.190:8765`
5. 点击 **Open**

### 4. 可见话题

连接成功后，左侧面板会列出所有话题：

| 话题 | 类型 | 内容 |
|------|------|------|
| `/thunder/imu` | `sensor_msgs/Imu` | IMU 加速度、角速度、姿态四元数 |
| `/thunder/joint_states` | `sensor_msgs/JointState` | 16 DOF 关节位置/速度/力矩 |
| `/thunder/detections` | `std_msgs/String` | JSON 格式检测结果 |
| `/camera/image_raw` | `sensor_msgs/Image` | 相机原始图像 |
| `/cmd_vel` | `geometry_msgs/Twist` | 速度指令 |

### 5. 推荐面板布局

- **Raw Messages** — 查看原始 JSON 数据
- **3D** — 把 `sensor_msgs/Imu` 拖入，查看姿态方块
- **Image** — 把 `/camera/image_raw` 拖入
- **Plot** — 把 `/thunder/imu` 的 `angular_velocity.z` 拖入画实时曲线

### 更改端口

```bash
FOXGLOVE_PORT=8766 bash ~/askme/scripts/foxglove_bridge.sh
```

---

## 方案 B — Rerun

### 1. S100P 端：安装并启动

```bash
# 安装依赖（在 askme venv 内）
source ~/askme/.venv/bin/activate
pip install rerun-sdk

# 启动桥接
source /opt/ros/humble/setup.bash
python3 ~/askme/scripts/rerun_bridge.py
```

输出正常时会显示：

```
Rerun server listening on 0.0.0.0:9876
On laptop: rerun --connect 192.168.66.190:9876
Bridge running. Press Ctrl+C to stop.
```

### 2. 笔记本端：连接

```bash
pip install rerun-sdk
rerun --connect 192.168.66.190:9876
```

Rerun 窗口自动打开，呈现预设布局：

```
┌─────────────────────┬──────────────────────────┐
│  Camera + Detections│  IMU Orientation (deg)   │
│  [2D image + boxes] │  roll / pitch / yaw 曲线  │
├─────────────────────┼──────────────────────────┤
│  Body Orientation   │  Joint Positions (rad)   │
│  [3D 姿态方块]       │  16 关节位置曲线          │
└─────────────────────┴──────────────────────────┘
```

### 3. 话题与 Rerun 路径对应

| ROS2 话题 | Rerun 路径 | 可视化类型 |
|-----------|-----------|-----------|
| `/thunder/imu` | `thunder/imu/roll_deg` 等 | 时间序列折线图 |
| `/thunder/imu` | `thunder/body_transform` | 3D 姿态变换 |
| `/thunder/joint_states` | `thunder/joints/{name}/position` | 时间序列折线图 |
| `/camera/image_raw` | `thunder/camera/image` | 2D 图像 |
| `/thunder/detections` | `thunder/camera/detections` | 2D 检测框叠加 |

检测框格式（`/thunder/detections` JSON）：

```json
[
  {"label": "person", "x1": 120, "y1": 80, "x2": 340, "y2": 460, "conf": 0.87},
  {"label": "chair",  "x1": 400, "y1": 200, "x2": 520, "y2": 380, "conf": 0.63}
]
```

### 4. 以 systemd 服务运行（开机自启）

```bash
# 复制 service 文件
sudo cp ~/askme/scripts/rerun-bridge.service /etc/systemd/system/

# 启用并启动
sudo systemctl daemon-reload
sudo systemctl enable rerun-bridge
sudo systemctl start rerun-bridge

# 查看日志
journalctl -u rerun-bridge -f
```

---

## 快速故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| Foxglove 连不上 | 端口被防火墙拦截 | `sudo ufw allow 8765/tcp` |
| `foxglove_bridge` 包不存在 | 未安装 | `sudo apt install ros-humble-foxglove-bridge` |
| Rerun 画面空白 | 话题没有数据 | 检查 `brainstem-ros2-bridge` 是否在运行 |
| Rerun 连接超时 | 端口 9876 未开放 | `sudo ufw allow 9876/tcp` |
| 关节图只有部分曲线 | 话题 `name` 字段为空 | bridge 用默认名称 `FL_hip` 等填充 |
| 相机图像花屏 | 编码格式不支持 | 仅支持 `rgb8 / bgr8 / mono8`，其他格式降级为亮度图 |

---

## 依赖版本参考

| 组件 | 版本要求 |
|------|---------|
| ROS2 | Humble (Ubuntu 22.04) |
| `ros-humble-foxglove-bridge` | apt 最新 |
| `rerun-sdk` (Python) | >= 0.16 |
| `numpy` | 已在 askme venv 中 |
| Foxglove Studio | 任意最新版 |
