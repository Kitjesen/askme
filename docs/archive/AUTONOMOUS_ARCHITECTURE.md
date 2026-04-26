# Thunder 自主运行架构

> 最后更新: 2026-03-21

## 系统总览

```
┌─────────────────────────────────────────────────────────────┐
│              Thunder 自主运行循环 (24/7)                      │
│                                                               │
│  ┌─ ProactiveAgent (自适应巡逻) ──────────────────────────┐  │
│  │                                                         │  │
│  │  巡逻调度 (AdaptiveScheduler)                           │  │
│  │    ├ 白天高峰 (09-11, 14-16): 每 60s 扫描              │  │
│  │    ├ 白天空闲: 每 120s 扫描                             │  │
│  │    ├ 夜间 (22-06): 每 300s 扫描                         │  │
│  │    └ 异常后: 立即缩短到 30s，渐进恢复                    │  │
│  │                                                         │  │
│  │  感知 ─────────────────────────────────────────────     │  │
│  │    ├ BPU YOLO (3ms, daemon 预计算) → 80 类物体检测      │  │
│  │    ├ 千问 VL (1s) → 开放词汇场景理解                    │  │
│  │    ├ Orbbec 深度 (毫米距离)                              │  │
│  │    └ Frame Daemon (1ms 读帧, 5Hz 持续更新)              │  │
│  │                                                         │  │
│  │  判断 ─────────────────────────────────────────────     │  │
│  │    ├ LLM 异常检测 (MiniMax M2.7)                        │  │
│  │    ├ 场景对比 (当前 vs 历史 5 帧)                       │  │
│  │    └ qp_memory 知识关联 (设备故障历史/日常流程)          │  │
│  │                                                         │  │
│  │  响应 ─────────────────────────────────────────────     │  │
│  │    ├ 正常 → 记录 qp_memory, 继续巡逻                   │  │
│  │    └ 异常 →                                             │  │
│  │        ├ speak_progress("发现异常: ...")                 │  │
│  │        ├ save_snapshot (证据拍照)                        │  │
│  │        ├ episodic_memory 记录事件                        │  │
│  │        └ 自动触发 solve_problem (OTREV 框架)            │  │
│  │            ├ O: 观察现场细节 (look_around + depth)       │  │
│  │            ├ T: LLM 分析根因                             │  │
│  │            ├ R: web_search (Bing/百度) 搜索方案          │  │
│  │            ├ E: bash/robot_api 执行修复                  │  │
│  │            └ V: 验证 → 成功则记忆，失败则重试(max 3)     │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌─ VoiceLoop (语音循环, 持续监听) ──────────────────────┐  │
│  │  麦克风 → 降噪 → VAD → [唤醒词] → ASR → IntentRouter   │  │
│  │    ├ ESTOP → 急停                                       │  │
│  │    ├ VOICE_TRIGGER → 40 个技能                          │  │
│  │    │   ├ find_object (帮我找瓶水)                       │  │
│  │    │   ├ safety_check (安全检查)                        │  │
│  │    │   ├ find_person (有人吗)                           │  │
│  │    │   ├ check_location (去看看门关了没)                 │  │
│  │    │   ├ patrol_scan (开始巡逻)                         │  │
│  │    │   ├ solve_problem (怎么办)                         │  │
│  │    │   └ ... 34 个其他技能                              │  │
│  │    └ GENERAL → BrainPipeline (LLM 对话)                │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌─ 记忆系统 ────────────────────────────────────────────┐  │
│  │  L1: Conversation (40 条滑动窗口)                       │  │
│  │  L2: SessionMemory (会话摘要 .md)                       │  │
│  │  L3: EpisodicMemory (事件日志 + 自动反思)               │  │
│  │  L4: qp_memory (长期知识, DashScope qwen-turbo 提取)    │  │
│  │      ├ SiteKnowledge (地点/坐标/事件)                   │  │
│  │      ├ ProceduralMemory (操作流程)                       │  │
│  │      └ MarkdownStore (设备/事件/人员/流程 .md)           │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌─ 基础设施 ────────────────────────────────────────────┐  │
│  │  systemd 服务:                                          │  │
│  │    ├ orbbec-camera (Gemini 335, 30fps)                  │  │
│  │    ├ askme-frame-daemon (帧 + BPU YOLO + depth)         │  │
│  │    └ askme (语音 + 巡逻 + Agent)                        │  │
│  │  硬件:                                                  │  │
│  │    ├ S100P RDK X5 (Nash BPU 128 TOPS)                   │  │
│  │    ├ Orbbec Gemini 335 (RGB 1280x720 + Depth 848x480)   │  │
│  │    ├ MCP01 USB 麦克风 + 扬声器                          │  │
│  │    └ 238G NVMe SSD + 45G eMMC                           │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| TTFT (首 token) | 0.76s | MiniMax M2.7 |
| YOLO 检测 | 0.2ms | BPU daemon 预计算 |
| 帧读取 | 1ms | daemon 共享文件 |
| VLM 问答 | ~1s | 千问 VL (DashScope) |
| TTS | ~1.6s | MiniMax Speech 2.8 HD |
| 深度测距 | 毫米精度 | Orbbec Gemini 335 |
| 搜索 | ~2s | Bing + 百度 |
| 360° 扫描 | ~10s | scan_around |
| find_object | ~30-40s | 5 轮 Agent 迭代 |

## 技术栈

| 层 | 技术 |
|----|------|
| LLM (对话) | MiniMax M2.7 highspeed |
| LLM (Agent) | MiniMax M2.7 highspeed |
| VLM (视觉理解) | 千问 VL (DashScope) |
| ASR | sherpa-onnx zipformer zh-int8 |
| TTS | MiniMax Speech 2.8 HD |
| YOLO | YOLOv8s-seg on Nash BPU |
| 深度相机 | Orbbec Gemini 335 (ROS2) |
| 记忆提取 | DashScope qwen-turbo |
| 搜索 | Bing → 百度 |
| 运动控制 | ROS2 cmd_vel / semantic nav |

## Agent 技能矩阵 (40 个)

### 自主推理
- `solve_problem` — OTREV 自主问题解决 (15 触发词)
- `agent_task` — 通用自律执行 (38 触发词)

### 视觉+运动
- `find_object` — 搜索任意物体 (12 触发词)
- `find_person` — 找人 (8 触发词)
- `safety_check` — 安全巡检 (9 触发词)
- `check_location` — 远程查看 (8 触发词)
- `patrol_scan` — 多点巡逻 (8 触发词)

### 导航
- `navigate` — 语义导航
- `mapping` — SLAM 建图
- `follow_person` — 人员跟随
- `nav_query` / `nav_cancel` — 导航状态

### 控制
- `dog_control` — 姿态控制 (站/坐/趴)
- `robot_move/grab/home/estop` — 机械臂

### 交互
- `volume_up/down/reset` — 音量
- `speed_up/down/reset` — 语速
- `mute_mic/unmute_mic` — 麦克风
- `stop_speaking/repeat_last` — 播放控制

### 工具
- `web_search` — 联网搜索
- `system_status` — 系统状态
- `get_time` — 时间查询
- `list_skills/list_directory` — 信息查看
- `workspace_info/workspace_clear` — 工作区管理
