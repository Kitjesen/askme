# Askme 边界定义

> askme 是语音 AI 助手，不是机器人操作系统。
> 它的职责是：听、理解、回答、说。
> 其他一切通过接口调用，不拥有。

---

## 核心身份

```
askme = 语音 AI 助手
     ≠ 机器人操作系统
     ≠ 导航系统
     ≠ 运动控制器
     ≠ 设备管理平台
```

---

## 三层边界

### 第一层：askme 核心（必须有，缺一个不能工作）

| 模块 | 职责 | 为什么是核心 |
|------|------|-------------|
| LLMModule | 理解语言、生成回答 | 没有 LLM 就不能对话 |
| MemoryModule | 记住上下文和经历 | 没有记忆就是金鱼 |
| PipelineModule | 对话流水线 | 串联 LLM + Memory + TTS |
| SkillModule | 技能调度 | 把"去仓库A"翻译成动作 |
| VoiceModule | 听和说 | 语音交互的入口出口 |
| TextModule | 文字交互 | SSH/TUI 的入口出口 |

```
用户说话 → VoiceModule(ASR) → PipelineModule(LLM+Memory) → VoiceModule(TTS) → 扬声器
```

### 第二层：askme 感知（可选，有了更智能，没有也能工作）

| 模块 | 职责 | 为什么是可选 |
|------|------|-------------|
| PulseModule | DDS 数据总线 | 没有传感器数据也能纯对话 |
| PerceptionModule | 场景理解 | 没有摄像头也能纯对话 |
| ReactionModule | 场景反应 | 没有它就不会主动说话，但被动对话不受影响 |

```
摄像头 → PulseModule → PerceptionModule → ReactionModule → 主动说话
                                                    ↓
                                              拔掉这一层
                                                    ↓
                                        用户说话 → 核心层 → 回答（正常）
```

### 第三层：外部插件（不属于 askme，通过接口调用）

| 能力 | 归属 | askme 怎么用 |
|------|------|-------------|
| 电机控制 | brainstem (gRPC) | SkillModule 调 dog-control-service HTTP |
| 导航 | LingTu (ROS2) | SkillModule 调 nav-gateway HTTP |
| LED 状态灯 | runtime 服务 | 通过 HTTP 设置颜色 |
| OTA 更新 | infra/ota | OTABridge 上报状态 |
| 硬件自检 | runtime 服务 | HealthModule 读状态 |
| 巡检路线 | 产品层 | ProactiveModule 定时触发 |

```
askme 不拥有这些能力，只是通过标准接口调用。
就像 Siri 不拥有地图导航，但可以调起地图 App。
```

---

## Blueprint 重新定义

### voice（纯语音助手）
```python
voice = (
    Runtime.use(LLMModule)         # 核心
    + Runtime.use(MemoryModule)     # 核心
    + Runtime.use(PipelineModule)   # 核心
    + Runtime.use(SkillModule)      # 核心
    + Runtime.use(VoiceModule)      # 核心
    + Runtime.use(TextModule)       # 核心
)
# 6 个模块。能听能说能记住。
```

### voice_with_perception（语音 + 感知）
```python
voice_with_perception = (
    voice
    + Runtime.use(PulseModule)        # 感知层
    + Runtime.use(PerceptionModule)   # 感知层
    + Runtime.use(ReactionModule)     # 感知层
)
# 9 个模块。能听能说能看能主动反应。
```

### edge_robot（边缘机器人 = 语音 + 感知 + 外部插件）
```python
edge_robot = (
    voice_with_perception
    + Runtime.use(ControlModule)     # 外部插件
    + Runtime.use(LEDModule)         # 外部插件
    + Runtime.use(ProactiveModule)   # 外部插件
    + Runtime.use(HealthModule)      # 外部插件
    + Runtime.use(ExecutorModule)    # 外部插件
)
# 14 个模块。完整的巡检机器人。
```

### text（纯文字，开发用）
```python
text = (
    Runtime.use(LLMModule)
    + Runtime.use(MemoryModule)
    + Runtime.use(PipelineModule)
    + Runtime.use(SkillModule)
    + Runtime.use(TextModule)
)
# 5 个模块。SSH 里用。
```

---

## 接口边界（askme 怎么调外部能力）

```
askme 内部                    接口                     外部服务
─────────────────────────────────────────────────────────────────
SkillModule ──────→ NavigatorBackend ──────→ nav-gateway (HTTP)
                                           → LingTu (gRPC)

SkillModule ──────→ dog-control-service ──→ brainstem (gRPC)
                    (HTTP REST)            → CAN bus → 电机

PulseModule ──────→ BusBackend ───────────→ CycloneDDS
                    (DDS topics)           → frame_daemon
                                           → brainstem_bridge

ReactionModule ───→ AlertDispatcher ──────→ 企业微信/钉钉/Webhook
                    (多通道)

PipelineModule ───→ LLMBackend ───────────→ MiniMax API
                    (HTTP)                 → Claude Relay
                                           → 本地 Ollama（未来）
```

**askme 只定义接口（NavigatorBackend、LLMBackend、BusBackend），不拥有实现。**
**实现通过 registry 注册，config 选择。**

---

## 判断标准

加一个功能前问三个问题：

1. **没有它 askme 还能对话吗？**
   - 能 → 不是核心，是感知层或外部插件
   - 不能 → 核心层

2. **它需要硬件吗？**
   - 需要 → 外部插件，通过接口调用
   - 不需要 → 可能是核心或感知层

3. **它属于"听理解回答说"吗？**
   - 属于 → 核心层
   - 不属于 → 感知层或外部插件
