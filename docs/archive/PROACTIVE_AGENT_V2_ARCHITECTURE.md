# ProactiveAgent V2 架构设计稿

**主题：从"定时巡检器"升级为"事件驱动的主动感知与处置系统"**
**日期：2026-03-21**
**状态：设计完成，待实施**

---

## 一、背景与当前问题

现有 ProactiveAgent 的核心工作方式，是按固定时间间隔执行 `_patrol_tick()`，周期性调用 `VisionBridge.describe_scene()` 获取场景描述，再结合历史帧让 LLM 判断是否异常。如果异常成立，则触发 TTS 报警和 `auto_solve`；如果没有异常，则写入 `scene_history`，并周期性输出"巡检正常"报告。

这套机制能工作，但本质上是"时钟驱动的定时轮询架构"，而不是"由环境变化触发的主动感知架构"。

核心问题：
1. VLM/LLM 被放在最前面，不管场景是否变化都消耗推理资源
2. 异常检测、语义理解、决策判断、自动处置都塞在同一个巡检闭环里，职责耦合过深
3. 系统没有显式的风险模型和注意力机制
4. 检测能力 5Hz（frame_daemon），消费能力 0.008Hz（120s 一次），99.8% 的检测结果被浪费

## 二、设计目标

1. **实时性**：人出现/消失等事件 1 秒级感知（不是 120 秒）
2. **成本下降**：静态时段不重复调用 VLM/LLM
3. **分层推理**：不同级别变化匹配不同级别处理
4. **可扩展**：后续能接入音频、门磁、机器人本体状态等

## 三、总体设计原则

1. 变化优先于时间 — 高成本推理由事件触发，不是周期触发
2. 检测、解释、决策、执行分层 — 不同阶段由不同模块承担
3. 简单事件用轻量规则 — 只有风险升高时才升级到 VLM/LLM
4. 注意力管理 — 知道"这件事是否值得进一步花算力"
5. 不确定性驱动兜底 — 避免 YOLO 漏检导致全局失明

## 四、六层架构

```
┌─ 感知源层 (Perception Sources) ──────────────────┐
│  frame_daemon: 目标检测 + ROI 统计 + 感知质量     │
│  (未来: 音频、门磁、IMU、环境传感器)              │
└──────────────────────┬───────────────────────────┘
                       ↓
┌─ 事件抽取层 (ChangeDetector) ────────────────────┐
│  连续观测 → 离散事件                              │
│  短期基线 + 长期基线对比                          │
│  事件去抖 + 持续性判断                            │
│  输出: person_entered_roi, object_removed, etc.   │
└──────────────────────┬───────────────────────────┘
                       ↓
┌─ 世界状态层 (WorldState) ────────────────────────┐
│  当前活动目标 + 区域占用 + 事件历史                │
│  处置记录 + 风险状态 + 感知质量趋势               │
│  最后 VLM 摘要                                    │
└──────────────────────┬───────────────────────────┘
                       ↓
┌─ 注意力与风险层 (AttentionManager) ──────────────┐
│  风险评分 (事件×区域×时段×持续×不确定性)          │
│  状态机: idle → tracking → suspicious → incident │
│  扫描频率动态调节                                 │
└──────────────────────┬───────────────────────────┘
                       ↓ (仅中高风险)
┌─ 主动决策层 (ProactiveAgent V2) ────────────────┐
│  消费高层事件 + 风险状态                          │
│  条件性调用 VLM/LLM (不是每轮都调)               │
│  决策: 记录 / 提醒 / VLM / LLM / auto_solve     │
└──────────────────────┬───────────────────────────┘
                       ↓
┌─ 执行反馈层 ─────────────────────────────────────┐
│  TTS 播报 + 消息通知 + auto_solve + 人工升级      │
│  结果回写 WorldState → 闭环                       │
└──────────────────────────────────────────────────┘
```

## 五、事件分级机制

| 级别 | 名称 | 处理方 | 延迟 | 成本 |
|------|------|--------|------|------|
| L0 | 原始观测 | frame_daemon | 实时 | 零 |
| L1 | 轻量事件 | ChangeDetector (规则) | 1s | 零 |
| L2 | 语义复核 | VLM (千问VL) | 2-3s | 低 |
| L3 | 推理决策 | LLM (MiniMax M2.7) | 3-5s | 中 |

静态时段 → 仅 L0/L1
可疑事件 → 升级到 L2
高风险 → 进入 L3

## 六、状态机

```
idle ──(轻微变化)──→ tracking
  ↑                      │
  │              (持续/敏感区域)
  │                      ↓
  │               suspicious ──(VLM确认)──→ incident
  │                      │                      │
  │              (自然消退)              (处置完成)
  │                      │                      ↓
  └──(稳定窗口)──── idle ←──────────── recovering
```

## 七、风险评分因子

- 事件基础权重 (person_in_restricted >> chair_moved)
- 事件持续时间 (连续 5s > 单帧)
- 检测置信度
- 区域优先级 (限制区域 > 普通区域)
- 时间段权重 (夜间 > 白天)
- 历史重复度
- 感知质量 (低质量 → 增加不确定性复核需求)

## 八、VLM 兜底触发条件

1. **最大静默时间** — 高价值区域长期无高质量观测更新
2. **不确定性触发** — 检测置信度持续低/目标数异常抖动/画面模糊
3. **周期性保底** — 低风险 30min，高价值 10-15min（远低于现在的 2min）

## 九、模块拆分

```
askme/
├── perception/
│   └── change_detector.py    # 事件提取、去抖、基线对比
├── memory/
│   └── world_state.py        # 对象状态、ROI、历史事件、状态机
├── attention/
│   └── attention_manager.py  # 风险评分、状态迁移、扫描调节
├── pipeline/
│   └── proactive_agent.py    # V2: 事件消费、深度复核、决策处置
├── schemas/
│   ├── observation.py        # 单帧观测结构
│   ├── events.py             # 标准化事件结构
│   └── state.py              # WorldState 快照
└── incident/
    └── incident_logger.py    # 异常记录和审计
```

## 十、主循环伪代码

```python
while True:
    obs = frame_daemon.read_observation()
    events = change_detector.extract_events(obs, world_state)

    for event in events:
        world_state.apply_event(event)

    attention = attention_manager.evaluate(world_state, events)

    if attention.should_trigger_vlm:
        vlm_summary = vision_bridge.describe_scene(attention.focus_region)
        world_state.update_vlm_summary(vlm_summary)

    if attention.should_escalate_to_agent:
        decision = proactive_agent.handle(
            events=events,
            world_state=world_state,
            vlm_summary=world_state.latest_vlm_summary,
        )
        executor.run(decision.actions)
        world_state.apply_decision(decision)

    sleep(attention.next_tick_hint)
```

## 十一、三阶段迁移计划

### Phase 1: 事件能力建设
- 保留现有 ProactiveAgent
- 实现 ChangeDetector（最小版）
- frame_daemon 稳定输出观测结构
- 新旧并行运行，对比验证

### Phase 2: 注意力层引入
- 引入 WorldState + AttentionManager
- VLM 调用决定权从 patrol_tick 移到注意力层
- 成本和实时性显著改善

### Phase 3: 架构切换
- 删除主定时巡检逻辑
- ProactiveAgent 收缩为事件消费+决策处置
- 完成"定时器架构"到"注意力架构"切换

## 十二、效果评估指标

1. **事件检测延迟**: 场景变化 → 系统产生事件的时间
2. **深度模型调用密度**: 每小时 VLM/LLM 触发次数
3. **误报率/漏报率**: 分高风险/普通区域
4. **处置闭环完成率**: 异常 → 动作 → 记录
5. **静态场景成本下降**: 反映架构升级是否成功
