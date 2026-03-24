# 主动智能规划

> 让 askme 从"被动助手"变成"主动机器人"。
> 看到什么→决定做什么→说出来→执行→反思。

---

## 当前状态

```
被动链路（已通）:
  用户说话 → ASR → LLM → 回复 → TTS → 扬声器

缺失的主动链路:
  摄像头 → 检测 → 理解 → 决策 → 行动 → 说话
```

---

## 四个能力层

### Layer 1: 主动感知 — "我看到了什么"

```
摄像头(5Hz) → BPU YOLO → ChangeDetector → WorldState → 事件
                                                          ↓
                                              "检测到1名人，距离2.3米"
```

**已有的代码**：
- `ChangeDetector` — IoU 匹配 + 去抖（3帧确认出现，5帧确认消失）
- `WorldState` — 场景快照，`get_summary_sync()` 输出中文描述
- `AttentionManager` — 冷却+重要性过滤，防止反复报警
- `ProactiveAgent._change_event_loop()` — 消费事件，触发语音

**缺的**：
- 没有在 blueprint 模式下实测过
- WorldState 的中文描述没接入 TTS
- 事件只写 JSONL 文件，没有触发真正的说话

**要做的**：
1. ProactiveModule 启动时连接 ChangeDetector 事件流
2. 人出现 → `audio.speak_and_wait("检测到有人，距离2.3米")`
3. 人离开 → `audio.speak("人已离开")`
4. 在 S100P 上实测：走到摄像头前 → 3秒内说话

### Layer 2: 主动说话 — "我应该说什么"

```
事件: "person_appeared, distance=2.3m"
  ↓
决策: 这个人是新来的？之前见过？该打招呼还是警告？
  ↓
输出: "你好，欢迎来到仓库A。请注意安全帽。"
```

**已有的代码**：
- `ProactiveAgent._speak_alert()` — 固定模板说话
- `AlertDispatcher` — 多通道分发（语音/webhook/企业微信）
- `EpisodicMemory` — 记录经历，可以回忆

**缺的**：
- 说话内容是固定模板，不是 LLM 生成的
- 不会根据场景上下文调整语气
- 不会记住"这个人我5分钟前见过"

**要做的**：
1. 人出现时用 LLM 生成回复，不用固定模板
2. 传入 WorldState 场景描述 + EpisodicMemory 最近经历作为 context
3. LLM 决定说什么：
   ```
   [场景] 仓库A入口，1名人出现，距离2.3米
   [记忆] 5分钟前该区域有2人经过，未发现异常
   [决定] 打招呼 + 提醒安全帽
   → "你好，这里是仓库A。请确认安全帽已佩戴。"
   ```
4. AttentionManager 控制频率（同一个人10分钟内不重复打招呼）

### Layer 3: 任务编排 — "我该做什么"

```
触发: "仓库B温度传感器异常"
  ↓
规划: 1.导航到仓库B → 2.到达后拍照 → 3.分析异常 → 4.报告
  ↓
执行: navigate("仓库B") → capture_image() → analyze() → report()
  ↓
播报: "已到达仓库B，温度偏高，拍照记录已上传。"
```

**已有的代码**：
- `SkillDispatcher` + `MissionContext` — 多步骤任务跟踪
- `PlannerAgent` — LLM 分解目标为有序步骤
- 41个技能（navigate, patrol_report, find_object 等）
- `ThunderAgentShell` — 自律执行（5轮迭代，120s）

**缺的**：
- 任务只能由用户语音触发，不能由感知事件触发
- PlannerAgent 没有接入感知上下文
- 任务失败没有自动重试或降级策略

**要做的**：
1. 感知事件 → 自动触发任务：
   ```python
   # ProactiveAgent 里
   if event.type == "anomaly_detected":
       await dispatcher.dispatch("navigate", f"去{event.location}检查")
   ```
2. PlannerAgent 接入 WorldState：
   ```python
   planner.plan("去仓库B检查温度", context=world_state.get_summary())
   ```
3. 任务状态实时播报：
   ```
   "正在导航到仓库B..."
   "已到达，正在检查..."
   "检查完毕，温度37度，偏高但在安全范围内。"
   ```
4. 任务失败降级：导航失败 → 语音报告 "无法到达仓库B，通道可能被阻挡"

### Layer 4: 自主决策 — "我自己决定下一步"

```
每 N 分钟:
  1. 读取 WorldState（当前场景）
  2. 读取 EpisodicMemory（最近经历）
  3. 读取任务队列（还有什么没做的）
  4. LLM 决策下一步：
     - "巡检路线上2号点已经30分钟没去了，去看看"
     - "刚才仓库A有异常，15分钟后复查一次"
     - "当前区域安全，继续待命"
  5. 执行决策或待命
```

**已有的代码**：
- `ProactiveAgent._patrol_tick()` — 定时巡检（但只是 VLM 看一眼）
- `_adaptive_interval()` — 根据异常调整巡检频率

**缺的**：
- 没有"决策循环"——不会自己想下一步做什么
- 没有任务优先级排序
- 没有"反思"——不会回顾今天做了什么、发现了什么

**要做的**：
1. `DecisionLoop`（新模块）：
   ```python
   class DecisionModule(Module):
       world_state: In[WorldState]
       memory: In[MemoryContext]

       async def _decision_tick(self):
           scene = self.world_state.get_summary()
           recent = self.memory.get_recent_digest()

           decision = await llm.chat([
               {"role": "system", "content": "你是巡检机器人。根据当前场景和记忆，决定下一步。"},
               {"role": "user", "content": f"场景:{scene}\n记忆:{recent}\n决定:"}
           ])

           if "导航" in decision:
               await dispatcher.dispatch("navigate", decision)
           elif "报告" in decision:
               await dispatcher.dispatch("patrol_report", decision)
           else:
               pass  # 待命
   ```
2. 反思机制：每小时触发 EpisodicMemory.reflect()，总结发现的问题
3. 任务队列：维护一个优先级队列，新感知事件插入，按重要性排序执行

---

## 实施顺序

| 阶段 | 内容 | 依赖 | 工期 |
|------|------|------|------|
| **Phase A** | 主动感知 → 说话 | 摄像头 + TTS（已通） | 1天 |
| **Phase B** | LLM 驱动的主动说话 | Phase A + LLM | 1天 |
| **Phase C** | 感知驱动的任务编排 | Phase B + SkillDispatcher | 2天 |
| **Phase D** | 自主决策循环 | Phase C + DecisionModule | 2天 |

```
Phase A: 摄像头看到人 → "检测到有人"（固定模板）
Phase B: 摄像头看到人 → LLM → "你好，请注意安全帽"（智能回复）
Phase C: 传感器异常 → 自动导航检查 → 拍照 → 报告
Phase D: 机器人自己决定巡检路线、复查时机、待命还是行动
```

---

## 目标验证场景

### 场景 1: 有人来了（Phase A+B）
```
[摄像头检测到人出现，距离3米]
[askme] 你好，欢迎来到仓库A区。请确认安全装备已佩戴。
[人离开]
[askme] (记录到 EpisodicMemory: 15:30 仓库A有访客)
```

### 场景 2: 异常处理（Phase C）
```
[感知到: 仓库B区温度传感器报警]
[askme] 检测到仓库B温度异常，我去看看。
[自动导航到仓库B]
[askme] 已到达仓库B，正在检查... 温度38度，略偏高。已拍照记录并上报。
[自动返回待命点]
```

### 场景 3: 自主巡检（Phase D）
```
[已待命20分钟，无事件]
[决策循环] 3号巡检点已45分钟未巡检，去看看
[askme] 我去3号点巡检一下。
[自动导航 → 检查 → 返回]
[askme] 3号点一切正常。
```
