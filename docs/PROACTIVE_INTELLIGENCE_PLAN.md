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

---

## Appendix: ReactionEngine Design

## Summary

The askme codebase already has a solid perception-to-alert pipeline (ChangeDetector -> ChangeEvent -> ProactiveAgent -> AlertDispatcher), but the reaction logic is a single if-branch at `proactive_agent.py:469` that speaks a fixed template for every person event. The design below introduces a **ReactionEngine** -- a rule-first, LLM-second decision layer that sits between the existing ChangeEvent stream and the existing AlertDispatcher/SkillDispatcher outputs. It reuses every existing component (WorldState, AttentionManager, EpisodicMemory, SiteKnowledge, AlertDispatcher, SkillDispatcher) and adds exactly one new class plus one config section.

## Analysis

### What Exists Today (with file:line references)

**Perception pipeline (working):**
- `askme/perception/change_detector.py:56` -- ChangeDetector compares consecutive YOLO frames via greedy IoU matching, emits debounced ChangeEvents
- `askme/perception/world_state.py:77` -- WorldState tracks all visible objects with track_id, class_id, bbox, distance_m, first_seen, last_seen, duration_s
- `askme/perception/attention_manager.py:64` -- AttentionManager enforces per-event-type cooldowns and importance thresholds

**Current reaction logic (the bottleneck):**
- `askme/pipeline/proactive_agent.py:459-487` -- `_handle_change_event()` is the sole reaction handler. It does three things: (1) log to episodic memory, (2) if `event.is_person_event` speak the fixed template, (3) if importance >= 0.7 trigger auto-solve
- `askme/schemas/events.py:98-112` -- `description_zh()` produces hardcoded strings like "检测到有人出现" -- no context, no scene understanding

**Available context signals (already computed, not used for decisions):**
- `askme/perception/world_state.py:45-59` -- TrackedObject has `distance_m`, `duration_s`, `first_seen`, `last_seen`, `bbox`
- `askme/memory/site_knowledge.py:88` -- SiteKnowledge has Location with coords, tags, anomaly_count, visit_count
- `askme/memory/episodic_memory.py:227-245` -- `retrieve(query)` does Park 2023-style scoring (recency + importance + keyword relevance)
- `askme/memory/episodic_memory.py:378-406` -- `get_recent_digest()` returns Chinese-language recent experience summary

**Output channels (already working):**
- `askme/pipeline/alert_dispatcher.py:62` -- AlertDispatcher with severity routing to voice/webhook/wecom/dingtalk/feishu
- `askme/pipeline/skill_dispatcher.py:1-10` -- SkillDispatcher with MissionContext for multi-step skill execution
- `askme/pipeline/proactive_agent.py:162` -- `set_solve_callback()` already wires ProactiveAgent to SkillDispatcher

**Module system:**
- `askme/runtime/module.py:252` -- Module base class with In/Out typed ports, auto-wiring, topo-sort
- `askme/runtime/modules/proactive_module.py:25` -- ProactiveModule wraps ProactiveAgent, depends on (pipeline, memory)

### The Gap

The gap is between `_handle_change_event()` receiving a ChangeEvent and choosing a reaction. Currently that decision is one if-statement (`if event.is_person_event: speak`). There is no consideration of:
- Person distance or movement direction
- How long the person has been in scene
- Time of day
- Location/zone context
- Whether this person was seen recently
- What the robot is currently doing (in a mission? idle?)

### Design Constraints (from hardware reality)

- **S100P**: 4GB RAM, aarch64, BPU handles YOLO at 5Hz -- CPU budget is tight
- **LLM latency**: MiniMax M2.7 ~1-2s TTFT, relay Opus ~5s -- cannot call LLM for every frame
- **TTS latency**: MiniMax TTS ~0.5-1s -- fast enough for real-time reactions
- **YOLO output**: class_id, confidence, bbox, distance_m -- no pose, no face recognition, no gaze direction

---

## Design: Reaction Engine

### Architecture Overview

```
ChangeEvent (from ChangeDetector)
     |
     v
+------------------+
| ReactionEngine   |  <-- NEW: the only new class
|                  |
|  1. Build        |  WorldState.get_objects_sync() -> scene signals
|     SceneContext  |  SiteKnowledge.get_location() -> zone context
|                  |  EpisodicMemory.retrieve() -> recent memory
|                  |  time.localtime() -> time context
|                  |  robot_state (idle/mission) -> activity context
|                  |
|  2. Rule-based   |  SceneContext -> ReactionType (fast, <1ms)
|     Decision     |  (deterministic matrix, no LLM)
|                  |
|  3. Content      |  if needs_content: LLM generates spoken text
|     Generation   |  if rule-only: use template (fast)
|     (optional)   |
|                  |
|  4. Execute      |  AlertDispatcher.dispatch() for speak/alert
|     Reaction     |  SkillDispatcher.dispatch() for ACT
|                  |  EpisodicMemory.log() for all
+------------------+
```

### 1. SceneContext -- signals available for decision-making

Every signal below is already computed by existing code. No new sensors needed.

```python
@dataclass
class SceneContext:
    """All signals available for reaction decisions. Built from existing data."""

    # From ChangeEvent
    event: ChangeEvent                          # the triggering event
    
    # From WorldState (already tracked)
    person_count: int                           # how many persons visible right now
    person_distance_m: float | None             # nearest person distance (TrackedObject.distance_m)
    person_duration_s: float                    # how long this person has been in scene (TrackedObject.duration_s)
    person_bbox_position: str                   # "left" / "center" / "right" (from bbox center_x)
    total_objects: int                          # total tracked objects
    
    # Derived from consecutive WorldState snapshots (2-frame delta)
    person_approaching: bool                    # distance decreasing over last 2 readings
    person_stationary: bool                     # bbox center moved < 5% of frame width
    
    # From time
    hour: int                                   # 0-23
    is_business_hours: bool                     # configurable, default 08:00-18:00
    
    # From SiteKnowledge (if robot knows its position)
    zone_name: str                              # "" if unknown
    zone_tags: list[str]                        # ["restricted"], ["entrance"], ["work_area"], []
    
    # From EpisodicMemory
    seen_person_recently: bool                  # person_appeared event in last N minutes
    minutes_since_last_person: float            # time since last person_appeared event
    
    # From robot state
    robot_busy: bool                            # currently executing a mission
    wake_word_heard: bool                       # wake word detected (from VoiceModule)
```

**Where each signal comes from:**

| Signal | Source | Cost |
|--------|--------|------|
| `person_count` | `WorldState.get_persons_sync()` at `world_state.py:201` | O(n), ~0 |
| `person_distance_m` | `TrackedObject.distance_m` at `world_state.py:55` | Already computed |
| `person_duration_s` | `TrackedObject.duration_s` property at `world_state.py:58-59` | Already computed |
| `person_approaching` | Compare current distance_m vs previous (store in ReactionEngine) | 1 float comparison |
| `person_stationary` | Compare bbox center vs previous | 1 float comparison |
| `hour` | `datetime.now().hour` | ~0 |
| `zone_name/tags` | `SiteKnowledge.find_nearby()` at `site_knowledge.py:166` | O(locations), <1ms |
| `seen_person_recently` | `EpisodicMemory.retrieve("person")` at `episodic_memory.py:227` | O(buffer), <5ms |
| `robot_busy` | Check if `SkillDispatcher.current_mission` is RUNNING | 1 attribute read |

**Key design decision**: `person_approaching` requires storing the previous distance reading. ReactionEngine holds a `_prev_distance: dict[str, float]` keyed by track_id. This is a single dict, not a new sensor.

### 2. Reaction Types

```python
class ReactionType(Enum):
    IGNORE = "ignore"         # log only, no output
    OBSERVE = "observe"       # update WorldState, log, no output
    GREET = "greet"           # short TTS greeting
    INFORM = "inform"         # provide information
    WARN = "warn"             # security/safety warning
    ASSIST = "assist"         # offer help after delay
    ALERT = "alert"           # escalate to operator (webhook + IM)
    ACT = "act"               # trigger a skill/mission
```

### 3. Decision Matrix (rule-based, no LLM)

The matrix is evaluated top-to-bottom; first match wins. Each row is a pure function of SceneContext fields, evaluated in <1ms.

```python
# Priority-ordered rules. First match wins.
_REACTION_RULES: list[tuple[str, Callable[[SceneContext], bool], ReactionType, dict]] = [
    # ---- Safety / Security (highest priority) ----
    
    ("restricted_zone_person",
     lambda ctx: ctx.event.is_person_event 
                 and ctx.event.event_type == ChangeEventType.PERSON_APPEARED
                 and "restricted" in ctx.zone_tags,
     ReactionType.WARN,
     {"severity": "warning", "template": "请注意，此区域需要授权进入。"}),
    
    ("after_hours_unknown",
     lambda ctx: ctx.event.is_person_event
                 and ctx.event.event_type == ChangeEventType.PERSON_APPEARED
                 and not ctx.is_business_hours,
     ReactionType.ALERT,
     {"severity": "warning", "template": "非工作时间检测到人员，已通知管理人员。",
      "escalate": True}),
    
    # ---- Wake word (always respond) ----
    
    ("wake_word",
     lambda ctx: ctx.wake_word_heard,
     ReactionType.ACT,
     {"action": "enter_conversation"}),
    
    # ---- Robot busy (suppress most reactions) ----
    
    ("busy_ignore",
     lambda ctx: ctx.robot_busy and ctx.event.is_person_event
                 and "restricted" not in ctx.zone_tags,
     ReactionType.OBSERVE,
     {}),
    
    # ---- Person passing through (IGNORE) ----
    
    ("person_passing",
     lambda ctx: ctx.event.event_type == ChangeEventType.PERSON_APPEARED
                 and ctx.person_duration_s < 3.0
                 and not ctx.person_approaching,
     ReactionType.IGNORE,
     {}),
    
    # ---- Recently seen (suppress duplicate greetings) ----
    
    ("person_seen_recently",
     lambda ctx: ctx.event.event_type == ChangeEventType.PERSON_APPEARED
                 and ctx.seen_person_recently
                 and ctx.minutes_since_last_person < 10.0,
     ReactionType.OBSERVE,
     {}),
    
    # ---- Person approaching and stops (GREET) ----
    
    ("person_approaching_greet",
     lambda ctx: ctx.event.event_type == ChangeEventType.PERSON_APPEARED
                 and ctx.person_approaching
                 and ctx.person_distance_m is not None
                 and ctx.person_distance_m < 4.0,
     ReactionType.GREET,
     {"use_llm": True}),
    
    # ---- Person standing a long time (ASSIST) ----
    
    ("person_lingering",
     lambda ctx: ctx.event.event_type != ChangeEventType.PERSON_LEFT
                 and ctx.person_duration_s > 120.0  # 2 minutes
                 and ctx.person_stationary,
     ReactionType.ASSIST,
     {"use_llm": True, "template_fallback": "你好，需要帮助吗？"}),
    
    # ---- Person appeared at entrance (GREET with template) ----
    
    ("entrance_greet",
     lambda ctx: ctx.event.event_type == ChangeEventType.PERSON_APPEARED
                 and "entrance" in ctx.zone_tags,
     ReactionType.GREET,
     {"template": "你好，欢迎。"}),
    
    # ---- Person appeared, generic (OBSERVE only) ----
    
    ("person_appeared_default",
     lambda ctx: ctx.event.event_type == ChangeEventType.PERSON_APPEARED,
     ReactionType.OBSERVE,
     {}),
    
    # ---- Person left ----
    
    ("person_left",
     lambda ctx: ctx.event.event_type == ChangeEventType.PERSON_LEFT,
     ReactionType.IGNORE,
     {}),
    
    # ---- Non-person events (existing anomaly logic handles these) ----
    
    ("object_change_default",
     lambda ctx: True,
     ReactionType.OBSERVE,
     {}),
]
```

**Critical change from current behavior**: The default for `PERSON_APPEARED` is now `OBSERVE` (silent), not speak. Only specific conditions trigger speech. This directly addresses the requirement "not everyone wants to hear a response."

### 4. LLM vs Rules -- the Hybrid Split

| Phase | Method | Latency | When Used |
|-------|--------|---------|-----------|
| **Decision** (WHAT to do) | Rule matrix | <1ms | Always |
| **Content** (WHAT to say) | LLM or template | 0ms or ~2s | Only when reaction requires speech |
| **Execution** (DO it) | AlertDispatcher / SkillDispatcher | varies | Only when reaction is not IGNORE/OBSERVE |

The LLM is called **only** when the rule metadata includes `"use_llm": True`. This means:
- `IGNORE`, `OBSERVE`: zero LLM calls (most events)
- `WARN`, `ALERT` with template: zero LLM calls (safety-critical = fast + deterministic)
- `GREET`, `ASSIST` with `use_llm: True`: one LLM call to generate contextual response

**LLM prompt for content generation** (called after rule decision, only for GREET/ASSIST/INFORM):

```
你是巡检机器人Thunder。根据以下场景信息，用一句简短的中文回应。
不超过30字。语气友好但专业。

场景: {world_state_summary}
时间: {time_str}
地点: {zone_name}
记忆: {recent_digest}
反应类型: {reaction_type}

回应:
```

This prompt is ~150 tokens input, ~20 tokens output. At MiniMax M2.7 speed, total latency ~1-2s, acceptable for a greeting.

### 5. Integration with Existing Code

**Where ReactionEngine lives**: New file `askme/pipeline/reaction_engine.py`. Not a new Module -- it is a component owned by ProactiveModule, same as how ProactiveAgent is owned today.

**Minimal changes to existing code:**

**(a) `proactive_agent.py:459` -- replace `_handle_change_event()`**

The current method:
```python
async def _handle_change_event(self, event):
    description = event.description_zh()
    if event.is_person_event:
        await self._speak_alert(description, ...)
    if event.importance >= 0.7 and self._auto_solve:
        await self._solve_callback(...)
```

Becomes:
```python
async def _handle_change_event(self, event):
    reaction = await self._reaction_engine.decide(event)
    await self._reaction_engine.execute(reaction)
```

**(b) `proactive_module.py:32` -- inject dependencies into ReactionEngine**

ReactionEngine needs: WorldState, EpisodicMemory, SiteKnowledge, AttentionManager, LLMClient, AlertDispatcher, SkillDispatcher. All of these are already accessible via the ModuleRegistry during `build()`. No new In/Out ports needed.

```python
# In ProactiveModule.build():
self.reaction_engine = ReactionEngine(
    world_state=world_state,          # from PerceptionModule (new attribute to expose)
    episodic=episodic,                # from MemoryModule
    site_knowledge=site_knowledge,    # from MemoryModule
    attention=AttentionManager(cfg),  # already exists, just instantiate
    llm=llm,                          # from LLMModule
    alert_dispatcher=self.agent._alert_dispatcher,  # from ProactiveAgent
    skill_dispatcher=skill_dispatcher, # from SkillModule
    config=cfg,
)
self.agent._reaction_engine = self.reaction_engine
```

**(c) WorldState needs to be accessible from PerceptionModule**

Currently `PerceptionModule` at `perception_module.py:21` does not instantiate WorldState. The WorldState needs to be created and populated by ChangeDetector events. Two options:

- **Option A**: PerceptionModule creates WorldState, ChangeDetector feeds it. ProactiveModule reads it via registry.
- **Option B**: ReactionEngine creates its own WorldState and subscribes to the same event stream.

**Recommendation**: Option A. Add `self.world_state = WorldState()` to `PerceptionModule.build()` and have ChangeDetector call `world_state.apply_event_sync()` in its `_emit_events()`. This keeps WorldState as a shared singleton that any module can read.

**(d) Config addition (under `proactive:`):**

```yaml
proactive:
  reaction:
    enabled: true
    business_hours: [8, 18]           # start, end hour
    greet_cooldown: 600               # seconds before re-greeting same zone
    llm_content_model: "MiniMax-M2.7-highspeed"
    llm_content_timeout: 5.0
    zones:                            # optional zone definitions
      - name: "仓库A入口"
        tags: ["entrance"]
        coords: [2.5, 1.0]
      - name: "仓库B"
        tags: ["restricted"]
        coords: [10.0, 5.0]
```

**No new In/Out ports needed**. The ReactionEngine is a plain class composed inside ProactiveModule, not a separate Module. This avoids adding complexity to the module topology.

### 6. Data flow diagram (with existing components highlighted)

```
YOLO (5Hz, BPU)
     |
     v
ChangeDetector [EXISTING, change_detector.py]
     |  ChangeEvent
     v
AttentionManager.should_alert() [EXISTING, attention_manager.py]
     |  (filters low-importance / cooldown)
     v
ReactionEngine.decide() [NEW, reaction_engine.py]
     |
     +-- reads WorldState [EXISTING, world_state.py] -- person_count, distance, duration
     +-- reads SiteKnowledge [EXISTING, site_knowledge.py] -- zone_name, zone_tags
     +-- reads EpisodicMemory [EXISTING, episodic_memory.py] -- seen_person_recently
     +-- reads robot_state (SkillDispatcher.current_mission)
     +-- reads time-of-day
     |
     v  (rule matrix, <1ms)
ReactionDecision {type: ReactionType, metadata: dict}
     |
     v
ReactionEngine.execute()
     |
     +-- IGNORE → EpisodicMemory.log() only
     +-- OBSERVE → EpisodicMemory.log() + WorldState update
     +-- GREET/ASSIST → LLM content gen (optional) → AlertDispatcher.dispatch()
     +-- WARN → template → AlertDispatcher.dispatch(severity="warning")
     +-- ALERT → template → AlertDispatcher.dispatch(severity="error") + webhook/IM
     +-- ACT → SkillDispatcher.dispatch_skill()
```

### 7. The "person_approaching" problem -- YOLO gives bbox, not velocity

YOLO does not output velocity. But we can derive approach/retreat from consecutive frames:

**Method**: ReactionEngine stores `_prev_distances: dict[str, float]` (track_id -> last distance_m). On each event, compare current distance to stored distance. If current < previous by >0.3m, person is approaching. If current > previous by >0.3m, person is retreating. Otherwise stationary.

**Limitation**: distance_m comes from depth estimation, which may be None on S100P if depth sensor is not wired. When distance_m is None, `person_approaching` defaults to False and the robot falls back to bbox-size heuristic: if bbox area increased by >15%, person is likely approaching.

### 8. The "lingering person" detection

This is not a new event type -- it is a **timed re-evaluation** of an existing tracked object. ReactionEngine runs a lightweight tick every ~10s (piggybacking on the existing ChangeDetector read interval) that checks:

```python
for person in world_state.get_persons_sync():
    if person.duration_s > 120 and self._not_already_assisted(person.track_id):
        # Synthesize an ASSIST reaction without a new ChangeEvent
        ctx = self._build_context_for_tracked(person)
        reaction = self._evaluate_rules(ctx)
        await self._execute(reaction)
```

This reuses WorldState's `TrackedObject.duration_s` at `world_state.py:58` which is already computing the value.

## Root Cause

The fundamental issue is that `_handle_change_event()` at `proactive_agent.py:459` treats all person events identically -- no scene context, no rule differentiation, no memory recall. The fix is not more if-statements in that method, but a separate decision engine that receives the full SceneContext.

## Recommendations

1. **Create `askme/pipeline/reaction_engine.py`** with `ReactionEngine` class, `SceneContext` dataclass, `ReactionType` enum, and the rule matrix. -- Medium effort, high impact. This is the core new work.

2. **Expose WorldState from PerceptionModule** by adding `self.world_state = WorldState()` and feeding it from ChangeDetector. Wire `apply_event_sync()` calls in `ChangeDetector._emit_events()`. -- Low effort, enables all WorldState-based decisions.

3. **Replace `ProactiveAgent._handle_change_event()`** with a two-line delegation to ReactionEngine. Keep existing `_patrol_tick()` and `_event_monitor_loop()` untouched. -- Low effort, surgical change.

4. **Add zone config** under `proactive.reaction.zones` in config.yaml. On S100P, populate from LingTu waypoint names. -- Low effort, enables location-aware decisions.

5. **Add lingering-person tick** as a periodic check in ReactionEngine (every 10s, scan WorldState for persons with duration_s > threshold). -- Low effort, addresses the "standing 5 minutes" use case.

6. **Wire LLM content generation** for GREET/ASSIST reactions only, using MiniMax M2.7 with a 150-token prompt. Template fallback when LLM is unavailable or times out. -- Medium effort, makes greetings contextual.

## Trade-offs

| Option | Pros | Cons |
|--------|------|------|
| **Rule-first (this design)** | Deterministic, <1ms decision, no LLM cost for 80% of events, works offline, debuggable | Cannot handle truly novel situations; rule maintenance as scenarios grow |
| **LLM-first (call LLM for every event)** | Maximum flexibility, handles edge cases | 1-3s latency per event, ~$0.001/event cost, 4GB RAM pressure on S100P, fails when LLM unavailable |
| **Hybrid (this design: rules decide, LLM speaks)** | Best of both: fast decision + contextual speech | Two codepaths to maintain; LLM fallback needed when content gen fails |
| **No WorldState dependency (pure event-based)** | Simpler, no shared mutable state | Cannot detect lingering, cannot check person count, loses rich context |
| **New Module vs component in ProactiveModule** | Module: cleaner separation, own In/Out ports | Module: heavier, adds topology complexity for what is a single-consumer component |

The recommended path is the **hybrid approach as a component inside ProactiveModule**, not a separate Module. This keeps the topology unchanged (no new wiring), minimizes risk, and the ReactionEngine can be promoted to a full Module later if other modules need to consume reaction decisions.

## References

- `askme/pipeline/proactive_agent.py:459-487` -- current `_handle_change_event()`, the method to replace
- `askme/pipeline/proactive_agent.py:365-381` -- `_speak_alert()`, reused via AlertDispatcher
- `askme/perception/world_state.py:45-59` -- TrackedObject with distance_m, duration_s, bbox
- `askme/perception/world_state.py:195-202` -- `get_persons_sync()`, key accessor for reaction decisions
- `askme/perception/change_detector.py:260-324` -- debounce logic, confirm_frames/disappear_frames
- `askme/perception/attention_manager.py:90-120` -- `should_alert()`, already does cooldown + threshold
- `askme/schemas/events.py:15-20` -- ChangeEventType enum, the 5 event types
- `askme/schemas/events.py:98-112` -- `description_zh()`, the hardcoded templates to be replaced
- `askme/memory/episodic_memory.py:227-245` -- `retrieve(query)` for "seen_person_recently" check
- `askme/memory/episodic_memory.py:378-406` -- `get_recent_digest()` for LLM content generation context
- `askme/memory/site_knowledge.py:88-177` -- SiteKnowledge with Location, find_nearby(), zone support
- `askme/pipeline/alert_dispatcher.py:62` -- AlertDispatcher, reused as-is for all output channels
- `askme/pipeline/skill_dispatcher.py:40-79` -- MissionContext/MissionState, reused for ACT reactions
- `askme/runtime/modules/proactive_module.py:32-58` -- build(), where ReactionEngine gets wired
- `askme/runtime/modules/perception_module.py:33-53` -- build(), where WorldState should be added
- `askme/schemas/observation.py:13-28` -- Detection with bbox, distance_m, center property
- `config.yaml:297-310` -- existing proactive config section to extend
- `docs/PROACTIVE_INTELLIGENCE_PLAN.md:46-78` -- Layer 2 plan, which this design implements