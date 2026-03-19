# Proactive Clarification & Task Completion System

> 最后更新：2026-03-13
> 当前状态：**Phase 2 完成，Phase 3 架构就绪**

---

## 系统定位

把 askme 从"命令匹配器"推向"初级任务代理"。核心能力：

**用户说得不完整时，系统主动把任务补完整，以最短交互代价把事做成。**

---

## 演进路线

```
Phase 1 (基础槽位) ──→ Phase 2 (类型化 + 模糊检测 + 多Agent) ──→ Phase 3 (策略性主动) ──→ Phase 4 (复合任务)
    ✅ 完成                      ✅ 完成                              ⬜ 架构就绪                  ⬜ 未开始
```

---

## Phase 1：基础槽位追问

**完成时间：2026-03-13（本次 session 早期）**
**测试：** `tests/test_voice_loop_slot.py` — 24 个测试

### 实现内容

- `required_prompt` 字段加入 `SkillDefinition`
- `VoiceLoop._maybe_collect_slot()` — 触发技能前检查参数
- `_slot_present()` — 比较触发词去除后的剩余内容长度
- 受益技能：navigate、mapping、robot_grab、web_search

### 已知局限

- `_slot_present` 只做长度检查，"搜索一下那个"（6字）会误判为已填充
- 没有 retry，一次问完即放行
- 没有记忆提示（本次是否与上次相同）

---

## Phase 2：多 Agent + 类型化槽位 + 模糊检测

**完成时间：2026-03-13**
**新增测试：**
- `tests/test_proactive_agents.py` — 28 个测试
- `tests/test_voice_loop_slot.py` — 24 个测试（已有，升级 proxy）
- `tests/test_slot_analyst.py` — 19 个测试
- `tests/test_clarification_agent.py` — 17 个测试

### 系统架构

```
VOICE_TRIGGER
    ↓
ProactiveOrchestrator
    ├── ClarificationPlannerAgent   (类型化槽位 + 模糊检测 + 多槽合并)
    ├── SlotCollectorAgent          (旧 required_prompt 后备，向后兼容)
    └── ConfirmationAgent           (危险动作二次确认)
    ↓
dispatcher.dispatch()
```

### 新能力一览

#### 类型化槽位（`SlotSpec`）

```yaml
required_slots:
  - name: destination
    type: location          # 会用 extract_semantic_target 提取
    prompt: 导航去哪里？
  - name: place_target
    type: location
    prompt: 放到哪里？
    optional: true          # 可选槽位不阻塞执行
```

支持类型：`text` | `location` | `referent` | `datetime` | `enum`

#### 模糊占位词检测（30+ 词）

```
"搜索一下那个"  → 剩余="那个"  → is_vague=True  → 追问
"搜索一下北京"  → 剩余="北京"  → is_vague=False → 执行
"抓取那件"      → 值="那件"    → is_vague=True  → 追问
```

完整词表：那个/这个/那里/那件/一下/看看/查查/某处/随便/都行/…

#### 多槽位合并问法

```
缺 object + place_target → "抓取什么物体，放到哪里？"   ✅ 一句话
缺 destination           → "导航去哪里？"
缺 query                 → "搜索什么内容？"
```

#### 记忆提示

```
当前 mission 中上次 navigate 去了"仓库B"
→ 再次说"导航"时询问："上次是仓库B，这次还是吗？或者导航去哪里？"
```

#### 危险动作确认（`confirm_before_execute`）

```yaml
confirm_before_execute: true  # robot_grab, robot_move
```

```
用户: "抓取红色瓶子"
系统: "即将执行机械臂抓取，说'确认'继续，说'取消'停止。"
用户: "确认" → dispatch
用户: "取消" / 沉默 → 取消（安全默认）
```

### 已接入技能

| 技能 | required_slots | confirm_before_execute |
|------|----------------|------------------------|
| navigate | destination (location) | — |
| web_search | query (text) | — |
| robot_grab | object (referent) + place_target? (location) | ✅ |
| robot_move | — | ✅ |
| mapping | map_scope (text) | — |

---

## Phase 3：策略性主动（架构就绪，未实现）

**目标：** 不只是"缺槽就问"，而是"主动推进任务"

### 设计中的能力

| 能力 | 触发条件 | 示例 |
|------|----------|------|
| 默认补全 | 有 `default` 值 + 场景唯一 | "查天气" → 默认查今天 |
| 候选确认 | 多个匹配候选 | "控制室" 有 2 个 → 列出让用户选 |
| 安全提醒 | runtime 状态异常 | 电量 <20% → 建议先充电 |
| 模板推荐 | 已有匹配任务模板 | "A 区巡检" → 直接用标准模板 |

### 需要的新 Agent：`ProactivePolicyAgent`

```python
class ProactivePolicyAgent(ProactiveAgent):
    """查 runtime 状态后做策略决策：默认/确认/警告/阻止"""

    def should_activate(self, skill, user_text, context):
        # 只对有 runtime 副作用的技能激活
        return skill.safety_level != "normal"

    async def interact(self, skill, user_text, audio, context):
        # 1. 查电量 / 设备状态 / 任务队列
        # 2. 按策略决定是否警告/确认/替换
```

### 需要的 runtime 接口

- `GET /api/v1/robot/status` → 电量、运动状态
- `GET /api/v1/nav/candidates?q={target}` → 目标点候选列表
- `GET /api/v1/missions/templates` → 已有任务模板

---

## Phase 4：复合任务编排（未开始）

**目标：** 一句话拆成多个技能串联

```
"去 A 区巡检一下，有异常拍照发我"
    → navigate(A区) → inspect(A区巡检) → on_anomaly(capture+notify)
```

需要：TaskPlanner、MultiSkillOrchestrator、状态机

---

## 测试覆盖矩阵

| 测试类型 | 文件 | 数量 | 状态 |
|---------|------|------|------|
| 槽位存在性检测 | `test_voice_loop_slot.py` | 24 | ✅ |
| 类型化槽位分析 | `test_slot_analyst.py` | 19 | ✅ |
| ClarificationPlannerAgent | `test_clarification_agent.py` | 17 | ✅ |
| SlotCollectorAgent / ConfirmAgent / Orchestrator | `test_proactive_agents.py` | 28 | ✅ |
| 语义歧义 | `test_semantic_ambiguity.py` | — | ⬜ 待补 |
| 边界词 / 错误任务 | `test_boundary_cases.py` | — | ⬜ 待补 |
| 快慢通道分流 | `test_routing_fast_slow.py` | — | ⬜ 待补 |
| 失败恢复 | `test_failure_recovery.py` | — | ⬜ 待补 |
| 人机协同端到端 | `test_human_machine.py` | — | ⬜ 待补 |
| 长时稳定性 | `test_stability.py` | — | ⬜ 设计中 |
| API 合约 | `test_api_contracts.py` | — | ⬜ 设计中 |
| 真实视觉 vs stub | — | — | ⬜ 待接入摄像头 |

---

## 关键文件索引

```
askme/pipeline/proactive/
├── __init__.py             — 包入口，统一导出
├── base.py                 — ProactiveAgent ABC + ProactiveResult + ask_and_listen()
├── orchestrator.py         — ProactiveOrchestrator，链式运行三个 Agent
├── clarification_agent.py  — ClarificationPlannerAgent（类型化槽位 + 模糊 + 合并）
├── slot_agent.py           — SlotCollectorAgent（旧 required_prompt 后备）
├── confirm_agent.py        — ConfirmationAgent（确认/取消危险动作）
├── slot_analyst.py         — analyze_slots() + is_vague() + 模糊词表
├── slot_types.py           — SlotFill + SlotAnalysis 数据类
└── slot_utils.py           — slot_present()，VoiceLoop._slot_present 的后端

askme/skills/skill_model.py — SlotSpec + SkillDefinition.required_slots
askme/pipeline/voice_loop.py — 集成点：proactive.run() 替换旧 _maybe_collect_slot
```
