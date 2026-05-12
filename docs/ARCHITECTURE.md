# askme Architecture

更新时间：2026-05-10

本文是 askme 当前唯一的架构入口。

## 一句话架构

```text
User speech/text
  -> Interaction Gate
  -> ASR / text input
  -> LLM + Memory + RAG policy
  -> Cognition Planner
  -> TaskHandoff
  -> SafetyPreflight
  -> Runtime Arbiter
  -> fake/sim/shadow/lab runtime
  -> TaskReport + Audit evidence
```

核心原则：LLM 只负责理解、规划、解释和交互；硬件动作必须由 runtime、安全服务和机器人控制系统负责。

## 运行时模块

askme 使用 declarative runtime module 组合。主要模块：

| 模块 | 职责 |
| --- | --- |
| `LLMModule` | LLM client、模型健康、延迟指标 |
| `MemoryModule` | MemoryBridge、KnowledgeCatalog、RAG 检索与导入 |
| `PipelineModule` | 文本/语音 turn 执行链路 |
| `VoiceModule` | ASR、VAD、TTS、VoiceLoop、InteractionGate |
| `CognitionModule` | WorldState、WorkingMemory、CognitivePlanner、ActivePerceptionResolver |
| `RuntimeHandoffModule` | TaskHandoff、TaskRun、runtime profile、pause/resume/cancel/advance |
| `HealthModule` | Dashboard、HTTP API、health snapshot、readiness evidence |
| `SkillModule` | 工具/技能注册、SkillGate、安全边界 |

## 语音链路

推荐国产低延迟链路：

```text
Realtime ASR
  -> MiniMax-M2.7-highspeed
  -> askme TaskHandoff / SafetyPreflight / runtime arbiter
  -> MiniMax Speech 2.8 TTS
```

语音入口必须遵守同一套状态机：

- “确认”在 planning 阶段确认计划。
- “取消”在 planning 阶段取消草案，在 executing 阶段取消 TaskRun。
- “停下”走安全优先路径。
- 语音误识别不能绕过安全确认。

## Interaction Gate

Interaction Gate 是真实场景的准入门。它不会把所有人声都送进 LLM。

输入：

- ASR final transcript 与 confidence。
- 是否被明确呼叫。
- 视觉注意力、距离、姿态、手势。
- 声源方向、声画一致性。
- 多人仲裁结果。
- 感知 freshness。

输出：

- `respond`：进入大脑回复或规划。
- `clarify`：先澄清说话对象或意图。
- `record_only`：只记录环境，不回复。
- `ignore`：忽略。
- `refuse`：安全/隐私拒绝。

关键规则：

- 旁观者说“这个机器狗好可爱”不等于唤醒。
- 多人且 speaker lock 不清楚时澄清。
- 声源和画面人物不一致时不猜。
- stop/emergency intent 优先安全路径。

## RAG 与知识生命周期

KnowledgeCatalog 是可信知识事实源。Memory backend 只是检索实现，不是最终信任来源。

知识状态：

- `draft`
- `pending`
- `approved`
- `published`
- `rejected`
- `deleted`
- `conflicted`

硬约束：

- expired 不进入 prompt。
- draft/pending/rejected/deleted 不进入 prompt。
- 同一 `entity_key + fact_key` 出现互斥 `value` 时进入 conflict。
- 检索命中后必须用 `record_id + evidence_version` 回 catalog 二次校验。
- `answer_policy` 会随回答返回，约束 LLM 不用无证据、过期、冲突知识编答案。

## 任务运行链路

任务对象分层：

- `TaskPlan`：用户想做什么。
- `TaskHandoff`：交给 runtime 的结构化计划。
- `TaskRun`：这一次实际执行发生了什么。
- `RuntimeEvent`：状态变化、step 事件、operator action。
- `TaskReport`：完成或失败后的结构化结果。

典型状态：

```text
draft
  -> awaiting_confirmation
  -> ready_for_arbiter
  -> submitted
  -> validating
  -> preflight
  -> queued
  -> executing
  -> paused / blocked / completed / failed / cancelled
```

Profile：

- `fake`：本地演示。
- `sim`：可手动 advance 的模拟运行。
- `shadow`：只做将要执行什么的验证，不发硬件动作。
- `lab`：受控实验室，默认禁用。
- `prod`：生产，默认禁用。

external/lab runtime 必须显式配置 endpoint 和 enable flag。默认不会联网或触碰硬件。

## 感知与 WorldState

WorldState 是 planner 和 safety 的事实来源。感知快照要带 `observed_at` 和 freshness。

当前支持的 interaction perception 字段：

- person detected/count/distance/angle。
- visual attention。
- person facing robot。
- posture。
- gesture。
- sound source angle。
- sound source matches person。

仍待接入真实 provider：

- 姿态/视线估计。
- 手势识别。
- 麦克风阵列 DOA。
- 声画关联。
- 接近/停留追踪。
- 多人仲裁。

## 安全不变量

- 不直接发 motor/gait/arm/serial/cmd_vel。
- 不绕过 runtime arbiter。
- 不绕过 SafetyPreflight。
- 不用 stale/conflict/unapproved knowledge 驱动高风险任务。
- operator action 必须记录 actor、reason、risk acknowledgement。
- lab/prod 必须显式启用，默认安全。

## 主要文件

| 文件 | 作用 |
| --- | --- |
| `askme/voice/interaction_gate.py` | 交互准入门 |
| `askme/voice/perception_context.py` | 感知快照归一化 |
| `askme/memory/catalog.py` | 知识生命周期事实源 |
| `askme/runtime/modules/memory_module.py` | 知识导入、检索、重建、批量更新 |
| `askme/cognition/active_perception.py` | 缺事实时主动刷新感知 |
| `askme/runtime/handoff.py` | TaskHandoff、TaskRun、runtime state machine |
| `askme/runtime/arbiter_client.py` | external/lab contract-only client |
| `askme/runtime/modules/health_module.py` | HTTP/Dashboard wiring 与 evidence report |
| `askme/static/dashboard.html` | Voice Mission Center |
