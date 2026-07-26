# Conversation Core

Conversation Core 定义机器人跨语音、文本和多模态渠道的对话连续性。它提供统一的交互语言，使连接恢复、轮次提交、长期记忆、视觉证据和任务执行不会被混为同一个概念。

本文是目标领域语言，不等于所有对象都已落地。Phase 1 已实现 Thread、Turn、Generation 和 Provider Session 关联；Person 聚合、Conversation Summary、跨日期 Session Window，以及 Memory/Vision/Task 事件消费者仍是后续能力。

## Language

### Identity

**Person（交互主体）**:
可以跨日期、设备和对话持续识别的自然人，是对话连续性的主体。
_Avoid_: User, Speaker, Customer, Session Owner

**Operator（操作员）**:
Person 在特定客户项目、站点或机器人上的授权操作角色。
_Avoid_: Person, Administrator, Session

### Conversation Lifecycle

**Conversation Thread（对话线程）**:
Person 感知到的一段连续逻辑对话，可以跨越多次连接、设备切换和不同日期。
_Avoid_: Session, Provider Session, Chat Session, Socket

**Turn（轮次）**:
从一份确认后的用户输入开始，到机器人内容交付、取消或失败结束的一次完整交互，是对话的最小业务与审计单元。
同一 Conversation Thread 同时最多有一个非终态 Turn；供应商重试属于该 Turn 下的新 Generation。不同 Turn 撞上活动 Turn 时应排队、取消旧 Turn 或明确拒绝，不能静默交错提交。
_Avoid_: Message, Utterance, Request, Generation

**Generation（生成尝试）**:
同一 Turn 内的一次候选响应生成；重试或供应商恢复可以产生多个 Generation，但未交付内容不构成正式 Turn 内容。
_Avoid_: Turn, Response, Provider Session

**Realtime Provider Session（实时供应商会话）**:
与 ASR、LLM、TTS 或端到端语音供应商建立的一段临时实时连接。
_Avoid_: Conversation Thread, Conversation, History

**Conversation Summary（对话摘要）**:
对已提交 Turn 的可追溯压缩表达，用于降低上下文成本而不取代原始对话事实。
_Avoid_: Memory, History, Transcript, Source of Truth

### Context Consumers

**Memory Record（记忆记录）**:
从获准且已提交的交互中提炼出的可复用事实、偏好或经验，并保留来源关系。
_Avoid_: Conversation History, Thread, Message, Summary

**Visual Observation（视觉观察）**:
在特定时间与场景中由感知系统形成的可追溯视觉事实。
_Avoid_: Image, Video, Memory, Visual Artifact

**Visual Artifact（视觉制品）**:
支撑视觉观察的原始或派生媒体证据，例如图像、视频片段或标注结果。
_Avoid_: Observation, Memory, Conversation

**Task（任务）**:
由对话或其他事件触发、具有明确目标和结果边界的一项受控工作。
_Avoid_: Tool Call, Turn, Intent, Mission

**Mission（任务编排）**:
为一个现场目标组织起来的一组相关 Task 及其进度关系。
_Avoid_: Task, Conversation Thread, Workflow Session

**Consent Grant（同意授权）**:
Person 对特定记忆、视觉使用或敏感动作给予的有范围、可撤销授权。
_Avoid_: Blanket Consent, Login, Role, Confirmation
