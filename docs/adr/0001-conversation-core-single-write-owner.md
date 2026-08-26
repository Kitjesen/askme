# Conversation Core 是会话事实的唯一写入者

状态：Accepted target；Phase 1 partial implementation（2026-07-19）。

机器人语音链路此前可能由网关、供应商会话、Pipeline 和 Memory 分别保存相似但不一致的历史。目标架构决定将 Conversation Core 设为 Conversation Thread、Turn、Generation 的唯一写入者，并在后续加入 Conversation Summary 投影。只有单一提交边界才能在打断、重试、断线恢复和供应商切换时保证“用户最终说了什么、机器人实际交付了什么”一致可审计。

## Consequences

- Gateway 和 Provider 只通过命令、事件及关联 ID 参与对话；Realtime Provider Session 可以被替换，不等同于 Conversation Thread，也不能直接提交历史。
- Pipeline 负责执行编排，但 Turn 的生命周期和最终提交由 Conversation Core 记录。
- Phase 1 已实现 Thread/Turn/Generation 的本地单进程 JSONL 账本、重放、alias 归一、Generation 替换和主要语音结算路径。
- 账本将“同一 Thread 至多一个非终态 Turn”作为领域不变量：同一 `turn_id` 可幂等恢复，不同 Turn 冲突会 fail-closed；HTTP 表面返回 409。语音 barge-in 必须先结算旧 Turn，再开启新 Turn。
- Conversation Summary 仍在 Memory/Voice Gateway 兼容投影中；Memory、Perception/Vision 和 Task/Mission 的 committed-event consumer、checkpoint 与证据 ID 回链是后续工作。
- 迁移期将 `conversation_session_id`、`conversation_id`、`chat_session_id` 等旧字段视为 `thread_id` 的兼容别名，逐步停止旧路径的新增双写。
- 当前账本依赖进程内锁和 fsync，只支持本地单进程 writer；多进程/分布式存储、outbox 和 consumer checkpoint 尚未实现。
- Phase 1 的跨入口冲突策略是显式拒绝并重试；跨本地/实时/runtime 的共享排队 lease 尚未实现。
- `ThreadStatus.ERASED` 目前只生成逻辑脱敏 tombstone；原始 JSONL 和旧历史的物理擦除或加密销钥尚未实现，不能据此声称满足合规删除。
