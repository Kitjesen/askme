# 小穹 — ZeroClaw 实验工具清单

> 以下是通过 Askme MCP Server 暴露的工具和能力。

---

## 机器人控制

| 工具 | 描述 | 参数 |
|------|------|------|
| `robot_move` | 移动机械臂到目标位置（毫米坐标） | `x`, `y`, `z` |
| `robot_pick` | 闭合夹爪拾取物体 | `target` |
| `robot_place` | 张开夹爪释放物体 | `location` |
| `robot_home` | 机械臂回到归零/休息位 | 无 |
| `robot_wave` | 执行挥手动作 | 无 |
| `robot_state` | 查询机械臂当前状态（关节角、连接、急停） | 无 |
| `robot_estop` | **紧急停止** — 立即停止所有机械臂运动 | 无 |

## 语音交互

| 工具 | 描述 | 参数 |
|------|------|------|
| `voice_listen` | 录制一段语音并转写为文字 | 无 |
| `voice_speak` | 将文本合成为语音并播放 | `text` |

## 视觉感知

| 工具 | 描述 | 参数 |
|------|------|------|
| `look_around` | 描述当前场景，可指定问题聚焦 | `question` (可选) |
| `find_target` | 在当前视野中搜索指定物体 | `target` |

## 记忆系统

| 工具 | 描述 | 参数 |
|------|------|------|
| `memory_search` | 跨所有层级搜索记忆 | `query`, `n`, `layer` |
| `memory_save` | 保存一条事实到长期记忆 | `text`, `source` |

## 技能执行

| 工具 | 描述 | 参数 |
|------|------|------|
| `execute_skill` | 执行 SKILL.md 中定义的一个技能 | `skill_name`, `user_input` |

## 通用对话

| 工具 | 描述 | 参数 |
|------|------|------|
| `chat` | 通过稳定 MCP 上下文发送文本对话 | `text` |

---

## 资源（只读查询）

| 资源 URI | 描述 |
|----------|------|
| `askme://health` | 服务器健康状态、版本、子系统运行情况 |
| `askme://config` | 当前配置（API Key 已脱敏） |
| `askme://skills` | 所有可用技能目录 |
| `askme://skills/openapi` | 技能合约的 OpenAPI 文档 |
| `askme://contracts/io` | 产品 I/O 合约（感知、意图、动作、UI） |
| `askme://contracts/examples` | 合约示例负载 |
| `askme://perception/detections` | 当前帧检测结果 |
| `askme://perception/events` | 最近的感知事件流 |
| `askme://perception/depth` | 深度相机状态和中心深度值 |
| `askme://memory/knowledge` | 长期知识文件清单 |
| `robot://status` | 机械臂连接状态、模式、急停状态 |
| `robot://joint/{id}/state` | 指定关节的静态信息（名称、限位） |
| `robot://safety/config` | 安全系统配置（关节限位、速度限制、急停关键词） |
