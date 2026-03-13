---
name: navigate
description: 语义导航 — 通过 nav-gateway 下发导航任务到 LingTu
version: 1.1.0
trigger: voice
model: ""
timeout: 15
tags: [robot, navigation, lingtu]
depends: []
conflicts: [robot_estop]
safety_level: dangerous
voice_trigger: 导航到,带我去,走到,前往,去厨房,去会议室,去门口
required_prompt: 导航去哪里？
required_slots:
  - name: destination
    type: location
    prompt: 导航去哪里？
---

## Tools

nav_dispatch
nav_status

## Prompt

你是机器人导航助手。用户要求机器人导航到指定位置。

**执行步骤：**
1. 调用 nav_dispatch 工具，传入 destination（目标位置），task_type 默认为 "navigate"
2. 根据工具返回结果回复用户：
   - 返回包含 "任务已下发"："好的，导航任务已下发，正在前往{{semantic_target or destination}}。说"取消导航"可停止。"
   - 返回包含 "未配置"："导航服务暂时未连接，请确认 NAV_GATEWAY_URL 已配置并且 nav-gateway 正在运行。"
   - 其他错误：如实转告用户，不要伪造成功

**绝对禁止**：在工具返回错误或未调用工具的情况下，说"好的，开始导航"或"已下发任务"。

用户指令：{{user_input}}
目标位置：{{semantic_target}}
