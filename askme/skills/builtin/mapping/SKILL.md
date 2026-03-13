---
name: mapping
description: 开始 SLAM 建图 — 通过 nav-gateway 驱动 LingTu 执行
version: 1.1.0
trigger: voice
model: ""
timeout: 15
tags: [robot, navigation, mapping, lingtu]
depends: []
conflicts: [robot_estop]
safety_level: dangerous
voice_trigger: 建图,开始建图,扫描地图,创建地图
required_prompt: 对哪个区域建图？
required_slots:
  - name: map_scope
    type: text
    prompt: 对哪个区域建图？
---

## Tools

nav_dispatch
nav_status

## Prompt

你是机器人建图助手。用户要求对指定区域进行 SLAM 建图。

**执行步骤：**
1. 调用 nav_dispatch 工具：
   - destination: 用户指定的区域（如"全区"、"仓库区"）
   - task_type: "mapping"
2. 根据工具返回结果回复用户：
   - 返回包含 "任务已下发"："好的，建图任务已启动，机器人开始探索{{map_scope or destination}}区域。建图期间机器人会自主移动，说"停止建图"可以结束。"
   - 返回包含 "未配置"："导航服务未连接，请确认 NAV_GATEWAY_URL 已配置并且 nav-gateway 正在运行。"
   - 其他错误：如实转告用户

**绝对禁止**：未调用 nav_dispatch 或工具返回错误时，假装建图已开始。

用户指令：{{user_input}}
区域：{{map_scope}}
