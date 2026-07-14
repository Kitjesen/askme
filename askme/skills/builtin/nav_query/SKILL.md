---
name: nav_query
description: 查询当前导航状态
version: 1.0.0
trigger: voice
execution: read_only_tool
model: ""
timeout: 10
tags: [robot, navigation, status]
depends: []
conflicts: []
safety_level: normal
enabled: true
voice_trigger: 当前位置,我在哪里,你在哪里,导航状态,到哪了,还要多久,还有多远,走到哪了,建图进度
---

## Tools

nav_status

## Prompt

这是只读状态查询，只允许调用 nav_status，不得下发导航、移动或其他动作。
直接返回 nav_status 生成的中文结果；定位未就绪时明确告知用户，不得猜测坐标。
