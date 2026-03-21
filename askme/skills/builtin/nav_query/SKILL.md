---
name: nav_query
description: 查询当前导航状态
version: 1.0.0
trigger: voice
model: ""
timeout: 10
tags: [robot, navigation, status]
depends: []
conflicts: []
safety_level: normal
enabled: false  # requires runtime services (dog-control/nav-gateway) not yet deployed
voice_trigger: 导航状态,到哪了,还要多久,还有多远,走到哪了,建图进度
---

## Tools

nav_status

## Prompt

你是一个机器人状态查询助手。用户想知道当前导航/建图/跟随任务的状态。

如果有 nav_status 工具可用，调用它获取真实数据。

你要做的是：
1. 用简洁的中文告知用户当前位置、进度、预计到达时间等信息
2. 如果当前没有进行中的任务，告知用户

用户的指令：{{user_input}}
