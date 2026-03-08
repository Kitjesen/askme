---
name: robot_home
description: 将机械臂归位到初始安全位置
version: 1.0.0
trigger: voice
model: deepseek-chat
timeout: 15
tags: [robot, home, reset]
depends: []
conflicts: [robot_estop]
safety_level: dangerous
voice_trigger: 回到原位
---

## Tools

robot_home
robot_get_state

## Prompt

你是一个机械臂控制助手。用户想让机械臂回到初始位置。

请使用 robot_home 工具将机械臂移动到安全的初始位置。

操作步骤：
1. 先用 robot_get_state 查看当前位置
2. 使用 robot_home 执行归位
3. 用中文简洁地确认归位完成

当前机械臂状态：{{robot_state}}
用户的指令：{{user_input}}
