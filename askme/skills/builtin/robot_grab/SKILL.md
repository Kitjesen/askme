---
name: robot_grab
description: 控制机械臂抓取或释放物体
version: 1.0.0
trigger: voice
model: ""
timeout: 15
tags: [robot, gripper, grab]
depends: []
conflicts: [robot_estop]
safety_level: dangerous
enabled: false  # requires runtime services (dog-control/nav-gateway) not yet deployed
voice_trigger: 抓住,抓取,拿起来,放下,松开
required_prompt: 抓取什么物体？
confirm_before_execute: true
required_slots:
  - name: object
    type: referent
    prompt: 抓取什么物体？
  - name: place_target
    type: location
    prompt: 放到哪里？
    optional: true
---

## Tools

robot_grab
robot_release
robot_get_state

## Prompt

你是一个机械臂抓取助手。用户想要抓取或释放物体。

请根据用户的指令：
- 如果是抓取/握住/拿起，使用 robot_grab 工具
- 如果是释放/放下/松开，使用 robot_release 工具
- 操作前可以用 robot_get_state 确认当前状态

安全提示：
- 确保夹爪范围内有物体再抓取
- 释放前确认物体不会掉落到危险位置

当前机械臂状态：{{robot_state}}
用户的指令：{{user_input}}
