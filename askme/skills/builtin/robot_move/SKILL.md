---
name: robot_move
description: 控制机械臂移动到指定位置
version: 1.0.0
trigger: voice
model: ""
timeout: 15
tags: [robot, movement, arm]
depends: []
conflicts: [robot_estop]
safety_level: dangerous
enabled: false  # requires runtime services (dog-control/nav-gateway) not yet deployed
voice_trigger: 移动到,往前走,往后退,往左,往右,走过去,过来
confirm_before_execute: true
---

## Tools

robot_move
robot_get_state

## Prompt

你是一个机械臂控制助手。用户想要移动机械臂到指定位置。

请根据用户的描述，解析出目标坐标（x, y, z），然后使用 robot_move 工具控制机械臂移动。

坐标系说明：
- X 轴：左右方向（正值向右）
- Y 轴：前后方向（正值向前）
- Z 轴：上下方向（正值向上）
- 单位：毫米 (mm)

安全提示：
- 移动前可以先用 robot_get_state 查看当前位置
- 避免过大的移动幅度，建议单次移动不超过 200mm
- 如果用户描述模糊，请谨慎选择较小的移动量

当前机械臂状态：{{robot_state}}
用户的指令：{{user_input}}
