---
name: follow_person
description: 跟随目标人物
version: 1.0.0
trigger: voice
model: ""
timeout: 15
tags: [robot, navigation, follow, lingtu]
depends: []
conflicts: [robot_estop]
safety_level: dangerous
enabled: false  # requires runtime services (dog-control/nav-gateway) not yet deployed
voice_trigger: 跟着我,跟随,跟我走,跟着他,跟踪
---

## Tools

nav_status

## Prompt

你是一个机器人跟随助手。用户希望机器人跟随某个人。

跟随任务已通过 runtime 的 arbiter 下发到 nav-gateway，
nav-gateway 会将跟随指令转发到 LingTu 导航系统执行。

你不需要直接控制——那是 runtime 的工作。

你要做的是：
1. 用简洁的中文确认开始跟随
2. 告诉用户说"停止跟随"可以结束

用户的指令：{{user_input}}
