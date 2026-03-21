---
name: nav_cancel
description: 取消当前导航/建图/跟随任务
version: 1.0.0
trigger: voice
model: ""
timeout: 10
tags: [robot, navigation, cancel]
depends: []
conflicts: []
safety_level: normal
enabled: false  # requires runtime services (dog-control/nav-gateway) not yet deployed
voice_trigger: 取消导航,停止导航,取消建图,停止建图,取消跟随,停止跟随
---

## Tools

nav_status

## Prompt

你是一个机器人控制助手。用户要取消当前的导航/建图/跟随任务。

取消请求已通过 runtime 的 arbiter 下发到 nav-gateway，
nav-gateway 会将取消指令转发到 LingTu 导航系统执行。

你不需要直接控制——那是 runtime 的工作。

你要做的是：
1. 用简洁的中文确认已取消当前任务

用户的指令：{{user_input}}
