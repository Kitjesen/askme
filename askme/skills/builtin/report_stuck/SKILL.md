---
name: report_stuck
description: "卡住无法运动：记录现场、通知保安或运维并归档"
version: 1.0.0
trigger: voice
model: ""
timeout: 30
tags: [field, incident, robot_fault, security]
depends: []
conflicts: []
safety_level: dangerous
confirm_before_execute: true
voice_trigger: "卡住无法运动,机器狗卡住了,机器人动不了,报告卡住事故"
required_prompt: "卡住发生在哪个位置？"
required_slots:
  - name: location
    type: location
    prompt: "卡住发生在哪个位置？"
---

## Tools

field_event_trigger

## Prompt

你是园区机器狗现场事件处置助手。用户报告机器狗卡住或无法运动时，进入事件闭环。

必须调用 `field_event_trigger`，参数：
- scenario_id: robot_abnormal_incident
- fault_type: immobilized
- location: 从用户输入或 {{semantic_target}} 提取，缺失时用“未知位置”
- operator_id: dashboard.operator
- description: 说明“机器狗卡住无法运动，需要保安/运维到场处理”

回复只说结果，不承诺已经自行脱困。若工具返回缺少证据，要求补充现场照片或位置。

用户输入：{{user_input}}
当前位置/目标：{{semantic_target}}
