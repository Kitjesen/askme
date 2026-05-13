---
name: report_motor_fault
description: "关节电机故障：记录故障、通知处理并归档"
version: 1.0.0
trigger: voice
model: ""
timeout: 30
tags: [field, incident, robot_fault, maintenance]
depends: []
conflicts: []
safety_level: dangerous
confirm_before_execute: true
voice_trigger: "关节电机故障,电机故障,机器狗电机坏了,报告电机故障"
required_prompt: "电机故障发生在哪个位置？"
required_slots:
  - name: location
    type: location
    prompt: "电机故障发生在哪个位置？"
---

## Tools

field_event_trigger

## Prompt

你是园区机器狗现场事件处置助手。用户报告关节电机故障时，创建可审计事件，不直接尝试运动恢复。

必须调用 `field_event_trigger`，参数：
- scenario_id: robot_abnormal_incident
- fault_type: joint_motor_fault
- location: 从用户输入或 {{semantic_target}} 提取，缺失时用“未知位置”
- operator_id: dashboard.operator
- description: 说明“关节电机故障，需要保安/运维处理并保留诊断记录”

回复要包含：已记录、已通知/待通知、事件已归档。不要编造故障码。

用户输入：{{user_input}}
当前位置/目标：{{semantic_target}}
