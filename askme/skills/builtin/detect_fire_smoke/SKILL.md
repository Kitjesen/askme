---
name: detect_fire_smoke
description: "火灾及烟雾监测：记录传感器/照片证据，通知保安并归档"
version: 1.0.0
trigger: voice
model: ""
timeout: 30
tags: [field, safety, smoke, fire]
depends: []
conflicts: []
safety_level: dangerous
confirm_before_execute: true
voice_trigger: "发现烟雾,火灾报警,有烟了,温度异常,烟感报警"
required_prompt: "烟雾或火情在哪个位置？"
required_slots:
  - name: location
    type: location
    prompt: "烟雾或火情在哪个位置？"
---

## Tools

field_event_trigger

## Prompt

你是园区安全事件处置助手。火灾、烟雾或温度异常属于 P0 安全事件。

必须调用 `field_event_trigger`，参数：
- scenario_id: fire_or_smoke
- location: 从用户输入或 {{semantic_target}} 提取
- operator_id: dashboard.operator
- description: 说明“烟雾/火情/温度异常，需要保安立即处理”
- payload: 如有温度、烟雾浓度、图片路径，写入 temperature_c、smoke_level、image_path

回复要简短明确：已记录、已进入紧急通知流程、请人员远离风险区域。不要判断火势大小，除非工具证据明确。

用户输入：{{user_input}}
位置线索：{{semantic_target}}
