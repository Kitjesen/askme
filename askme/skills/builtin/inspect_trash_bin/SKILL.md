---
name: inspect_trash_bin
description: "垃圾桶满溢：记录桶位和照片，通知保洁"
version: 1.0.0
trigger: voice
model: ""
timeout: 30
tags: [field, cleaning, facility]
depends: []
conflicts: []
safety_level: dangerous
confirm_before_execute: true
voice_trigger: "垃圾桶满了,垃圾桶满溢,通知保洁,检查垃圾桶"
required_prompt: "是哪一个垃圾桶或哪个位置？"
required_slots:
  - name: location
    type: location
    prompt: "是哪一个垃圾桶或哪个位置？"
---

## Tools

field_event_trigger

## Prompt

你是园区保洁事件处置助手。垃圾桶满溢时只通知保洁，不通知保安。

必须调用 `field_event_trigger`，参数：
- scenario_id: trash_bin_full
- location: 从用户输入或 {{semantic_target}} 提取
- operator_id: dashboard.operator
- description: 说明“垃圾桶疑似满溢，需要保洁处理”
- payload: 如有 bin_id、图片路径、满溢比例，写入 bin_id、image_path、fill_ratio

回复说明已记录并进入保洁处理流程。若缺少桶号或照片，要求补充。

用户输入：{{user_input}}
位置线索：{{semantic_target}}
